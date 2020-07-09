#!/usr/bin/env python3
"""Command-line interface to compute tangent-space projectors either linearly for densly-sampled datasets, or using
auto-encoders for producing a densely-sampled embedding for sparsely-sampled datasets.
"""
import logging
import os
from argparse import Namespace

import h5py
import torch

import click
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from tqdm import tqdm
from scipy.spatial.distance import cdist
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from fairwasher.ae import VQVAE


def csints(string):
    """Split a comma separated string into a list of integers"""
    if string:
        return [int(elem) for elem in string.split(',')]
    else:
        return []


def full_data(dataset):
    """Get a 'batch' of the full dataset.
    This applies all transformations from the dataset, as opposed to simply accessing the raw data of the dataset
    itself.

    Parameters
    ----------
    dataset : obj:`torch.utils.data.Dataset`
        Dataset object to be used.

    Returns
    -------
    `torch.Tensor`
        Ordered Batch of whole `dataset` with all transformations applied.

    """
    return next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))


def batches(*args, batch_size=32):
    """Pure-python generator to get mini-batches from a set of some slice-indexable objects.

    Parameters
    ----------
    *args
        Slice-indexable objects which are iterated, must have the same lengths.
    batch_size : int, optional
        Size of the slices to take

    Yields
    ------
    tuple of object
        Slices of the `args` with at least size `batch_size`.

    """
    L = len(args[0])
    n = 0
    while n < L:
        k = n + batch_size
        if k > L:
            k = L
        yield tuple(obj[n:k] for obj in args)
        n = k


def bcdist(XA, XB, batch_size=32):
    """Compute batch-wise (over `XA`) the distance between each pair of the two collections for inputs.

    Parameters
    ----------
    XA : obj:`np.ndarray`
        First collection of inputs, over which batches will be used.
    XB : obj:`np.ndarray`
        Second collection of inputs.
    batch_size : int
        Size of the batches over which will be iterated on `XA`.

    Returns
    -------
    Y : obj:`np.ndarray`
        Pairwise distances between inputs of collections `XA` and `XB`, with shape XA.shape[0] x XB.shape[0].

    """
    Y = np.empty((XA.shape[0], XB.shape[0]))
    total = (Y.shape[0] + batch_size - 1) // batch_size
    for b_Y, b_XA in tqdm(batches(Y, XA, batch_size=batch_size), total=total):
        b_Y[:] = cdist(b_XA, XB)
    return Y


def distance(train_data, test_data):
    """Pairwise euclidean distances between a training dataset and a test dataset, so that we get the distances for
    each point in our test_data to each point in our train data.

    Parameters
    ----------
    train_data : obj:`torch.Tensor` or obj:`np.ndarray`
        A tensor with samples in the first dimension and features in all others, to which we want to compute the
        distance of each point in `test_data`. (The ones we want the tsp-projection for)
    test_data : obj:`torch.Tensor` or obj:`np.ndarray`
        A tensor with samples in the first dimension and features in all others, where we want to find the distances of
        each sample to all other samples in `train_data`.

    Returns
    -------
    dist : obj:`np.ndarray`
        The pairwise euclidean distances of each point in `test_data` to each point in `train_data`.
        Its shape is `test_data.shape[0]` x `train_data.shape[0]`

    """
    if isinstance(train_data, torch.Tensor):
        train_data = train_data.numpy()
    if isinstance(test_data, torch.Tensor):
        test_data = test_data.numpy()

    train_data = train_data.reshape((train_data.shape[0], np.prod(train_data.shape[1:])))
    test_data = test_data.reshape(
        (test_data.shape[0], np.prod(test_data.shape[1:]))
    )
    dist = bcdist(test_data, train_data)
    return dist


def knn(dist, n_neighbours):
    """Find the k-nearest neighbours given a matrix of distances.

    Parameters
    ----------
    dist : obj:`np.ndarray`
        Matrix of distances, where we want to find nearest neighbours for each sample in dimension 0.
    n_neighbours : int
        Number of nearest neighbours to find.

    Returns
    -------
    cols : obj:`np.ndarray`
        2-d array with samples in rows and nearest neighbours indice in columns.
    """
    # get k-nearest neighbours
    cols = dist.argsort(1)[:, :n_neighbours]

    return cols


def get_projectors(train_data, test_data, indices, d_singular, batchsize, strict_mode, exists):
    """Find the singular vectors to compute the tangent-space projectors for each point in `test_data`, given all
    points in `train_data`.

    Parameters
    ----------
    train_data : obj:`torch.Tensor`
        Points in rows under which to find tangent-space for each point in `test_data`.
    test_data : obj:`torch.Tensor`
        Points in rows for which to find tangent-space.
    indices : obj:`np.ndarray`
        Indices of k-nearest as returned by `knn`
    d_singular : int
        Number of singular dimensions to use for projectors.
    batchsize : int
        Size of mini-batches for SVD.
    strict_mode : bool
        If true, center tangent around the datapoint for which to find tangent-space. If false, center around the mean
        of the data point and its nearest neighbours.
    exists : bool
        Specifies whether the point itself is already in `train_data` (i.e. when `train_data` and `test_data` are the
        same). If `strict_mode` is False and `exists` is False, replace the furthest neighbour with the point itself.
        Otherwise, `exists` does not change the outcome.

    Returns
    -------
    V : obj:`np.ndarray`
        The singular vectors. The tangent-space projectors can be computed by V dot V^T.
    S : obj:`np.ndarray`
        The singular values.

    """
    # calc svd. Proj matrix can be obtained by v.v^T. Storing v is more efficient.
    with torch.no_grad():
        train_data = train_data.to(torch.float32).numpy()
        test_data = test_data.numpy()

        # B - batches, N - neighbours, D - features
        B, N = indices.shape
        _, D = train_data.shape

        S = np.empty((B, N), dtype=np.float32)
        V = np.empty((B, N, D), dtype=np.float32)
        # U = np.empty((batchsize, N, D), dtype=np.float32)
        n = 0

        total = (B + batchsize - 1) // batchsize
        for b_index, b_test_data in tqdm(
            batches(indices, test_data, batch_size=batchsize),
            total=total
        ):
            blen = len(b_test_data)
            ind = slice(n, n + blen)
            n = n + blen
            if strict_mode:
                b_points = train_data[b_index, :] - b_test_data[:, None]
            else:
                b_points = train_data[b_index, :]
                if not exists:
                    b_points = np.concatenate([b_test_data[:, None], b_points[:, :-1]], axis=1)
                b_points = b_points - np.mean(b_points, axis=1)[:, None, :]
            _, S[ind], V[ind] = np.linalg.svd(b_points, full_matrices=False)

        V = V.transpose(0, 2, 1)
        S = S[:, :d_singular]
        V = V[:, :, :d_singular]

    return V, S


def load_data(dataroot, download=False, train=True, dataset='cifar', flatten=True, normalization=None):
    """Load a full normalized dataset split as a tensor by using an identifier string.

    Parameters
    ----------
    download : bool, optional
        If true, download the data if it does not exist already.
    train : bool, optional
        If true, return the training split, otherwise return the test split.
    dataset : str, optional
        String identifier of the dataset. Possible values are 'mnist', 'fmnist' and 'cifar'. Default value is 'cifar'.
    flatten : bool, optional
        If True, flatten (combine to a single dimension) the features of each training sample.
    normalization : str, optional
        Use one of the predetermined dataset normalizations. Possible values are 'mnist', 'fmnist', 'cifar' and
        'cifar-shift'. Default is the same value as `dataset`.
    """
    norms = {
        'mnist': transforms.Normalize((0.1307,), (0.3081,)),
        'fmnist': transforms.Normalize((0.2860,), (0.3530,)),
        'cifar': transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "cifar-shift": transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    }

    if normalization is None:
        normalize = norms[dataset]
    else:
        normalize = norms[normalization]

    kwargs = {
        'root': dataroot,
        'train': train,
        'download': download
    }
    if dataset == 'mnist':
        kwargs['transform'] = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        dataset_ = MNIST(**kwargs)
    elif dataset == 'fmnist':
        kwargs['transform'] = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        dataset_ = FashionMNIST(**kwargs)
    elif dataset == 'cifar':
        kwargs['transform'] = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        dataset_ = CIFAR10(**kwargs)
    else:
        raise TypeError('Unsupported dataset type: \'{}\''.format(dataset))

    c_data, c_target = full_data(dataset_)
    if flatten:
        c_data = c_data.reshape(c_data.shape[0], np.prod(c_data.shape[1:]))

    return c_data, c_target


def target_filter(c_data, target, c_target):
    """Given labels, filter a dataset by a specified target label. This reduces the dataset to the specified class.

    Parameters
    ----------
    c_data : obj:`torch.Tensor`
        Tensor containing a full dataset.
    target : obj:`torch.Tensor`
        Target label (class) to filter `c_data` by.
    c_target : obj:`torch.Tensor`
        Labels (classes) of data points in `c_data`

    Returns
    -------
    c_data : obj:`torch.Tensor`
        The filtered tensor, containing only samples from `c_data` which have the label `target`.

    """
    if target is None:
        indices = (slice(None), ...)
    else:
        indices = (c_target.cpu().numpy() == target, ...)
    c_data = c_data[indices]
    return c_data, indices


def get_jac_proj(model, data, d_singular, batchsize, device):
    """Compute the singular vectors (and values) of the jacobian of the decoder of a VQVAE to compute the tangent-space
    projectors given the auto encoder's (hopefully densly-sampled) embedding.

    Parameters
    ----------
    model : obj:`torch.nn.Module`
        The VQVAE model.
    data : obj:`torch.Tensor`
        Data points for which to find the tangent-space projectors.
    d_singular : int
        Number of singular dimensions.
    batchsize : int
        Size of mini-batches to compute the jacobian.
    device : obj:`torch.device`
        Device for the computation of the jacobian

    Returns
    -------
    V : obj:`np.ndarray`
        The singular vectors. The tangent-space projectors can be computed by V dot V^T.
    S : obj:`np.ndarray`
        The singular values.

    """
    model = model.to(device)
    B = data.shape[0]
    D = data.shape[1:]

    S = np.empty((B, d_singular), dtype=np.float32)
    V = np.empty((B, np.prod(D).item(), d_singular), dtype=np.float32)
    n = 0

    total = (B + batchsize - 1) // batchsize
    for batch, in tqdm(
        batches(data, batch_size=batchsize),
        total=total,
        disable=None
    ):
        lb = len(batch)
        ind = slice(n, n + lb)
        n = n + lb

        batch = batch.to(device)
        z = model._pre_vq_conv(model._encoder(batch))
        z.requires_grad_()
        zb = model._vq_vae(z)[1]
        output = model._decoder(zb)
        D = output.shape[1:]
        d = z.shape[1:]

        output = output.reshape(lb, np.prod(D))

        out_grads = torch.eye(np.prod(D).item(), device=device)
        jacobian = torch.empty(np.prod(D).item(), lb, np.prod(d).item(), device=device)

        for jac, eye in zip(jacobian, out_grads):
            jac[:] = torch.autograd.grad(
                output,
                z,
                grad_outputs=eye[None].repeat(lb, 1),
                retain_graph=True
            )[0].reshape(lb, np.prod(d))

        jacobian = jacobian.permute(1, 0, 2)
        u, s, v = torch.svd(jacobian)

        min_eig = min(d_singular, np.prod(d))

        V[ind, :, :min_eig] = u[:, :, :min_eig].detach().cpu().numpy()
        S[ind, :min_eig] = s[:, :min_eig].detach().cpu().numpy()

    return V, S


def train_ae(
    loaders,
    batch_size,
    num_training_updates,
    learning_rate,
    log_dir,
    log_rate,
    checkpoint,
    save_rate,
    load,
    target,
    device
):
    """Training routine for VQVAE.
    """
    training_loader, validation_loader = loaders

    if log_dir:
        writer = SummaryWriter(log_dir=log_dir)

    model = VQVAE()
    if load is not None:
        state_dict = torch.load(load)
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    model.train()

    with tqdm(range(num_training_updates)) as pbar:
        for i in pbar:
            data, = next(iter(training_loader))
            data = data.to(device)
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity = model(data)
            recon_error = F.mse_loss(data_recon, data)
            loss = recon_error + vq_loss
            loss.backward()

            optimizer.step()

            pbar.set_description(f"step: {i} recon error: {recon_error.item()} perplexity: {perplexity.item()}")

            if log_dir and (i+1) % log_rate == 0:
                writer.add_scalar('recon-err',  recon_error.item(), global_step=i)
                writer.add_scalar('perplexity', perplexity.item(),  global_step=i)

            if (i % save_rate) == (save_rate - 1):
                if checkpoint is not None:
                    state_dict = model.state_dict()
                    fname = checkpoint.format(iter=i + 1)
                    print('Saving checkpoint: %s', fname)
                    torch.save(state_dict, fname)

            # For some reason, training often gets initally stuck.
            # This is a hack to deal with this problem.
            if i == 40 and perplexity.item() == 1:
                return False

    return True


@click.group()
def main():
    """A command-line interface to compute tangent-space projectors."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@main.command()
@click.option(
    '--dataroot',
    type=click.Path(exists=True, file_okay=False, writable=True),
    default='share/data',
    help='path to store datasets'
)
@click.option(
    '--dataset',
    type=click.Choice(['cifar', 'mnist', 'fmnist']),
    default='fmnist',
    help='dataset to train on'
)
@click.option('--neighbours', type=int, default=30, help='number of nearest neighbours to use')
@click.option('--d-singular', type=int, default=30, help='number of singular dimension for the tsp-projection')
@click.option('--download/--no-download', default=False, help='download data if not available')
@click.option('--overwrite/--no-overwrite', default=False, help='overwrite output if it exists')
@click.option('--all-data/--no-all-data', default=False, help='also compute projectors for training split')
@click.option('--strict-mode/--no-strict-mode', default=False, help='tangent planes strictly contain data points')
@click.option(
    '--save',
    type=click.Path(writable=True, dir_okay=False),
    default='var/tsp/{dataset}.h5',
    help='target hdf5 file to store singular vectors'
)
@click.option('--batch-size', type=int, default=128, help='size of mini-batches for svd')
@click.option('--targets', type=csints, default=','.join(str(i) for i in range(10)), help='classes to compute for')
def linear(
    dataroot,
    dataset,
    all_data,
    neighbours,
    d_singular,
    download,
    overwrite,
    strict_mode,
    save,
    batch_size,
    targets
):
    """A command-line interface to compute linear tangent-space projectors for densely-sampled datasets."""
    logging.info("Loading data...")
    train_data, train_targets = load_data(dataroot, download, dataset=dataset)
    test_data, test_targets = load_data(dataroot, download, train=False, dataset=dataset)

    modes = [('test', (test_data, test_targets))]
    if all_data:
        modes += [('train', (train_data, train_targets))]

    fname = save.format(dataset=dataset)
    logging.info(f"Saving to '{fname}' ...")
    fdir = os.path.dirname(fname)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    fd = h5py.File(fname, 'a')

    for mode, (mode_data, mode_targets) in modes:
        group = fd.require_group(mode)
        dset_s = group.require_dataset('s', shape=(mode_data.shape[0], d_singular), dtype=np.float)
        dset_v = group.require_dataset(
            'v',
            shape=(mode_data.shape[0], np.prod(mode_data.shape[1:]), d_singular),
            dtype=np.float
        )

        logging.info(f"Using mode '{mode}'")

        for target in targets:
            logging.info(f"Using target {target}")

            c_train, _ = target_filter(train_data, target, train_targets)
            c_test, idx_test = target_filter(mode_data, target, mode_targets)

            logging.info("Computing distances...")
            dist = distance(c_train, c_test)

            logging.info("Checking out neighbourhood...")
            n_indices = knn(dist, neighbours)

            logging.info("Projecting...")
            v, s = get_projectors(
                c_train,
                c_test,
                n_indices,
                d_singular,
                batch_size,
                strict_mode,
                mode == "train",
            )
            dset_s[idx_test] = s
            dset_v[idx_test] = v

    fd.close()


@main.group()
@click.option('--seed', type=int, default=0xDEADBEEF, help='random seed')
@click.option('--load', type=click.Path(), help='autoencoder model parameters to load')
@click.option('--use-cpu/--use-gpu', default=False, help='device to use')
@click.pass_context
def ae(ctx, seed, load, use_cpu):
    """A command-line interface to train autoencoders for tangent-space projection"""
    torch.manual_seed(seed)

    # define computation device
    if torch.cuda.is_available() and not use_cpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    ctx.ensure_object(Namespace)

    model = VQVAE()
    if load is not None:
        state_dict = torch.load(load)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)

    ctx.obj.model = model
    ctx.obj.device = device


@ae.command("train")
@click.option(
    '--dataroot',
    type=click.Path(exists=True, file_okay=False, writable=True),
    default='share/data',
    help='path to store datasets'
)
@click.option('--dataset', type=click.Choice(['cifar']), default='cifar', help='used dataset')
@click.option('--download/--no-download', default=False, help='download dataset if it does not exists')
@click.option("--batch-size", type=int, default=256, help='number of mini-batches used for training')
@click.option("--num-training-updates", type=int, default=15000, help='number of training updates')
@click.option("--learning-rate", type=float, default=1e-3, help='learning rate')
@click.option('--log-dir', type=click.Path(), help='path to log tensorboard')
@click.option('--log-rate', type=int, default=100, help='after every how many epochs to write log')
@click.option('--checkpoint', type=click.Path(), help='target to save checkpoints')
@click.option('--save-rate', type=int, default=1000, help='after every how many epochs to store model parameters')
@click.option('--load', type=click.Path(), help='where to load parameters from')
@click.option('--target', type=int, default=1, help='target label to filter')
@click.pass_context
def ae_train(
    ctx,
    dataroot,
    dataset,
    download,
    batch_size,
    num_training_updates,
    learning_rate,
    log_dir,
    log_rate,
    checkpoint,
    save_rate,
    load,
    target,
):
    """A command-line interface to train autoencoders for tangent-space projection"""
    device = ctx.obj.device

    logging.info("Loading data...")
    loaders = []
    for train in (True, False):
        data, targets = load_data(
            dataroot,
            download,
            train,
            dataset=dataset,
            flatten=False,
            normalization='cifar-shift'
        )
        c_data, _ = target_filter(data, target, targets)
        loader = DataLoader(TensorDataset(c_data), batch_size=batch_size, shuffle=train)
        loaders.append(loader)

    for i in range(30):
        logging.info(f"Starting training attempt {i}...")
        result = train_ae(
            loaders,
            batch_size,
            num_training_updates,
            learning_rate,
            log_dir,
            log_rate,
            checkpoint,
            save_rate,
            load,
            target,
            device
        )

        if result:
            break


@ae.command('projectors')
@click.option(
    '--dataroot',
    type=click.Path(exists=True, file_okay=False, writable=True),
    default='share/data',
    help='path to store datasets'
)
@click.option("--target", type=int, default=0)
@click.option('--dataset', type=click.Choice(['cifar']), default='cifar')
@click.option('--all-data/--no-all-data', default=False)
@click.option('--d-singular', type=int, default=30)
@click.option('--download/--no-download', default=False)
@click.option('--save', type=click.Path(), default='./svd.h5')
@click.option('--batchsize', type=int, default=64)
@click.pass_context
def ae_projectors(
    ctx,
    dataroot,
    target,
    dataset,
    all_data,
    d_singular,
    download,
    save,
    batchsize,
):
    """Command-line interface to compute tsp-projectors using a VQVAE."""
    device = ctx.obj.device
    model = ctx.obj.model

    model.eval()

    mode_list = ['test']
    if all_data:
        mode_list.append('train')

    for mode in mode_list:
        logging.info("Loading data...")
        if mode == 'train':
            data, targets = load_data(
                dataroot,
                download,
                train=True,
                dataset=dataset,
                flatten=False,
                normalization="cifar-shift"
            )
        elif mode == 'test':
            data, targets = load_data(
                dataroot,
                download,
                train=False,
                dataset=dataset,
                flatten=False,
                normalization="cifar-shift"
            )
        else:
            raise TypeError('Unsupported mode!')

        logging.info("Using mode '%s'", mode)
        logging.info("Using target %d", target)

        c_data, idx = target_filter(data, target, targets)

        v, s = get_jac_proj(model, c_data, d_singular, batchsize, device)
        with h5py.File(save, 'a') as fd:
            group = fd.require_group(mode)
            sdset = group.require_dataset('s', shape=(data.shape[0], d_singular), dtype=np.float32)
            vdset = group.require_dataset(
                'v',
                shape=(data.shape[0], np.prod(data.shape[1:]), d_singular),
                dtype=np.float32
            )
            vdset[idx] = v
            sdset[idx] = s

        logging.info("Saving as '{}' ...".format(save))


if __name__ == '__main__':
    main()
