#!/usr/bin/env python3
"""Command-line interface to train and evaluate models to force their explanations to reproduce some target heatmap
while imitating the predictions of a teacher model.
"""
import torch
import click
import logging
import os
from functools import reduce
from sys import stdout

from tqdm import tqdm

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import fairwasher.utils as utils
import fairwasher.models as models
import fairwasher.nn.networks as networks
from fairwasher.datasets import ProjectedDataset


def csints(string):
    """Split a comma separated string into a list of integers"""
    if string:
        return [int(elem) for elem in string.split(',')]
    else:
        return []


def train_model(
    student_model,
    teacher_model,
    optimizer,
    train_loader,
    method,
    device,
    target_heatmap,
    alpha=0.01,
    loss_fn=torch.nn.MSELoss()
):
    """Train a model for one epoch to produce the same output as a teacher model, but of which the explanation is a
    target heatmap.

    Parameters
    ----------
    student_model : obj:`torch.Module`
        Student model to imitate predictions of `teacher_model` and force explanation to become `target_heatmap`.
    teacher_model : obj:`torch.Module`
        Teacher model of which the predictions are imitated by `student_model`.
    optimizer : obj:`torch.optim.Optimizer`
        Optimizer used for the training of `student_model`.
    train_loader : obj:`torch.utils.data.DataLoader`
        Dater loader which is iterated to get batches of data samples.
    method : str
        Explanation method for which `student_model`'s explanation shall resemble `target_heatmap`.
    device : obj:`torch.device`
        Device on which to execute the training.
    target_heatmap : obj:`torch.Tensor`
        Target explanation which `student_model` shall resemble.
    alpha : float, optional
        Weighting of explanation loss in the loss term.
    loss_fn : function, optional
        Function to compute the scalar loss between `student_model` and `teacher_model`.

    """
    for batch in tqdm(train_loader, disable=None):
        if len(batch) == 3:
            input, _, proj = batch
            proj = proj.to(device)
        else:
            input, _ = batch
            proj = None
        input = input.to(device)

        # model comparison:
        with torch.no_grad():
            teacher_pred = F.softmax(teacher_model(input), dim=1)
        student_pred = F.softmax(student_model(input), dim=1)
        loss_model = loss_fn(student_pred, teacher_pred)

        # heatmap comparision:
        heatmap = utils.get_heatmap(student_model, input, method, proj=proj)
        mse_heatmap = F.mse_loss(heatmap, target_heatmap.expand(heatmap.shape), reduction="sum") / heatmap.shape[0]

        loss = (1 - alpha) * loss_model + alpha * mse_heatmap

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate_model(
    student_model,
    teacher_model,
    test_loader,
    method,
    device,
    target_heatmap,
    alpha=0.01,
    loss_fn=torch.nn.MSELoss(),
    meter=None
):
    """Train a model for one epoch to produce the same output as a teacher model, but of which the explanation is a
    target heatmap.

    Parameters
    ----------
    student_model : obj:`torch.Module`
        Student model to imitate predictions of `teacher_model` and force explanation to become `target_heatmap`.
    teacher_model : obj:`torch.Module`
        Teacher model of which the predictions are imitated by `student_model`.
    optimizer : obj:`torch.optim.Optimizer`
        Optimizer used for the training of `student_model`.
    test_loader : obj:`torch.utils.data.DataLoader`
        Dater loader which is iterated to get batches of data samples.
    method : str
        Explanation method for which `student_model`'s explanation shall resemble `target_heatmap`.
    device : obj:`torch.device`
        Device on which to execute the training.
    target_heatmap : obj:`torch.Tensor`
        Target explanation which `student_model` shall resemble.
    alpha : float, optional
        Weighting of explanation loss in the loss term.
    loss_fn : function, optional
        Function to compute the scalar loss between `student_model` and `teacher_model`.
    meter : obj:`fairwasher.utils.MetricMeter`, optional
        Meter to accumulate statistics.

    """
    for batch in tqdm(test_loader, disable=None):
        if len(batch) == 3:
            input, target, proj = batch
            proj = proj.to(device)
        else:
            input, target = batch
            proj = None
        input = input.to(device)
        target = target.to(device)

        heatmap = utils.get_heatmap(student_model, input, method, proj=proj)

        with torch.no_grad():
            sister_pred = F.softmax(teacher_model(input), dim=1)
            pred = F.softmax(student_model(input), dim=1)
            loss_model = loss_fn(pred, sister_pred)

            mse_heatmap = F.mse_loss(heatmap, target_heatmap.expand(heatmap.shape), reduction="sum") / heatmap.shape[0]

            loss = (1 - alpha) * loss_model + alpha * mse_heatmap
            acc = utils.get_topk_acc(pred, target)[0]

            # update meter
            if meter:
                meter["hm_mse"].update(mse_heatmap.item())
                meter["nn_loss"].update(loss_model.item())
                meter["loss"]  .update(loss.item())
                meter["nn_acc"].update(acc.item())


def save_model(out_dir, model, dset, method, step, acc):
    """Generate a filename and save a model using `torch.save`

    Parameters
    ----------
    out_dir : string
        Base path to which the model is saved.
    model : obj:`torch.nn.Module`
        Model to be saved.
    dset : string
        Name of the dataset
    method : string
        Name of the explanation method
    step : int
        Number to represent at which step this model was saved.
    acc : float
        Model accuracy to be stored alongside the model.

    """
    filename = os.path.join(out_dir, 'fairwashed_{}_{}_{}.pth'.format(dset, method, step))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save({'epoch': step, 'state_dict': model.state_dict(), 'acc': acc}, filename)


def create_loader(train_dataset, test_dataset, batch_size, proj_file=None, num_workers=4):
    """Wrap datasets in `fairwasher.datasets.ProjectedDataset` and create a pair of training and test loader instances.

    Parameters
    ----------
    train_dataset : obj:`torch.utils.data.Dataset`
        Dataset to draw training samples from.
    test_dataset : obj:`torch.utils.data.Dataset`
        Dataset to draw test samples from.
    batch_size : int
        Size of mini-batches to be drawn using loaders.
    proj_file : string, optional
        Path to a projection file of type hdf5. The loader will supply projectors for each sample if supplied
    num_workers : int, optional
        Number of workers used by `torch.utils.data.DataLoader`

    Returns
    -------
    train_loader : obj:`torch.utils.data.DataLoader`
        Loader of `train_dataset` wrapped in `fairwasher.dataset.ProjectedDataset` to supply projectors from
        `proj_file`, if supplied.
    test_loader : obj:`torch.utils.data.DataLoader`
        Loader of `test_dataset` wrapped in `fairwasher.dataset.ProjectedDataset` to supply projectors from
        `proj_file`, if supplied.

    """
    result = []
    for mode, dataset in ((True, train_dataset), (False, test_dataset)):
        dataset = ProjectedDataset(dataset, projection_file=proj_file, train=mode)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        result.append(loader)

    return tuple(result)


def load_data(dataset="cifar", dataroot='data'):
    """Load a dataset given a string identifier, downloading if necessary.

    Parameters
    ----------
    dataset : string, optional
        Dataset to load, one of 'cifar', 'mnist' or 'fmnist'
    dataroot : string, optional
        Path from which to load the dataset, or to which to store if the dataset is not found.

    Returns
    -------
    obj:`torch.utils.data.Dataset`
        Dataset object identified by `dataset`.

    Raises
    ------
    TypeError
        If `dataset` is an unknown identifier.

    """
    if dataset == "cifar":
        return load_cifar_data(dataroot=dataroot)
    elif dataset == 'mnist':
        return load_mnist_data(dataroot=dataroot)
    elif dataset == 'fmnist':
        return load_fmnist_mnist_data(dataroot=dataroot)
    else:
        raise TypeError('No dataset \'{}\''.format(dataset))


def load_cifar_data(dataroot='data'):
    """Load a pair of training and test splits of the CIFAR10 dataset, with different pre-processing transformations
    applied.

    Parameters
    ----------
    dataroot : string, optional
        Path from which to load the dataset, or to which to store if the dataset is not found.

    Returns
    -------
    obj:`torch.utils.data.Dataset`
        CIFAR10 Dataset object.

    """
    data_mean = np.array([0.485, 0.456, 0.406])
    data_std = np.array([0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=data_mean, std=data_std)
    train_dataset = datasets.CIFAR10(
        root=dataroot,
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True
    )
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    )

    return train_dataset, test_dataset


def load_mnist_data(dataroot='data'):
    """Load a pair of training and test splits of the MNIST dataset, with different pre-processing transformations
    applied.

    Parameters
    ----------
    dataroot : string, optional
        Path from which to load the dataset, or to which to store if the dataset is not found.

    Returns
    -------
    obj:`torch.utils.data.Dataset`
        MNIST Dataset object.

    """
    train_dataset = datasets.MNIST(
       dataroot,
       train=True,
       download=True,
       transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
       ])
    )

    test_dataset = datasets.MNIST(
        dataroot,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    return train_dataset, test_dataset


def load_fmnist_mnist_data(dataroot='data'):
    """Load a pair of training and test splits of the FashionMNIST dataset, with different pre-processing
    transformations applied.

    Parameters
    ----------
    dataroot : string, optional
        Path from which to load the dataset, or to which to store if the dataset is not found.

    Returns
    -------
    obj:`torch.utils.data.Dataset`
        FashionMNIST Dataset object.

    """
    train_dataset = datasets.FashionMNIST(
        dataroot,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    )

    test_dataset = datasets.FashionMNIST(
        dataroot,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    )

    return train_dataset, test_dataset


def create_model(method, checkpt, dataset="cifar"):
    """Create a torch model conditioned on the dataset, and with necessary steps taken to assure the proper execution
    of a specified explanation method.

    Parameters
    ----------
    method : string
        Identifier of the explanation method. Only 'lrp' changes the model.
    checkpt : string
        Path to a parameter file of the model to be loaded.
    dataset : string, optional
        Dataset identifier for model, one of 'cifar', 'mnist' or 'fmnist'

    Returns
    -------
    model : obj:`torch.Module`
        Model object fit for dataset identified by `dataset`

    Raises
    ------
    TypeError
         If `dataset` is an unknown identifier.

    """
    if dataset == "cifar":
        return create_cifar_model(method, checkpt)
    elif dataset in ('mnist', 'fmnist'):
        return create_mnist_model(method, checkpt)
    else:
        raise TypeError('No model for dataset \'{}\''.format(dataset))


def create_cifar_model(method, checkpt):
    """Create a torch model for CIFAR10, and with necessary steps taken to assure the proper execution
    of a specified explanation method.

    Parameters
    ----------
    method : string
        Identifier of the explanation method. Only 'lrp' changes the model.
    checkpt : string
        Path to a parameter file of the model to be loaded.

    Returns
    -------
    model : obj:`torch.Module`
        Model object fit for CIFAR10.

    """
    student_model = models.VGG16("relu" if method == "lrp" else "softplus", beta=20)
    if checkpt:
        # todo: this is hacky, change!
        student_model.features = torch.nn.DataParallel(student_model.features)
        student_model.load_state_dict(torch.load(checkpt, map_location="cpu")["state_dict"])

    if method == "lrp":
        student_model.features = student_model.features.module
        student_model = networks.ExplainableNet(student_model, beta=50)

    return student_model


def create_mnist_model(method, checkpt):
    """Create a torch model for MNIST or FashionMNIST, and with necessary steps taken to assure the proper execution of
    a specified explanation method.

    Parameters
    ----------
    method : string
        Identifier of the explanation method. Only 'lrp' changes the model.
    checkpt : string
        Path to a parameter file of the model to be loaded.

    Returns
    -------
    model : obj:`torch.Module`
        Model object fit for MNIST or FashionMNIST.

    """
    student_model = models.ConvNet("relu" if method == "lrp" else "softplus", beta=20)

    if checkpt:
        student_model.load_state_dict(torch.load(checkpt, map_location="cpu")["state_dict"])

    if method == "lrp":
        student_model = networks.ExplainableNet(student_model, beta=50)

    return student_model


@click.group()
def main():
    """Command-line interface for training and evaluation of fairwashing models."""
    logging.basicConfig(level=logging.INFO)


@main.command()
@click.option(
    '--dataroot', type=click.Path(file_okay=False, writable=True), default='share/data', help='path to store datasets'
)
@click.option(
    '--dataset',
    type=click.Choice(['cifar', 'mnist', 'fmnist']),
    default='fmnist',
    help='dataset to train on'
)
@click.option(
    '--method',
    type=click.Choice(['lrp', 'grad', 'xgrad', 'intgrad']),
    default='grad',
    help='explanation method'
)
@click.option(
    '--teacher-model', type=click.Path(exists=True, dir_okay=False), default='share/models/fmnist_acc.pth', help='model checkpoint'
)
@click.option('--batch-size', type=int, default=256, help='size of mini-batches for SGD')
@click.option('--num-workers', type=int, default=4, help='number data loading workers')
@click.option(
    '--target', type=click.Path(exists=True, dir_okay=False), default='share/targets/42.jpg', help='target heatmap'
)
@click.option('--lr', type=float, default=5e-5, help='learning rate')
@click.option('--n-epochs', type=int, default=1000, help='number of epochs')
@click.option(
    '--checkpt',
    type=click.Path(exists=True, dir_okay=False),
    default='share/models/fmnist_acc.pth',
    help='checkpoint to continue training from'
)
@click.option('--alpha', type=float, default='0.8', help='weight factor between model and hm mse')
@click.option(
    '--out-dir',
    type=click.Path(file_okay=False, writable=True),
    default='var/params',
    help='directory to which checkpts will be saved'
)
@click.option('--proj-file', type=click.Path(exists=True, dir_okay=False), help='projector on tangent space')
@click.option('--val-interval', type=int, default=5, help='interval when validation and save is performed')
@click.option(
    '--optimizer', 'optimizer_name', type=click.Choice(['adam', 'sgd']), default='adam', help='optimizer for training'
)
@click.option(
    '--loss',
    'loss_name',
    type=click.Choice(['mse', 'ce']),
    default='mse',
    help='loss function between student and teacher output'
)
@click.option('--lr-schedule', type=csints, default='', help='epochs at which to divide lr by 10')
def train(
    dataroot,
    dataset,
    method,
    teacher_model,
    batch_size,
    num_workers,
    target,
    lr,
    n_epochs,
    checkpt,
    alpha,
    out_dir,
    proj_file,
    val_interval,
    optimizer_name,
    loss_name,
    lr_schedule
):
    """Train a model to produce a target explanation while keeping the output similar to a teacher model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger()

    train_loader, test_loader = create_loader(
        *load_data(dataset, dataroot=dataroot),
        batch_size=batch_size,
        proj_file=proj_file
    )
    target_heatmap = utils.load_target_heatmap(target).to(device)

    # model:
    # eval mode disables dropout. Necessary since we want to exactly reproduce teacher
    teacher_model = utils.load_model(teacher_model, False, device, dataset).eval()
    student_model = create_model(method, checkpt, dataset).to(device).eval()

    optimizer = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD
    }[optimizer_name](student_model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        (lambda epoch: reduce(float.__mul__, [0.1 for limit in lr_schedule if epoch >= limit], 1.0))
    )

    loss_fn = {
        'mse': torch.nn.MSELoss,
        'ce': torch.nn.CrossEntropyLoss,
    }[loss_name]()

    acc = 0.

    for step in tqdm(range(n_epochs), disable=None):
        if (step + 1) % val_interval == 0:
            meter = utils.MetricMeter({
                "hm_mse": utils.Metric(),
                "nn_loss": utils.Metric(),
                "loss": utils.Metric(),
                "nn_acc": utils.Metric(),
            })

            validate_model(
                student_model,
                teacher_model,
                test_loader,
                method,
                device,
                target_heatmap,
                alpha,
                loss_fn=loss_fn,
                meter=meter
            )
            stat = meter.aggregate()
            logger.info(
                "step: {}  acc: {:.2f} loss: {:.4f} hm_mse: {:.8f} model_mse: {:.5f})".format(
                    step,
                    stat['nn_acc'][0],
                    stat['loss'][0],
                    stat['hm_mse'][0],
                    stat['nn_loss'][0]
                )
            )

            acc = stat["nn_acc"][0]
            save_model(out_dir, student_model, dataset, method, step + 1, acc)

        train_model(
            student_model,
            teacher_model,
            optimizer,
            train_loader,
            method,
            device,
            target_heatmap,
            alpha,
            loss_fn=loss_fn
        )
        scheduler.step()

    save_model(out_dir, student_model, dataset, method, n_epochs, acc)


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
@click.option(
    '--method',
    type=click.Choice(['lrp', 'grad', 'xgrad', 'intgrad']),
    default='grad',
    help='explanation method'
)
@click.option('--model', 'model_file', type=click.Path(exists=True, dir_okay=False), help='student model parameters')
@click.option('--proj-file', type=click.Path(exists=True, dir_okay=False), help='projector on tangent space')
@click.option('--target', 'target_map', type=click.Path(exists=True, dir_okay=False), help='target heatmap')
@click.option(
    '--teacher-model', 'teacher_file', type=click.Path(exists=True, dir_okay=False), help='teacher to compare to'
)
@click.option('--batch-size', type=int, default=256, help='batch size')
@click.option('--output', type=click.Path(dir_okay=False, writable=True))
def evaluate(dataroot, dataset, method, model_file, proj_file, target_map, batch_size, teacher_file, output):
    """Evaluate how well student model reproduces target explanation, and how similar it is to a teacher model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = create_loader(
        *load_data(dataset, dataroot=dataroot),
        batch_size=batch_size,
        proj_file=proj_file
    )

    model = utils.load_model(model_file, method == "lrp", device, dataset).eval()

    meter_dict = {"nn_acc": utils.Metric()}
    if target_map:
        meter_dict.update({
            "hm_mse": utils.Metric(),
            "hm_pcc": utils.Metric(),
            "hm_ssim": utils.Metric(),
        })
    if teacher_file:
        meter_dict['nn_mse'] = utils.Metric()
        teacher = utils.load_model(teacher_file, False, device, dataset).eval()

    meter = utils.MetricMeter(meter_dict)

    for batch in tqdm(test_loader, disable=None):
        if len(batch) == 3:
            input, target, proj = batch
        else:
            input, target = batch
            proj = None
        target = target.to(device)
        input = input.to(device)
        if proj is not None:
            proj = proj.to(device)

        if target_map:
            # get heatmaps
            heatmap = utils.get_heatmap(model, input, method, proj).to(device)
            target_heatmap = torch.stack([utils.load_target_heatmap(target_map).to(device)] * len(heatmap))

            # calculate similarity metrices
            mse_heatmap = F.mse_loss(heatmap, target_heatmap).item()
            pcc_heatmap = utils.calc_pcc(heatmap, target_heatmap)
            ssim_heatmap = utils.calc_ssim(heatmap, target_heatmap)

            meter["hm_mse"].update(mse_heatmap)
            meter["hm_pcc"].update(pcc_heatmap)
            meter["hm_ssim"].update(ssim_heatmap)

        if teacher_file:
            model_out = F.softmax(model(input), dim=1)
            teacher_out = F.softmax(teacher(input), dim=1)
            nn_mse = F.mse_loss(model_out, teacher_out).item()
            meter['nn_mse'].update(nn_mse)

        # calculate acc and loss
        nn_acc = utils.get_topk_acc(model(input), target)

        # update meter
        meter["nn_acc"].update(nn_acc)

    caption = {
        'nn_acc': 'Student accuracy',
        'nn_mse': 'MSE output student vs. teacher',
        'hm_mse': 'MSE explanation student vs. target',
        'hm_pcc': 'PCC explanation student vs. target',
        'hm_ssim': 'SSIM explanation student vs. target'
    }
    msg = '\n'.join([
        '{:<36}: {:.4e}+-{:.4e}'.format(caption[key], mean, std) for key, (mean, std) in meter.aggregate().items()
    ]) + '\n'
    if output:
        with open(output, 'w') as fd:
            fd.write(msg)
    else:
        stdout.write(msg)


if __name__ == '__main__':
    main()
