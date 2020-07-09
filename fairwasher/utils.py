import torch
import math

import numpy as np

from scipy.stats import pearsonr
from skimage.metrics import structural_similarity
from collections import UserDict
from PIL import Image
from PIL.ImageChops import invert

import fairwasher.models
import fairwasher.nn.networks as networks


def get_iid_statistics(metric):
    with torch.no_grad():
        history = metric.history
        count = history.shape[0]
        mean = history.mean()
        sq_mean = (history**2).mean()

        std = torch.sqrt(abs(sq_mean - mean**2))
        err = std / math.sqrt(count)

        return mean, err


class MetricMeter(UserDict):
    def __init__(self, observables, stat_func=get_iid_statistics):
        self.data = observables
        self.stat_func = stat_func

    def update(self, sample):
        for k in self.data:
            self.data[k].update(sample)

    def reset(self):
        for k in self.data:
            self.data[k].reset()

    def aggregate(self):
        return {k: self.stat_func(self.data[k]) for k in self.data}

    def __str__(self):
        str = ""
        r = self.aggregate()

        for k in r:
            m, std = r[k]
            str += "\n{}: {}+-{}".format(k, m, std)

        return str


class Metric:
    def __init__(self):
        self.history = None

    def update(self, config):
        self._append(config)

    def reset(self):
        self.history = None

    def _append(self, value):
        tensor = torch.tensor(value).unsqueeze(0)

        if self.history is None:
            self.history = tensor
        else:
            self.history = torch.cat((self.history, tensor))


def load_model(filename, explainable=False, device=torch.device("cpu"), dataset="cifar"):
    if dataset == "cifar":
        model = load_cifar_model(filename, explainable, device)
    else:
        model = load_mnist_model(filename, explainable, device)

    return model


def load_mnist_model(filename, explainable=False, device=torch.device("cpu")):
    model = fairwasher.models.ConvNet('relu').to(device)

    if explainable:
        with torch.no_grad():
            model = networks.ExplainableNet(model, beta=200)

    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def load_cifar_model(filename, explainable=False, device=torch.device("cpu")):
    model = fairwasher.models.VGG16('relu').to(device)

    if explainable:
        with torch.no_grad():
            model = networks.ExplainableNet(model, beta=200)
    else:
        model.features = torch.nn.DataParallel(model.features)

    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def load_target_heatmap(filename):
    img = Image.open(filename).convert("1")
    img = np.array(invert(img), dtype='float32')
    img /= img.sum()
    return torch.from_numpy(img)


def get_heatmap(model, inputs, method, proj=None, absolute=True):
    if method in ('grad', 'xgrad'):
        heatmap = get_grad_heatmap(model, inputs)
    elif method == "intgrad":
        heatmap = get_intgrad_heatmap(model, inputs)
    elif method == "lrp":
        heatmap = get_lrp_heatmap(model, inputs)

    b, c, h, w = heatmap.shape
    if proj is not None:
        proj = proj.matmul(proj.transpose(1, 2))
        heatmap = torch.sum(proj.float() * heatmap.reshape(b, -1)[:, None, :], dim=-1)
        heatmap = heatmap.reshape(b, c, h, w)

    if method == 'xgrad':
        heatmap = heatmap * inputs.detach()

    # take abs sum over channels and normalize
    if absolute:
        heatmap = heatmap.abs()
    heatmap = heatmap.sum(1)
    heatmap /= heatmap.sum((1, 2), keepdims=True)
    return heatmap


def get_grad_heatmap(model, images):
    images.requires_grad = True
    output = model(images)
    pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
    out_rel = torch.eye(output.shape[1])[pred].to(images.device)
    one_hot = torch.sum(output * out_rel)

    grad, = torch.autograd.grad(one_hot, images, create_graph=True)

    return grad


def get_xgrad_heatmap(model, images):
    images.requires_grad = True
    output = model(images)
    pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
    out_rel = torch.eye(output.shape[1])[pred].to(images.device)
    one_hot = torch.sum(output * out_rel)
    grad, = torch.autograd.grad(one_hot, images, create_graph=True)

    result = grad * images.detach()

    return result


def get_intgrad_heatmap(model, images, steps=8, baseline=None):
    if baseline is None:
        baseline = torch.zeros_like(images)

    images.requires_grad = True
    gradient = None

    for alpha in list(np.linspace(1. / steps, 1.0, steps)):
        imgmod = baseline + (images - baseline) * alpha
        output = model(imgmod)
        pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
        out_rel = torch.eye(output.shape[1])[pred].to(images.device)
        one_hot = torch.sum(output * out_rel)
        _attr = torch.autograd.grad(one_hot, images, create_graph=True)

        if gradient is None:
            gradient = _attr
        else:
            gradient = [g + a for g, a in zip(gradient, _attr)]

    results = [g * (x - b) / steps for g, x, b in zip(gradient, images, baseline)]

    return results[0]


def get_lrp_heatmap(model, images):
    output = model(images)
    pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
    out_rel = torch.eye(output.shape[1])[pred].to(images.device)
    one_hot = output * out_rel
    heatmap = model.analyze("lrp", one_hot)
    return heatmap


def get_topk_acc(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def torch_to_numpy(tensor):
    if len(tensor.shape) == 4:
        numpy_array = tensor.permute(0, 2, 3, 1).contiguous().squeeze().detach().cpu().numpy()
    else:
        numpy_array = tensor.contiguous().squeeze().detach().cpu().numpy()

    return numpy_array


def calc_pcc(tensor1, tensor2, **kwargs):
    # permute if channels are first
    if not (isinstance(tensor1, np.ndarray) and isinstance(tensor2, np.ndarray)):
        array1 = torch_to_numpy(tensor1)
        array2 = torch_to_numpy(tensor2)

    return pearsonr(array1.reshape(-1), array2.reshape(-1), **kwargs)[0]


def calc_ssim(tensor1, tensor2, **kwargs):
    if not (isinstance(tensor1, np.ndarray) and isinstance(tensor2, np.ndarray)):
        if len(tensor1.shape) == 4:
            array1 = tensor1.permute(0, 2, 3, 1).contiguous().squeeze().detach().cpu().numpy()
            array2 = tensor2.permute(0, 2, 3, 1).contiguous().squeeze().detach().cpu().numpy()

        else:
            array1 = tensor1.contiguous().squeeze().detach().cpu().numpy()
            array2 = tensor2.contiguous().squeeze().detach().cpu().numpy()

    max_v = max(array1.max(), array2.max())
    min_v = min(array1.min(), array2.min())

    # check for 3 channel image
    if len(array1.shape) == 3:
        kwargs['multichannel'] = True

    return structural_similarity(array1, array2, data_range=max_v - min_v, **kwargs)


def estimate_jacobian_frobenius(x, output, n_mcsamples):
    norm_est = 0.
    for _ in range(n_mcsamples):
        vec = torch.randn_like(output)
        vjp, = torch.autograd.grad(output, x, grad_outputs=vec, create_graph=True)
        norm_est = norm_est + (vjp ** 2).sum()
    norm_est = norm_est / n_mcsamples / x.shape[0]
    return norm_est


def exact_jacobian_frobenius(batch, output):
    device = batch.device
    d = output.shape[1:]
    blen = batch.shape[0]
    D = batch.shape[1:]
    output = output.reshape(blen, np.prod(d))

    out_grads = torch.eye(np.prod(d).item(), device=device)
    jacobian = torch.empty(np.prod(d).item(), blen, np.prod(D).item(), device=device)

    for jac, eye in zip(jacobian, out_grads):
        jac[:] = torch.autograd.grad(
            output,
            batch,
            grad_outputs=eye[None].repeat(blen, 1),
            retain_graph=True,
            create_graph=True
        )[0].reshape(blen, np.prod(D))

    frobenius = (jacobian ** 2).sum() / blen
    return frobenius
