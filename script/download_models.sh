#!/usr/bin/env sh
MODELS="cifar_acc.pth mnist_intgrad.pth fmnist_acc.pth mnist_lrp.pth fmnist_lrp.pth cifar_lrp.pth cifar_intgrad.pth mnist_acc.pth fmnist_intgrad.pth defended/cifar_acc.pth defended/mnist_intgrad.pth defended/fmnist_acc.pth defended/mnist_lrp.pth defended/fmnist_lrp.pth defended/cifar_lrp.pth defended/cifar_intgrad.pth defended/mnist_acc.pth defended/fmnist_intgrad.pth defended/fmnist_grad.pth defended/cifar_xgrad.pth defended/cifar_grad.pth defended/fmnist_xgrad.pth defended/mnist_grad.pth defended/mnist_xgrad.pth fmnist_grad.pth cifar_xgrad.pth cifar_grad.pth fmnist_xgrad.pth mnist_grad.pth mnist_xgrad.pth"

mkdir -p share/models
cd share/models
curl "http://doc.ml.tu-berlin.de/fairwashing/models.zip" -Lo models.zip
rm -f $MODELS
unzip models.zip $MODELS
rm models.zip
