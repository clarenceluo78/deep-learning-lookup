import os
import sys
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

import inspect
import collections
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

from . import utils, data, vis

def accuracy(y_hat, y):
    """calculate accuracy"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """evaluate accuracy of a model on the given dataset"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # set eval mode
    metric = utils.Accumulator(2)  # use Accumulator to sum up tp 2 variables
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def evaluate_loss(net, data_iter, loss):
    """evaluate loss of a model on the given dataset"""
    metric = d2l.Accumulator(2)  # loss, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def predict_mnist(net, test_iter, n=6):
    """predict label for minst"""
    for X, y in test_iter:
        break
    trues = data.get_fashion_mnist_labels(y)
    preds = data.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    vis.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])