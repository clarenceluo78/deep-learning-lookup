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

from . import metrics, utils, vis

def sgd(params, lr, batch_size):
    """mini-batch sgd"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def train_epoch_image(net, train_iter, loss, updater):
    """one epoch of training"""
    # set train mode
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = utils.Accumulator(3)
    for X, y in train_iter:
        # calculate gradient and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # use custom updater and loss function
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), metrics.accuracy(y_hat, y), y.numel())
    # return train loss and train accuracy
    return metric[0] / metric[2], metric[1] / metric[2]

def train_image_cpu(net, train_iter, test_iter, loss, num_epochs, updater):
    """train model"""
    animator = vis.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_image(net, train_iter, loss, updater)
        test_acc = metrics.evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def squared_loss(y_hat, y):
    """square loss"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2