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
    metric = utils.Accumulator(2)  # loss, no. of examples
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

def sensitive_analysis(X, y, repeat_times, model):
    silhouette_list = []
    rand_index_list = []
    print('The repeated initialization times of each algorithm is set to:', repeat_times)
    print('Sensitivity analysis starts, please wait...')
    if (model == 'Kmeans'):
        # k-means
        for i in range(repeat_times):
            KMeans_ = KMeans(num_clusters=3, max_iterations=100, random_seed=i, show_epochs=False)
            y_pred = KMeans_.fit(X)
            silhouette_list.append(silhouette_coef(X, y_pred))
            rand_index_list.append(rand_index(y, y_pred))
        
        print("Sensitive analysis of K-means:")
        print("----Silhouette evaluation score: ", np.var(silhouette_list))
        print("----Rand index score: ", np.var(rand_index_list))
        
        silhouette_list.clear()
        rand_index_list.clear()
    elif (model == 'accKmeans'):
        # accelerated k-means
        for i in range(repeat_times):
            accKMeans_ = accKMeans(num_clusters=3, max_iterations=100, random_seed=i, show_epochs=False)
            accKMeans_.fit(X)
            add_one = np.array(np.ones(X.shape[0]))
            accKMeans_result = np.int64(accKMeans_.assignments + add_one)
            silhouette_list.append(silhouette_coef(X, accKMeans_result))
            rand_index_list.append(rand_index(y, accKMeans_result))
            
        print("Sensitive analysis of Accelerated K-means:")
        print("----Silhouette evaluation score: ", np.var(silhouette_list))
        print("----Rand index score: ", np.var(rand_index_list))
        
        silhouette_list.clear()
        rand_index_list.clear()
    elif (model == 'GMM'):
        # gmm
        for i in range(repeat_times):
            GMM_ = GMM(num_clusters=3, tolerance=1e-3, random_seed=i, show_epochs=False)
            GMM_.fit(X)
            add_one = np.array(np.ones(X.shape[0]))
            GMM_result = np.int64(GMM_.assignments + add_one)
            silhouette_list.append(silhouette_coef(X, GMM_result))
            rand_index_list.append(rand_index(y, GMM_result))
            
        print("Sensitive analysis of GMM-EM:")
        print("----Silhouette evaluation score: ", np.var(silhouette_list))
        print("----Rand index score: ", np.var(rand_index_list))
        
        silhouette_list.clear()
        rand_index_list.clear()
    else:
        raise NotImplementedError
