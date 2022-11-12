import os
import sys
import time
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

def add_to_class(Class):
    """Register functions as methods in a class after the class has been created"""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

class Accumulator:
    """accumulate on n variables, can be used to calculate metrics"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        """stop and record time"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """return average time"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """return total time"""
        return sum(self.times)

    def cumsum(self):
        """return cumulative time"""
        return np.array(self.times).cumsum().tolist()
