"""
util.py

Jaerin Lee
Seoul National University
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np


transform = T.Compose([
    T.ToTensor()
])

cifar10_train = dset.CIFAR10('./datasets', train=True, download=True,
    transform=transform)
cifar10_test = dset.CIFAR10('./datasets', train=False, download=True, 
    transform=transform)

# Training data
print(cifar10_train.train_data.shape)
print(cifar10_train.train_data.mean(axis=(0, 1, 2)) / 255)
print(cifar10_train.train_data.std(axis=(0, 1, 2)) / 255)

# Test data
print(cifar10_test.test_data.shape)
print(cifar10_test.test_data.mean(axis=(0, 1, 2)) / 255)
print(cifar10_test.test_data.std(axis=(0, 1, 2)) / 255)
