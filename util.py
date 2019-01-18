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


BATCH_SIZE = 128

transform = T.Compose([
    T.ToTensor()
])

cifar10_train = dset.CIFAR10('./datasets', train=True, download=True,
    transform=transform)
cifar10_val = dset.CIFAR10('./datasets', train=True, download=True,
    transform=transform)
cifar10_test = dset.CIFAR10('./datasets', train=False, download=True, 
    transform=transform)

torch_datasets = (('training', cifar10_train),
    ('validataion', cifar10_val),
    ('test', cifar10_test))
for label, torch_dataset in torch_datasets:
    dataloader = torch.utils.data.DataLoader(torch_dataset,
        batch_size=BATCH_SIZE, shuffle=False)

    data_mean = [] # Mean of the dataset
    data_std0 = [] # std of dataset
    data_std1 = [] # std with ddof = 1
    for i, data in enumerate(dataloader, 0):
        # shape (batch_size, 3, height, width)
        numpy_image = data[0].numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)

        data_mean.append(batch_mean)
        data_std0.append(batch_std0)
        data_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    data_mean = np.array(data_mean).mean(axis=0)
    data_std0 = np.array(data_std0).mean(axis=0)
    data_std1 = np.array(data_std1).mean(axis=0)

    print(label, data_mean, data_std0, data_std1)