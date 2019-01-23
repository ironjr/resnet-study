"""
main.py

Main routine for training and testing various network topology using CIFAR-10
dataset. 

Jaerin Lee
Seoul National University
"""

# TODO
# Move hyperparameter definitions as well as training options into a
# different file.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as T

# Log for tensorboard statistics
from logger import Logger
logger_train = Logger('./logs/train')
logger_val = Logger('./logs/val')

# Overcome laziness of managing checkpoints
from os import mkdir
from shutil import copy
from datetime import datetime


# Define hyperparameters
mode = 'test'
use_gpu = True
try_new = True
num_train = 50000
batch_size = 128
num_iter = 32000
iteration_begins = 391 * 0
print_every = 100
learning_rate = 0.1
weight_decay = 0.0001
momentum = 0.9

# Number of epochs are determined with other three parameters
num_epochs = (num_iter * batch_size + num_train - 1) // num_train
# ... or choose number of epochs manually
#  num_epochs = 100

# Define transforms
# Original paper followed data augmentation method by Deeply Supervised Net by
# Lee et al. (2015) [http://proceedings.mlr.press/v38/lee15a.pdf]
# Mean and variance of each set is evaluated by utils.py
transform_train = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.Normalize(mean=(0.49139968, 0.48215841, 0.44653091),
        #  std=(0.24703223, 0.24348513, 0.26158784)),
        std=(0.2023, 0.1994, 0.2010)),
])
transform_val = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.49139968, 0.48215841, 0.44653091),
        #  std=(0.24703223, 0.24348513, 0.26158784)),
        std=(0.2023, 0.1994, 0.2010)),
])
transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.49421428, 0.48513139, 0.45040909),
        std=(0.24665252, 0.24289226, 0.26159238)),
])

# Load CIFAR-10 dataset
print('Loading dataset CIFAR-10 ...')
cifar10_train = dset.CIFAR10('./datasets', train=True, download=True,
    transform=transform_train)
loader_train = DataLoader(cifar10_train, batch_size=batch_size,
    sampler=sampler.SubsetRandomSampler(range(num_train)))
cifar10_val = dset.CIFAR10('./datasets', train=True, download=True,
    transform=transform_val)
loader_val = DataLoader(cifar10_val, batch_size=batch_size,
    sampler=sampler.SubsetRandomSampler(range(num_train, 50000)))
cifar10_test = dset.CIFAR10('./datasets', train=False, download=True, 
    transform=transform_test)
loader_test = DataLoader(cifar10_test, batch_size=batch_size)
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print('Done!')

# Set device (GPU or CPU)
if use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
    # import torch.backends.cudnn as cudnn
    # cudnn.benchmark = True
else:
    device = torch.device('cpu')
print('Using device:', device)


# Set network model
from models import resnet
model = resnet.ResNetCIFAR10(n=3)

# Define new optimizer specified by hyperparameters defined above
# optimizer = optim.Adam(model.parameters(),
#                        lr=learning_rate,
#                        weight_decay=weight_decay)
optimizer = optim.SGD(model.parameters(),
                      lr=learning_rate,
                      momentum=momentum,
                      weight_decay=weight_decay)

#  print(model.parameters())

# Load previous model
if not try_new:
    print('PyTorch is currently the loading model ... ', end='')

    model.load_state_dict(torch.load('model.pth'))
    model.cuda()

    print('Done!')

    # Optimizer is loaded when we continue training
    if mode == 'train':
        print('PyTorch is currently loading the optimizer ... ', end='')

        optimizer.load_state_dict(torch.load('optimizer.pth'))
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        print('Done!')

# Overwrite current optimizer settings
for param_group in optimizer.param_groups:
    param_group['lr'] = learning_rate
    param_group['momentum'] = momentum
    param_group['weight_decay'] = weight_decay


# Train/Test the model
from optimizer import train, test
if mode == 'train':
    train(model, optimizer, loader_train, loader_val=loader_val,
        num_epochs=num_epochs, logger_train=logger_train, logger_val=logger_val,
        print_every=print_every, iteration_begins=iteration_begins)

    print('PyTorch is currently saving the model and the optimizer ...', end='')

    # Save model to checkpoint
    torch.save(model.state_dict(), 'model.pth')
    torch.save(optimizer.state_dict(), 'optimizer.pth')

    # Archive current model to checkpoints folder
    dirname = './checkpoints/' + datetime.today().strftime('%Y%m%d-%H%M%S')
    mkdir(dirname)
    copy('model.pth', dirname)
    copy('optimizer.pth', dirname)

    print('Done!')
elif mode == 'test':
    test(model, loader_test)
