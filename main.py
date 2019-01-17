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

# Use logger
from logger import Logger


# Define hyperparameters
NUM_TRAIN = 45000
BATCH_SIZE = 128
USE_GPU = True
PRINT_EVERY = 100
learning_rate = 0.003
weight_decay = 0.000
num_epochs = 40
# momentum = 0.9

# Define transforms
transform_train = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_val = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 dataset
print('Loading dataset CIFAR-10 ...')
cifar10_train = dset.CIFAR10('./datasets', train=True, download=True,
                             transform=transform_train)
loader_train = DataLoader(cifar10_train, batch_size=BATCH_SIZE,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
cifar10_val = dset.CIFAR10('./datasets', train=True, download=True,
                           transform=transform_val)
loader_val = DataLoader(cifar10_val, batch_size=BATCH_SIZE,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
cifar10_test = dset.CIFAR10('./datasets', train=False, download=True, 
                            transform=transform_test)
loader_test = DataLoader(cifar10_test, batch_size=BATCH_SIZE)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print('Done!')

# Set device (GPU or CPU)
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    # import torch.backends.cudnn as cudnn
    # cudnn.benchmark = True
else:
    device = torch.device('cpu')
print('Using device:', device)


# Set network model
# TODO Move this to resnet.py file
from models import resnet
from models.layers import flatten
model = nn.Sequential(
    # First layer
    nn.Conv2d(3, 16, 3, 1, padding=1),
    resnet.BasicBlock(16),
    resnet.BasicBlock(16),
    resnet.BasicBlock(16),
    resnet.BasicBlock(32, pooling=True),
    resnet.BasicBlock(32),
    resnet.BasicBlock(32),
    resnet.BasicBlock(64, pooling=True),
    resnet.BasicBlock(64),
    resnet.BasicBlock(64),
    nn.AvgPool2d(8),
    flatten.Flatten(),
    nn.Linear(64, 10),
    nn.Softmax(dim=1),
)

# Test code
# x = torch.zeros((128, 3, 32, 32), dtype=dtype)
# scores = model(x)
# print(scores.size()) # Should give torch.Size([128, 10])

# Load previous model
print('PyTorch is currently loading model ...')
model.load_state_dict(torch.load('model.pth'))
model.eval()
print('Done!')


# Log for tensorboard statistics
logger = Logger('./logs')

# Define new optimizer specified by hyperparameters defined above
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate,
                       weight_decay=weight_decay)

# Train the model with logging
from optimizer import train, test
train(model, optimizer, loader_train, loader_val=loader_val,
      num_epochs=num_epochs, logger=logger)

# Save model to checkpoint
# TODO Maybe differentiate the model name?
torch.save(model.state_dict(), 'model.pth')