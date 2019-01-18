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
logger = Logger('./logs')
# from tensorboardX import SummaryWriter
# writer = SummaryWriter()


# Define hyperparameters
USE_GPU = True
TRY_NEW = False
NUM_TRAIN = 45000
BATCH_SIZE = 128
PRINT_EVERY = 100
learning_rate = 0.1
weight_decay = 0.0001 # Weight decay is changed relative to learning rate
num_epochs = 50
momentum = 0.9

# Define transforms
# Original paper followed data augmentation method by Deeply Supervised Net by
# Lee et al. (2015) [http://proceedings.mlr.press/v38/lee15a.pdf]
# Mean and variance of each set is evaluated by utils.py
transform_train = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    # T.Pad(4),
    # T.TenCrop(32),
    # T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
    T.Normalize(mean=(0.49141386, 0.48216975, 0.44654447),
        std=(0.24668841, 0.24316198, 0.261165)),
])
transform_val = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.49141386, 0.48216975, 0.44654447),
        std=(0.24668841, 0.24316198, 0.261165)),
])
transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.49413028, 0.48513925, 0.4504057),
        std=(0.2463779, 0.24270386, 0.26123637)),
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
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
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

# Define new optimizer specified by hyperparameters defined above
# optimizer = optim.Adam(model.parameters(),
#                        lr=learning_rate,
#                        weight_decay=weight_decay)
optimizer = optim.SGD(model.parameters(),
                      lr=learning_rate,
                      momentum=momentum,
                      weight_decay=weight_decay)

# Load previous model
if not TRY_NEW:
    print('PyTorch is currently the loading model ...', end='')
    model.load_state_dict(torch.load('model.pth'))
    model.cuda()
    print('Done!')
    print('PyTorch is currently loading the optimizer ...', end='')
    optimizer.load_state_dict(torch.load('optimizer.pth'))
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    print('Done!')

for param_group in optimizer.param_groups:
    param_group['lr'] = learning_rate
    param_group['momentum'] = momentum
    param_group['weight_decay'] = weight_decay


# Train the model with logging
from optimizer import train, test
train(model, optimizer, loader_train, loader_val=loader_val,
      num_epochs=num_epochs, logger=logger)

# Save model to checkpoint
# TODO Maybe differentiate the model name?
print('PyTorch is currently saving the model and the optimizer ...', end='')
torch.save(model.state_dict(), 'model.pth')
torch.save(optimizer.state_dict(), 'optimizer.pth')
print('Done!')