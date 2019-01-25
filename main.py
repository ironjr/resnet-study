"""
main.py

Main routine for training and testing various network topology using CIFAR-10
dataset. 

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

# Per-parameter settings for torch.optim.Optimizer
from util import group_weight

# Overcome laziness of managing checkpoints
from os import mkdir
from shutil import copy
from datetime import datetime

# External hyper-parameters and running environment settings
import argparse
import json
from copy import deepcopy


def main(args):
    # Get hyper-parameters from arguments
    label = args.label
    mode = args.mode
    use_tb = args.use_tb
    use_gpu = args.use_gpu
    try_new = args.try_new
    num_train = args.num_train
    batch_size = args.batch_size
    num_iters = args.num_iters
    num_epochs = args.num_epochs
    iteration_begins = args.iter_init
    print_every = args.print_every
    learning_rate = args.lr
    weight_decay = args.weight_decay
    momentum = args.momentum

    # Number of epochs are determined with other three parameters if not directly
    # specified
    if num_epochs is -1:
        num_epochs = (num_iters * batch_size + num_train - 1) // num_train

    # Log for tensorboard statistics
    logger_train = None
    logger_val = None
    if use_tb:
        from logger import Logger
        if label is None:
            dirname = './logs/' + datetime.today().strftime('%Y%m%d-%H%M%S')
        else:
            dirname = './logs/' + label
        mkdir(dirname)
        logger_train = Logger(dirname + '/train')
        logger_val = Logger(dirname + '/val')


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
    model = resnet.ResNetCIFAR10(n=9)

    # Define new optimizer specified by hyperparameters defined above
    # optimizer = optim.Adam(model.parameters(),
    #                        lr=learning_rate,
    #                        weight_decay=weight_decay)
    optimizer = optim.SGD(group_weight(model),
                          lr=learning_rate,
                          momentum=momentum,
                          weight_decay=weight_decay)
                        #   nesterov=True)

    # Load previous model
    if not try_new or mode == 'test':
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

    # Overwrite default hyperparameters for new run
    group_decay, group_no_decay = optimizer.param_groups
    group_decay['lr'] = learning_rate
    group_decay['momentum'] = momentum
    group_decay['weight_decay'] = weight_decay
    group_no_decay['lr'] = learning_rate
    group_no_decay['momentum'] = momentum
    optimizer.defaults['lr'] = learning_rate
    optimizer.defaults['momentum'] = momentum
    optimizer.defaults['weight_decay'] = weight_decay

    # Train/Test the model
    from optimizer import train, test
    if mode == 'train':
        train(model, optimizer, loader_train, loader_val=loader_test, # changes
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


# Define parser at the end of file for readibility
if __name__ == '__main__':
    # Parameterize the running envionment to run with a shell script
    parser = argparse.ArgumentParser(
        description='Run PyTorch on the classification problem.')
    parser.add_argument('--label', dest='label', type=str, default=None,
        help='name of run and folder where logs are to be stored')
    parser.add_argument('--mode', dest='mode', type=str, default='train',
        help='choose run mode between training and test')
    parser.add_argument('--schedule', dest='schedule_file', type=str, default=None,
        help='running schedule in json format')
    parser.add_argument('--use-tb', dest='use_tb', type = bool, default=True,
        help='use tensorboard logging')
    parser.add_argument('--try-new', dest='try_new', type=bool, default=True,
        help='choose whether use newly initialized model or not')
    parser.add_argument('--use-gpu', dest='use_gpu', type=bool, default=True,
        help='allow CUDA to accelerate run')
    parser.add_argument('--num-train', dest='num_train', type=int, default=50000,
        help='number of training set with maximum of 50000')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1,
        help='batch size for training')
    parser.add_argument('--num-iters', dest='num_iters', type=int, default=1,
        help='number of iterations to train; will be overrided by \'--num-epochs\'')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=-1,
        help='number of epochs to train; overrides \'--num-iters\'')
    parser.add_argument('--iter-init', dest='iter_init', type=int, default=0,
        help='a point where iteration count begins for tensorboard stats')
    parser.add_argument('--print-every', dest='print_every', type=int, default=1,
        help='intermediate result evaluation period')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-1,
        help='learning rate for training')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float,
        default=1e-4, help='weight decay for training with SGD optimizer')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
        help='momentum for training with SGD optimizer')
    args = parser.parse_args()

    # Get schedule from the json file
    # --- Structure of schedule.json --- #
    # list(dict(arguments) as event) as schedule
    # Arguments can be 'lr' 'weight_decay' 'momentum' 'num_iters' 'num_epochs'
    # 'mode'
    if args.schedule_file is not None:
        if '.json' in args.schedule_file:
            with open(args.schedule_file) as f:
                contents = json.load(f)

                iterations = 0
                iteration_begins = 0
                it_per_epoch = (args.num_train + args.batch_size - 1) // \
                    args.batch_size
                for event in contents['schedule']:
                    args_instance = deepcopy(args)
                    if 'mode' in event:
                        args_instance.mode = event['mode']
                    if 'lr' in event:
                        args_instance.lr = event['lr']
                    if 'weight_decay' in event:
                        args_instance.weight_decay = event['weight_decay']
                    if 'momentum' in event:
                        args_instance.momentum = event['momentum']
                    if 'num_iters' in event:
                        args_instance.num_iters = event['num_iters']
                        iterations = ((event['num_iters'] + it_per_epoch - 1) // \
                            it_per_epoch) * it_per_epoch
                    if 'num_epochs' in event:
                        args_instance.num_epochs = event['num_epochs']
                        iterations = event['num_epochs'] * it_per_epoch

                    # Use existing model and optimizer after the first run
                    if iteration_begins is not 0:
                        args_instance.try_new = False

                    # Get the count of total iterations passed
                    args_instance.iter_init = iteration_begins

                    # Run once
                    main(args_instance)

                    # Post iteration count
                    iteration_begins += iterations
