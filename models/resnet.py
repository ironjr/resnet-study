"""
resnet.py

Implementation of Residual Network introduced by He et al.(2015).
[https://arxiv.org/abs/1512.03385]

Jaerin Lee
Seoul National University
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import flatten


class BasicBlock(nn.Module):

    def __init__(self, out_channels, shortcut_type='identity',
                 normalization='batchnorm', pooling=False):
        """Basic building block of Residual Network.
        
        Args:
            in_channels (int): Number of channels going inside the block.
            shortcut_type (:obj:`str`, optional): Type of shortcut path defined
                in the paper. 'identity' and 'projection' is allowed. Default
                is 'identity'.
            normalization (:obj:`str`, optional): Type of normalization used
                between conv and relu layers. Currently 'batchnorm' and None
                is supported. Default is 'batchnorm'
            pooling (bool): If true, output dimension is doubled and feature
                size is halved. Default is False.
        """
        super().__init__()

        # Downsampling is performed if pooling is on.
        if pooling:
            in_channels = out_channels // 2
            stride = 2
            self.pool = nn.AvgPool2d(1, stride=2)
        else:
            in_channels = out_channels
            stride = 1
        self.in_channels = in_channels
        self.pooling = pooling

        # Use BN as default normalization method.
        if normalization == 'batchnorm':
            self.use_batchnorm = True
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.use_batchnorm = False

        # Support two types of shortcut specified in the paper.
        if shortcut_type == 'projection':
            self.identity_shortcut = False
            self.conv3 = nn.Conv2d(in_channels, out_channels, 1, stride,
                padding=0)
            nn.init.kaiming_normal_(self.conv3.weight)
            if self.use_batchnorm:
                self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            # Identity is used for default.
            self.identity_shortcut = True


        # Convolutional path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)

    def forward(self, x):
        if self.use_batchnorm:
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if not self.identity_shortcut: 
                x = self.conv3(x)
                x = self.bn3(x)
            elif self.pooling:
                x = self.pool(x)
                x = F.pad(x, (0, 0, 0, 0, 0, self.in_channels))
        else:
            out = self.conv1(x)
            out = F.relu(out)
            out = self.conv2(out)
            if not self.identity_shortcut: 
                x = self.conv3(x)
            elif self.pooling:
                x = self.pool(x)
                x = F.pad(x, (0, 0, 0, 0, 0, self.in_channels))
        out += x
        out = F.relu(out)
        return out


class BottleneckBlock(nn.Module):

    def __init__(self, out_channels, int_channels, shortcut_type='identity',
                 normalization='batchnorm', pooling=False):
        """Bottleneck block in ResNet reduces the total number of parameters.
        
        Args:
            out_channels (int): Number of channels going outside the block.
            int_channels (int): Number of channels in the core conv layer.
                Should be smaller than ext_channels.
            shortcut_type (:obj:`str`, optional): Type of shortcut path defined
                in the paper. 'identity' and 'projection' is allowed. Default
            normalization (:obj:`str`, optional): Type of normalization used
                between conv and relu layers. Currently 'batchnorm' and None
                is supported. Default is 'batchnorm'
                is 'identity'.
            pooling (bool): If true, output dimension is doubled and feature
                size is halved. Default is False.
        """
        super().__init__()

        # Downsampling is performed if pooling is on.
        if pooling:
            in_channels = out_channels // 2
            stride = 2
            self.pool = nn.AvgPool2d(1, stride=2)
        else:
            in_channels = out_channels
            stride = 1
        self.in_channels = in_channels
        self.pooling = pooling

        # Use BN as default normalization method.
        if normalization == 'batchnorm':
            self.use_batchnorm = True
            self.bn1 = nn.BatchNorm2d(int_channels)
            self.bn2 = nn.BatchNorm2d(int_channels)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.use_batchnorm = False

        # Support two types of shortcut specified in the paper.
        if shortcut_type == 'projection':
            self.identity_shortcut = False
            self.conv4 = nn.Conv2d(in_channels, out_channels, 1, stride,
                padding=0)
            nn.init.kaiming_normal_(self.conv4.weight)
            if self.use_batchnorm:
                self.bn4 = nn.BatchNorm2d(out_channels)
        else:
            # Identity is used for default.
            self.identity_shortcut = True

        # Convolutional path
        self.conv1 = nn.Conv2d(in_channels, int_channels, 1, stride, padding=0)
        self.conv2 = nn.Conv2d(int_channels, int_channels, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(int_channels, out_channels, 1, 1, padding=0)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)

    def forward(self, x):
        if self.use_batchnorm:
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = F.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)
            if not self.identity_shortcut: 
                x = self.conv4(x)
                x = self.bn4(x)
            elif self.pooling:
                x = self.pool(x)
                x = F.pad(x, (0, 0, 0, 0, 0, self.in_channels))
        else:
            out = self.conv1(x)
            out = F.relu(out)
            out = self.conv2(out)
            out = F.relu(out)
            out = self.conv3(out)
            if not self.identity_shortcut: 
                x = self.conv4(x)
            elif self.pooling:
                x = self.pool(x)
                x = F.pad(x, (0, 0, 0, 0, 0, self.in_channels))
        out += x
        out = F.relu(out)
        return out


class ResNetCIFAR10(nn.Module):

    def __init__(self, n, shortcut_type='identity', normalization='batchnorm'):
        """Demonstration of the network introduced in the original paper.
        
        Args:
            n (int): Parameter specifying the depth of the network. In
                specific, there are (1+2n, 2n, 2n) number of (16, 32, 64)
                convolutional layers each having feature map with the size of
                (32X32, 16X16, 8X8).
            shortcut_type (:obj:`str`, optional): Type of shortcut path defined
                in the paper. 'identity' and 'projection' is allowed. Default
                is 'identity'.
            normalization (:obj:`str`, optional): Type of normalization used
                between conv and relu layers. Currently 'batchnorm' and None
                is supported. Default is 'batchnorm'
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)

        # Use BN as default normalization method.
        if normalization == 'batchnorm':
            self.use_batchnorm = True
            self.bn1 = nn.BatchNorm2d(16)
        else:
            self.use_batchnorm = False

        self.res1 = nn.ModuleList()
        for _ in range(n):
            self.res1.append(BasicBlock(16, shortcut_type=shortcut_type,
                normalization=normalization))

        self.res2 = nn.ModuleList()
        self.res2.append(BasicBlock(32, pooling=True,
            shortcut_type=shortcut_type, normalization=normalization))
        for _ in range(n - 1):
            self.res2.append(BasicBlock(32, shortcut_type=shortcut_type,
                normalization=normalization))

        self.res3 = nn.ModuleList()
        self.res3.append(BasicBlock(64, pooling=True,
            shortcut_type=shortcut_type, normalization=normalization))
        for _ in range(n - 1):
            self.res3.append(BasicBlock(64, shortcut_type=shortcut_type,
                normalization=normalization))

        self.pool = nn.AvgPool2d(8)
        self.flatten = flatten.Flatten()
        self.linear = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.conv1(x)
        if self.use_batchnorm:
            out = self.bn1(out)
        out = F.relu(out)
        for net in self.res1:
            out = net(out)
        for net in self.res2:
            out = net(out)
        for net in self.res3:
            out = net(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.softmax(out)
        return out

if __name__ == '__main__':
    model = ResNetCIFAR10(3)
    x = torch.zeros((128, 3, 32, 32), dtype=torch.float32)
    scores = model(x)
    print(scores.size()) # Should give torch.Size([128, 10])
