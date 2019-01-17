import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        # No internal parameters

    def forward(self, x):
        return x.view(x.shape[0], -1)