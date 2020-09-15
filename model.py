#!/usr/bin/env python

# import torch
from torch import nn


class PosENet(nn.Module):
    """Position Encoding Module."""
    def __init__(self, input_dim):
        super(PosENet, self).__init__()
        self.conv = nn.Conv2d(input_dim, 1, (3, 3), stride=1, padding=0)
        # torch.manual_seed(3)
        nn.init.xavier_uniform_(self.conv.weight, gain=1)

    def forward(self, x):
        return self.conv(x)
