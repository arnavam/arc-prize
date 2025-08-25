import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, 32)

    def forward(self, x):


        x = add_channel_dim(x)
        x = F.relu(self.conv1(x))     # [B, 32, H, W]
        x = self.pool1(x)             # [B, 32, 1, 1]
        x = self.flatten(x)           # [B, 32]
        x = self.fc(x)                # [B, 32]
        return x

def add_channel_dim(tensor):
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tensor.ndim}D tensor instead.{tensor.shape}{tensor}")
    return tensor.unsqueeze(1)  # Add a channel dimension at position 1
