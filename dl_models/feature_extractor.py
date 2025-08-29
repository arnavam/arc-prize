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
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 3 * 3, 32)


    def forward(self, x):
            x = self.add_channel_dim(x)
            x = F.relu(self.conv1(x))     # [B, 32, H, W]
            x = self.pool(x)              # [B, 32, 3, 3]
            x = self.flatten(x)           # [B, 288]
            x = self.fc(x)                # [B, 32]
            return x


    def add_channel_dim(self, tensor):

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)  # Add channel dimension at position 1
        elif tensor.ndim != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got {tensor.ndim}D tensor instead: {tensor.shape}")  # Add channel dimension at position 1

        # Check if spatial dimensions are (1, 1)
        height, width = tensor.shape[2], tensor.shape[3]

        # Add padding for problematic cases
        if (height, width) == (1, 1):
            tensor = F.pad(tensor, (1, 1, 1, 1))  # pad all sides
        elif (height, width) == (2, 1):
            tensor = F.pad(tensor, (1, 1, 0, 0))  # pad left/right only
        elif (height, width) == (1, 2):
            tensor = F.pad(tensor, (0, 0, 1, 1))  # pad top/bottom only

        return tensor
