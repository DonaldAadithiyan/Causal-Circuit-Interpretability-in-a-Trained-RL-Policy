"""
IMPALA CNN architecture:
  3 convolutional blocks (ConvSequence), each with:
    Conv2d → MaxPool2d → ResidualBlock × 2
  ReLU → Flatten → Linear(hidden_size)
"""

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x + residual


class ConvSequence(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(nn.Module):
    def __init__(self, obs_shape: tuple, hidden_size: int = 256):
        """obs_shape: (C, H, W)"""
        super().__init__()
        in_channels = obs_shape[0]

        self.seq1 = ConvSequence(in_channels, 16)
        self.seq2 = ConvSequence(16, 32)
        self.seq3 = ConvSequence(32, 32)

        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            flat = self._forward_conv(dummy)
            conv_out_size = flat.shape[1]

        self.linear = nn.Linear(conv_out_size, hidden_size)
        self.hidden_size = hidden_size
        self.conv_out_size = conv_out_size

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = torch.relu(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        x = self.linear(x)
        x = torch.relu(x)
        return x


class ImpalaCNNExtractor(BaseFeaturesExtractor):
    """SB3-compatible feature extractor wrapping ImpalaCNN."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        h, w, c = observation_space.shape  # SB3 stores (H, W, C)
        self.cnn = ImpalaCNN(obs_shape=(c, h, w), hidden_size=features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 passes (N, C, H, W) float32 in [0,1]
        return self.cnn(observations)
