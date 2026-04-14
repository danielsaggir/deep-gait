"""Spatio-Temporal Graph Convolutional Network for 128-d gait embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    """1D convolution along the time dimension (ST-GCN temporal branch)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SpatialGraphConv(nn.Module):
    """Spatial graph convolution with learnable adjacency (N, C, T, V) -> same."""

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.A = nn.Parameter(torch.ones(num_nodes, num_nodes), requires_grad=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = self.conv(x)
        output = torch.einsum("nctv,vw->nctw", x_conv, self.A)
        return self.relu(self.bn(output))


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        temporal_kernel: int = 9,
    ) -> None:
        super().__init__()
        self.spatial = SpatialGraphConv(in_channels, out_channels, num_nodes)
        self.temporal = TemporalConv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        n, c, t, v = x.shape
        x = x.permute(0, 3, 1, 2).reshape(n * v, c, t)
        x = self.temporal(x)
        x = x.reshape(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        return x


class STGCN(nn.Module):
    """
    ST-GCN producing a fixed-size embedding (default 128-D), ported from DeepGait_STGCN.

    Expected input: (N, C, T, V) with C=2 (x,y), T=window_size, V=num_joints.
    """

    def __init__(
        self,
        num_nodes: int = 17,
        in_channels: int = 2,
        embedding_dim: int = 128,
        hidden_channels: int = 64,
        temporal_kernel: int = 9,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.block1 = STGCNBlock(
            in_channels, hidden_channels, num_nodes, temporal_kernel=temporal_kernel
        )
        self.block2 = STGCNBlock(
            hidden_channels, hidden_channels, num_nodes, temporal_kernel=temporal_kernel
        )
        self.fc = nn.Linear(hidden_channels * num_nodes, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = F.avg_pool2d(x, (x.size(2), 1))
        x = x.view(x.size(0), -1)
        return self.fc(x)
