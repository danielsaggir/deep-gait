"""Siamese ST-GCN gait verifier.

Architecture is identical to reference/model.py. Production default in_channels is 8
because that is what reference/best_gait_verifier.pth was trained with. The research
file defaults to 11, which cannot load this checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.constants import EMBEDDING_DIM, HIDDEN_DIM, IN_CHANNELS, NUM_JOINTS


def get_physical_adjacency_matrix(num_nodes: int = NUM_JOINTS) -> torch.Tensor:
    neighbor_links = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (11, 12),
        (5, 11),
        (6, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for i, j in neighbor_links:
        if i < num_nodes and j < num_nodes:
            A[i, j] = 1.0
            A[j, i] = 1.0

    for i in range(num_nodes):
        A[i, i] = 1.0

    row_sum = A.sum(dim=1, keepdim=True)
    A = A / row_sum

    return A


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_nodes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        A_init = get_physical_adjacency_matrix(num_nodes)
        self.A = nn.Parameter(A_init, requires_grad=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = self.conv(x)
        output = torch.einsum("nctv,vw->nctw", x_conv, self.A)
        return self.relu(self.bn(output))


class TemporalConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class STGCN_Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        temporal_kernel: int = 9,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.spatial = SpatialGraphConv(in_channels, out_channels, num_nodes)
        self.temporal = TemporalConv(
            out_channels,
            out_channels,
            temporal_kernel,
            padding=temporal_kernel // 2,
            dropout=dropout,
        )

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.spatial(x)

        n, c, t, v = x.shape
        x = x.permute(0, 3, 1, 2).reshape(n * v, c, t)
        x = self.temporal(x)
        x = x.reshape(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        return x + res


class DeepGait_STGCN(nn.Module):
    def __init__(
        self,
        num_nodes: int = NUM_JOINTS,
        in_channels: int = IN_CHANNELS,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.block1 = STGCN_Block(in_channels, 64, num_nodes)
        self.block2 = STGCN_Block(64, 128, num_nodes)
        self.block3 = STGCN_Block(128, 256, num_nodes)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class SiameseGaitVerifier(nn.Module):
    def __init__(
        self,
        num_nodes: int = NUM_JOINTS,
        in_channels: int = IN_CHANNELS,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.encoder = DeepGait_STGCN(
            num_nodes=num_nodes,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)

        abs_diff = torch.abs(z1 - z2)
        prod = z1 * z2
        cos = F.cosine_similarity(z1, z2, dim=1, eps=1e-8).unsqueeze(1)

        fused = torch.cat([abs_diff, prod, cos], dim=1)
        logits = self.classifier(fused)
        return logits, z1, z2
