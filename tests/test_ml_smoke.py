"""Smoke tests for ml package (no GPU required)."""

import torch

from ml.model import STGCN


def test_stgcn_forward_shape():
    model = STGCN(num_nodes=17, embedding_dim=128, hidden_channels=64)
    x = torch.randn(2, 2, 64, 17)
    y = model(x)
    assert y.shape == (2, 128)
