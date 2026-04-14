"""Smoke tests for ml package (no GPU required)."""

import pickle

import numpy as np
import torch

from ml.dataset import CasiaPoseDataset
from ml.model import STGCN


def test_stgcn_forward_shape():
    model = STGCN(num_nodes=17, embedding_dim=128, hidden_channels=64)
    x = torch.randn(2, 2, 64, 17)
    y = model(x)
    assert y.shape == (2, 128)


def test_casia_pkl_tv3_resample(tmp_path):
    """HRNet-style (T, V, 3) pickle -> (2, window, V)."""
    f = tmp_path / "nm-001_test.pkl"
    arr = np.random.randn(20, 17, 3).astype(np.float32)
    with open(f, "wb") as fp:
        pickle.dump(arr, fp)
    ds = CasiaPoseDataset(root=tmp_path, window_size=64, joints=17)
    assert len(ds) == 1
    x, y = ds[0]
    assert x.shape == (2, 64, 17)
    assert y == 1
