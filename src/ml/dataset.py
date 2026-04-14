"""PyTorch Dataset for CASIA-B Pose–style sequences on disk."""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Optional: reuse project root / processed path from deep_gait package
try:
    from deep_gait.dataset import processed_casia_dir
except ImportError:
    processed_casia_dir = None  # type: ignore[misc, assignment]


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        z = np.load(path)
        keys = list(z.keys())
        if not keys:
            raise ValueError(f"Empty npz: {path}")
        return z[keys[0]]
    if path.suffix.lower() == ".pkl":
        with open(path, "rb") as f:
            o = pickle.load(f)
        if not isinstance(o, np.ndarray):
            raise ValueError(f"Expected ndarray in pkl, got {type(o)}: {path}")
        return o
    raise ValueError(f"Unsupported format: {path}")


def _to_ctv(arr: np.ndarray, joints: int, window: int) -> torch.Tensor:
    """Normalize arbitrary layout to (2, T, V)."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3:
        c0, c1, c2 = a.shape
        # (C, T, V)
        if c0 == 2 and c1 == window and c2 == joints:
            x = a
        elif c0 == joints and c1 == window and c2 == 2:
            x = np.transpose(a, (2, 1, 0))
        elif c0 == window and c1 == joints and c2 == 2:
            x = np.transpose(a, (2, 0, 1))
        elif c1 == joints and c2 == 3:
            # (T, V, 3) e.g. HRNet x,y,score — use xy only
            xy = a[:, :, :2]
            x = np.transpose(xy, (2, 0, 1))
        else:
            # (T, V, C)
            if c2 == 2:
                x = np.transpose(a, (2, 0, 1))
            else:
                raise ValueError(f"Cannot infer layout from shape {a.shape}")
    else:
        raise ValueError(f"Expected 3D array, got shape {a.shape}")

    if x.shape[0] != 2 or x.shape[2] != joints:
        raise ValueError(f"Expected (2, T, {joints}), got {x.shape}")
    if x.shape[1] != window:
        # resample time dim
        t = x.shape[1]
        idx = np.linspace(0, t - 1, window, dtype=np.float32).round().astype(np.int64)
        x = x[:, idx, :]
    return torch.from_numpy(x.copy())


class CasiaPoseDataset(Dataset):
    """
    Loads ``.npy`` / ``.npz`` / ``.pkl`` under a root directory (default: CASIA-B processed).

    Arrays: ``(2, T, V)``, or ``(T, V, 2)``, or ``(T, V, 3)`` (xy + extra); ``T`` may
    differ from ``window_size`` (resampled).

    Labels: ``nm-XXX`` / ``cl-XXX`` in the path; else CASIA-style ``.../CASIA-B_HRNet/<id>/...``;
    else hash of stem (weak supervision).
    """

    _label_re = re.compile(r"(?:nm|cl)-(\d+)", re.I)

    def __init__(
        self,
        root: Path | str | None = None,
        window_size: int = 64,
        joints: int = 17,
    ) -> None:
        super().__init__()
        if root is None:
            if processed_casia_dir is None:
                raise ImportError(
                    "Install the deep-gait package or pass root= to CasiaPoseDataset"
                )
            root = processed_casia_dir()
        self.root = Path(root)
        self.window_size = window_size
        self.joints = joints
        self.paths: list[Path] = []
        for pat in ("**/*.npy", "**/*.npz", "**/*.pkl"):
            self.paths.extend(sorted(self.root.glob(pat)))
        self.paths = sorted(set(self.paths))
        if not self.paths:
            logger.warning("No .npy/.npz/.pkl files under %s", self.root)

    def __len__(self) -> int:
        return len(self.paths)

    def _label_from_path(self, p: Path) -> int:
        s = p.as_posix()
        m = self._label_re.search(s)
        if m:
            return int(m.group(1))
        parts = p.parts
        for i, part in enumerate(parts):
            if part == "CASIA-B_HRNet" and i + 1 < len(parts):
                nxt = parts[i + 1]
                if nxt.isdigit():
                    return int(nxt)
        return hash(p.stem) % 100_003

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        p = self.paths[idx]
        arr = _load_array(p)
        x = _to_ctv(arr, self.joints, self.window_size)
        y = self._label_from_path(p)
        return x, y
