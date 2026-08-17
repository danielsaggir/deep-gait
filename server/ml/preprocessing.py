"""Training-compatible skeleton preprocessing.

Numerics match reference/dataset-2.py. Production inference always uses
is_training=False (no flip / scale / noise; center-crop or zero-pad to 64).
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml.constants import NUM_JOINTS, SEQUENCE_LENGTH


def load_raw_skeleton(pkl_path: str | Path) -> np.ndarray:
    pkl_path = str(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if "joint_2d" in data:
            skeleton = data["joint_2d"]
        elif "skeleton" in data:
            skeleton = data["skeleton"]
        elif "keypoints" in data:
            skeleton = data["keypoints"]
        else:
            skeleton = next(iter(data.values()))
    elif isinstance(data, list):
        skeleton = data[0]
    else:
        skeleton = data

    skeleton = np.asarray(skeleton, dtype=np.float32)

    if skeleton.ndim == 2:
        if skeleton.shape[1] % 2 != 0:
            raise ValueError(f"Invalid flattened keypoints shape for file: {pkl_path}")
        skeleton = skeleton.reshape(skeleton.shape[0], -1, 2)

    if skeleton.ndim == 3 and skeleton.shape[2] >= 3:
        skeleton = skeleton[:, :, :2]

    if skeleton.ndim != 3 or skeleton.shape[2] != 2:
        raise ValueError(
            f"Expected skeleton shape (T,V,2), got {skeleton.shape} in file: {pkl_path}"
        )

    if skeleton.shape[0] == 0 or skeleton.shape[1] == 0:
        raise ValueError(f"Empty skeleton sequence in file: {pkl_path}")

    return skeleton


def _calc_angles(skel: torch.Tensor) -> torch.Tensor:
    angles = torch.zeros((1, skel.shape[1], skel.shape[2]), dtype=skel.dtype)
    triplets = [
        (5, 7, 9),
        (6, 8, 10),
        (11, 13, 15),
        (12, 14, 16),
        (5, 11, 13),
        (6, 12, 14),
    ]

    for a, b, c in triplets:
        u = skel[:2, :, a] - skel[:2, :, b]
        v = skel[:2, :, c] - skel[:2, :, b]
        cos_sim = torch.clamp(F.cosine_similarity(u, v, dim=0), -1.0 + 1e-7, 1.0 - 1e-7)
        angles[0, :, b] = torch.acos(cos_sim)

    return angles


def _calc_distances(skel: torch.Tensor) -> torch.Tensor:
    distances = torch.zeros((1, skel.shape[1], skel.shape[2]), dtype=skel.dtype)

    mid_shoulder = (skel[:2, :, 5] + skel[:2, :, 6]) / 2.0
    mid_hip = (skel[:2, :, 11] + skel[:2, :, 12]) / 2.0
    torso = torch.clamp(torch.norm(mid_shoulder - mid_hip, dim=0), min=1e-5)

    pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
    for a, b in pairs:
        bone_len = torch.norm(skel[:2, :, a] - skel[:2, :, b], dim=0)
        distances[0, :, b] = bone_len / torso

    return distances


def _calc_kinematics(skel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    vel = torch.zeros_like(skel)
    vel[:, 1:, :] = skel[:, 1:, :] - skel[:, :-1, :]

    acc = torch.zeros_like(vel)
    acc[:, 1:, :] = vel[:, 1:, :] - vel[:, :-1, :]
    return vel, acc


def preprocess_skeleton(
    sequence: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
    is_training: bool = False,
) -> torch.Tensor:
    skel = torch.from_numpy(np.asarray(sequence, dtype=np.float32)).permute(2, 0, 1).contiguous()

    if skel.shape[2] != NUM_JOINTS:
        if skel.shape[2] > NUM_JOINTS:
            skel = skel[:, :, :NUM_JOINTS]
        else:
            pad_v = NUM_JOINTS - skel.shape[2]
            skel = torch.cat(
                [skel, torch.zeros((2, skel.shape[1], pad_v), dtype=skel.dtype)], dim=2
            )

    pelvis = (skel[:2, :, 11] + skel[:2, :, 12]) / 2.0
    skel[:2, :, :] = skel[:2, :, :] - pelvis.unsqueeze(2)

    angles = _calc_angles(skel)
    distances = _calc_distances(skel)
    vel, acc = _calc_kinematics(skel)

    x = torch.cat([skel, angles, distances, vel, acc], dim=0)

    if is_training:
        if random.random() < 0.5:
            x[0, :, :] *= -1.0
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            x[:2, :, :] *= scale
        if random.random() < 0.4:
            noise = torch.randn_like(x[:2, :, :]) * 0.01
            x[:2, :, :] += noise

    t = x.shape[1]
    if t > sequence_length:
        if is_training:
            start = random.randint(0, t - sequence_length)
        else:
            start = (t - sequence_length) // 2
        x = x[:, start : start + sequence_length, :]
    elif t < sequence_length:
        pad = torch.zeros((x.shape[0], sequence_length - t, x.shape[2]), dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)

    return x


def temporal_features(x: torch.Tensor) -> dict[str, list[float]]:
    """Real per-frame motion summaries from the 8-channel tensor (C, T, V)."""
    vel = x[4:6]
    mag = torch.sqrt(vel[0] ** 2 + vel[1] ** 2)
    overall = mag.mean(dim=1).tolist()
    lower = mag[:, 11:17].mean(dim=1).tolist()
    return {
        "velocityMagnitude": [float(v) for v in overall],
        "lowerBodyMotion": [float(v) for v in lower],
    }


def feature_composition(x: torch.Tensor) -> dict[str, float]:
    return {
        "position": float(x[:2].abs().mean().item()),
        "angles": float(x[2].abs().mean().item()),
        "proportions": float(x[3].abs().mean().item()),
        "velocity": float(x[4:6].abs().mean().item()),
        "acceleration": float(x[6:8].abs().mean().item()),
    }
