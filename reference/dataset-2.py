
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _load_raw_skeleton(pkl_path):
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

    # HRNet formats are often (T, V, 3): x, y, confidence.
    # Keep only x,y coordinates for this model.
    if skeleton.ndim == 3 and skeleton.shape[2] >= 3:
        skeleton = skeleton[:, :, :2]

    if skeleton.ndim != 3 or skeleton.shape[2] != 2:
        raise ValueError(f"Expected skeleton shape (T,V,2), got {skeleton.shape} in file: {pkl_path}")

    if skeleton.shape[0] == 0 or skeleton.shape[1] == 0:
        raise ValueError(f"Empty skeleton sequence in file: {pkl_path}")

    return skeleton


def _calc_angles(skel):
    angles = torch.zeros((1, skel.shape[1], skel.shape[2]), dtype=skel.dtype)
    triplets = [(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16), (5, 11, 13), (6, 12, 14)]

    for a, b, c in triplets:
        u = skel[:2, :, a] - skel[:2, :, b]
        v = skel[:2, :, c] - skel[:2, :, b]
        cos_sim = torch.clamp(F.cosine_similarity(u, v, dim=0), -1.0 + 1e-7, 1.0 - 1e-7)
        angles[0, :, b] = torch.acos(cos_sim)

    return angles


def _calc_distances(skel):
    distances = torch.zeros((1, skel.shape[1], skel.shape[2]), dtype=skel.dtype)

    mid_shoulder = (skel[:2, :, 5] + skel[:2, :, 6]) / 2.0
    mid_hip = (skel[:2, :, 11] + skel[:2, :, 12]) / 2.0
    torso = torch.clamp(torch.norm(mid_shoulder - mid_hip, dim=0), min=1e-5)

    pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
    for a, b in pairs:
        bone_len = torch.norm(skel[:2, :, a] - skel[:2, :, b], dim=0)
        distances[0, :, b] = bone_len / torso

    return distances


def _calc_kinematics(skel):
    vel = torch.zeros_like(skel)
    vel[:, 1:, :] = skel[:, 1:, :] - skel[:, :-1, :]

    acc = torch.zeros_like(vel)
    acc[:, 1:, :] = vel[:, 1:, :] - vel[:, :-1, :]
    return vel, acc


def preprocess_skeleton(sequence, sequence_length=64, is_training=False):
    skel = torch.from_numpy(sequence).permute(2, 0, 1).contiguous()

    if skel.shape[2] != 17:
        if skel.shape[2] > 17:
            skel = skel[:, :, :17]
        else:
            pad_v = 17 - skel.shape[2]
            skel = torch.cat([skel, torch.zeros((2, skel.shape[1], pad_v), dtype=skel.dtype)], dim=2)

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


class PairGaitDataset(Dataset):
    def __init__(
        self,
        data_path,
        sequence_length=64,
        positives_per_anchor=2,
        negatives_per_anchor=2,
        is_training=True,
        max_pairs=None,
    ):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.positives_per_anchor = positives_per_anchor
        self.negatives_per_anchor = negatives_per_anchor
        self.is_training = is_training

        self.subject_to_files = {}
        self.subjects = []
        self.pairs = []
        self.invalid_files = []

        self._collect_files()
        self._build_pairs(max_pairs=max_pairs)

        print(
            f"PairGaitDataset: identities={len(self.subjects)}, pairs={len(self.pairs)}, "
            f"training={self.is_training}"
        )
        if len(self.invalid_files) > 0:
            print(f"PairGaitDataset: skipped invalid files={len(self.invalid_files)}")

    def _collect_files(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")

        for subject in sorted(os.listdir(self.data_path)):
            subj_path = os.path.join(self.data_path, subject)
            if not os.path.isdir(subj_path):
                continue

            files = []
            for root, _, names in os.walk(subj_path):
                for name in names:
                    if name.endswith(".pkl"):
                        file_path = os.path.join(root, name)
                        try:
                            _load_raw_skeleton(file_path)
                            files.append(file_path)
                        except Exception:
                            self.invalid_files.append(file_path)

            if files:
                self.subject_to_files[subject] = sorted(files)

        self.subjects = sorted(self.subject_to_files.keys())

    def _build_pairs(self, max_pairs=None):
        rng = random.Random(42)
        pairs = []

        for subject in self.subjects:
            same_files = self.subject_to_files[subject]
            if len(same_files) == 0:
                continue

            other_subjects = [s for s in self.subjects if s != subject]
            if len(other_subjects) == 0:
                continue

            for anchor in same_files:
                if len(same_files) > 1:
                    positives = [f for f in same_files if f != anchor]
                    for _ in range(self.positives_per_anchor):
                        pos = rng.choice(positives)
                        pairs.append((anchor, pos, 1.0))

                for _ in range(self.negatives_per_anchor):
                    neg_subject = rng.choice(other_subjects)
                    neg = rng.choice(self.subject_to_files[neg_subject])
                    pairs.append((anchor, neg, 0.0))

        rng.shuffle(pairs)
        if max_pairs is not None and max_pairs > 0:
            pairs = pairs[:max_pairs]

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, y = self.pairs[idx]

        s1 = _load_raw_skeleton(p1)
        s2 = _load_raw_skeleton(p2)
        x1 = preprocess_skeleton(s1, sequence_length=self.sequence_length, is_training=self.is_training)
        x2 = preprocess_skeleton(s2, sequence_length=self.sequence_length, is_training=self.is_training)
        return x1, x2, torch.tensor(y, dtype=torch.float32)
