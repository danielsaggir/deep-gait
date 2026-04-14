"""BlazePose (MediaPipe) extraction + centering, scaling, temporal cleaning, 64-frame window."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# MediaPipe Pose (33 landmarks) -> 17 joints (COCO-style body subset for gait)
MP_TO17 = np.array(
    [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28], dtype=np.int64
)
NUM_JOINTS = 17


def _build_pose():
    import mediapipe as mp

    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


class GaitProcessor:
    """
    Video -> (1, 2, T, V) float tensor with T == window_size, V == num_joints.

    Pipeline: extract 2D normalized image coords, map to 17 joints, temporal repair,
    per-frame center (hip mid) + scale (torso length), resample/pad to window_size.
    """

    def __init__(
        self,
        window_size: int = 64,
        num_joints: int = NUM_JOINTS,
        visibility_threshold: float = 0.5,
    ) -> None:
        if num_joints != NUM_JOINTS:
            raise ValueError(
                f"num_joints must be {NUM_JOINTS} for the built-in BlazePose mapping"
            )
        self.window_size = window_size
        self.num_joints = num_joints
        self.visibility_threshold = visibility_threshold
        self._pose = None

    def _ensure_pose(self):
        if self._pose is None:
            logger.info("Initializing MediaPipe Pose (BlazePose backend)")
            self._pose = _build_pose()

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def process_video(self, video_path: str | Path) -> torch.Tensor:
        """Return tensor (1, 2, window_size, num_joints) on CPU."""
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")

        self._ensure_pose()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        frames_xy: list[np.ndarray] = []
        frames_vis: list[np.ndarray] = []
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                res = self._pose.process(rgb)
                xy = np.zeros((self.num_joints, 2), dtype=np.float32)
                vis = np.zeros((self.num_joints,), dtype=np.float32)
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    for j, mi in enumerate(MP_TO17):
                        p = lm[int(mi)]
                        xy[j, 0] = p.x
                        xy[j, 1] = p.y
                        vis[j] = p.visibility
                frames_xy.append(xy)
                frames_vis.append(vis)
        finally:
            cap.release()

        if not frames_xy:
            raise RuntimeError(f"No frames read from video: {path}")

        seq = np.stack(frames_xy, axis=0)
        vseq = np.stack(frames_vis, axis=0)
        logger.info("Extracted %d frames from %s", seq.shape[0], path.name)

        seq = self._temporal_clean(seq, vseq)
        seq = self._center_scale(seq)
        seq = self._resample_window(seq)
        # (T, V, 2) -> (1, 2, T, V)
        t = torch.from_numpy(seq).permute(2, 0, 1).unsqueeze(0).contiguous()
        return t

    def _temporal_clean(self, seq: np.ndarray, vis: np.ndarray) -> np.ndarray:
        """Linear interpolation for low-visibility joints."""
        t, v, _ = seq.shape
        out = seq.copy()
        for j in range(v):
            bad = vis[:, j] < self.visibility_threshold
            if not np.any(bad):
                continue
            good = ~bad
            if not np.any(good):
                out[:, j, :] = 0.0
                continue
            idx = np.arange(t)
            for c in range(2):
                out[:, j, c] = np.interp(idx, idx[good], seq[good, j, c])
        return out

    def _center_scale(self, seq: np.ndarray) -> np.ndarray:
        """Per-frame: subtract hip midpoint; divide by torso length."""
        out = np.zeros_like(seq)
        # indices 11, 12 = left/right hip in our 17-joint order
        for i in range(seq.shape[0]):
            lh = seq[i, 11]
            rh = seq[i, 12]
            mid_hip = (lh + rh) * 0.5
            ls = seq[i, 5]
            rs = seq[i, 6]
            mid_sh = (ls + rs) * 0.5
            torso = np.linalg.norm(mid_sh - mid_hip)
            scale = max(float(torso), 1e-6)
            out[i] = (seq[i] - mid_hip) / scale
        return out

    def _resample_window(self, seq: np.ndarray) -> np.ndarray:
        """(T, V, 2) -> (window_size, V, 2) by uniform resampling or padding."""
        t, v, c = seq.shape
        w = self.window_size
        if t == w:
            return seq
        if t < w:
            idx = (np.arange(w, dtype=np.float32) * (t - 1) / max(w - 1, 1)).round().astype(
                np.int64
            )
            return seq[idx]
        # t > w: uniform indices across time
        idx = np.linspace(0, t - 1, w, dtype=np.float32).round().astype(np.int64)
        return seq[idx]
