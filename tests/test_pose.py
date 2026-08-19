"""Pose-bridge tests that do not require YOLO weights."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from ml.constants import CASIA_IMAGE_WIDTH, NUM_JOINTS  # noqa: E402
from ml.errors import InsufficientGaitDataError, VideoDecodeError  # noqa: E402
from ml.pose import extract_pose_from_video  # noqa: E402


def _write_blank_video(path: Path, frames: int = 10, fps: int = 25, size=(320, 240)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    assert writer.isOpened()
    for i in range(frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:] = (i * 7) % 40
        writer.write(frame)
    writer.release()


def test_missing_video_raises():
    with pytest.raises(VideoDecodeError) as exc:
        extract_pose_from_video("/tmp/deepgait-does-not-exist.mp4", subject="A")
    assert exc.value.subject == "A"


def test_blank_video_insufficient_gait(tmp_path, monkeypatch):
    video = tmp_path / "blank.mp4"
    _write_blank_video(video)

    class FakeBoxes:
        xyxy = None

        def __len__(self):
            return 0

    class FakeKps:
        xy = None
        conf = None

    class FakeResult:
        boxes = FakeBoxes()
        keypoints = FakeKps()

    class FakeYOLO:
        def predict(self, *_a, **_k):
            return [FakeResult()]

    import ml.pose as pose_mod

    monkeypatch.setattr(pose_mod, "_load_yolo", lambda: FakeYOLO())
    with pytest.raises(InsufficientGaitDataError) as exc:
        extract_pose_from_video(video, subject="B")
    assert exc.value.subject == "B"
    assert exc.value.code == "INSUFFICIENT_GAIT_DATA"


def test_mocked_detections_scale_and_shape(tmp_path, monkeypatch):
    video = tmp_path / "walk.mp4"
    _write_blank_video(video, frames=30, fps=25, size=(640, 480))

    xy_arr = np.zeros((1, 17, 2), dtype=np.float32)
    for j in range(17):
        xy_arr[0, j] = [100 + j, 80 + j]
    boxes_arr = np.array([[80, 60, 200, 300]], dtype=np.float32)
    conf_arr = np.ones((1, 17), dtype=np.float32)

    class Tensorish:
        def __init__(self, arr):
            self._arr = arr

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    class FakeBoxes:
        def __init__(self):
            self.xyxy = Tensorish(boxes_arr)

        def __len__(self):
            return 1

    class FakeKps:
        def __init__(self):
            self.xy = Tensorish(xy_arr)
            self.conf = Tensorish(conf_arr)

    class FakeResult:
        boxes = FakeBoxes()
        keypoints = FakeKps()

    class FakeYOLO:
        def predict(self, *_a, **_k):
            return [FakeResult()]

    import ml.pose as pose_mod

    monkeypatch.setattr(pose_mod, "_load_yolo", lambda: FakeYOLO())
    extraction = extract_pose_from_video(video, subject="A")
    assert extraction.skeleton.shape[1] == NUM_JOINTS
    assert extraction.skeleton.shape[2] == 2
    assert extraction.frames_detected >= 16
    expected = 100.0 * (CASIA_IMAGE_WIDTH / 640.0)
    assert extraction.skeleton[0, 0, 0] == pytest.approx(expected)
    assert extraction.pose_frames[0]["joints"][0]["x"] == pytest.approx(100 / 640)
    assert extraction.pose_frames[0]["timestamp"] >= 0
