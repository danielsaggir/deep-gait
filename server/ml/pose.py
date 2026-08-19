"""Raw video → COCO-17 skeleton sequences compatible with the trained verifier.

YOLO11-pose emits native COCO-17 keypoints in pixel coordinates (y-down), matching
the CASIA-B HRNet joint convention. Coordinates are isotropically scaled so image
width maps to 320 (CASIA-B native width). Frames are sampled near 25 fps.

Low-confidence detections are kept (training used HRNet x,y even when confidence
was low). Frames with no person are omitted from the model sequence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ml.constants import (
    CASIA_IMAGE_WIDTH,
    MAX_INFERENCE_WIDTH,
    MIN_DETECTED_FRAMES,
    MIN_DETECTION_COVERAGE,
    NUM_JOINTS,
    TARGET_FPS,
    YOLO_IMGSZ,
    YOLO_POSE_WEIGHTS,
)
from ml.errors import InsufficientGaitDataError, VideoDecodeError

logger = logging.getLogger(__name__)

_yolo_model = None


@dataclass
class PoseExtraction:
    skeleton: np.ndarray  # (T, 17, 2) pixel coords scaled to CASIA width
    pose_frames: list[dict]  # visualization frames in original video space
    width: int
    height: int
    fps: float
    duration: float
    sampled_frames: int
    frames_detected: int
    frames_used: int
    coverage: float
    source_name: str = ""
    extra: dict = field(default_factory=dict)


def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        weights_path = Path(YOLO_POSE_WEIGHTS)
        weights = str(weights_path) if weights_path.is_file() else "yolo11n-pose.pt"
        logger.info("Loading YOLO pose weights: %s", weights)
        _yolo_model = YOLO(weights)
    return _yolo_model


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _select_person(
    boxes: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    prev_box: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if boxes.size == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if prev_box is not None:
        ious = np.array([_iou(prev_box, b) for b in boxes], dtype=np.float32)
        best_iou_i = int(np.argmax(ious))
        if float(ious[best_iou_i]) >= 0.3:
            i = best_iou_i
        else:
            i = int(np.argmax(areas))
    else:
        i = int(np.argmax(areas))
    return boxes[i], keypoints_xy[i], keypoints_conf[i]


def _downscale_for_inference(
    frame: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, float]:
    """Return a smaller frame for YOLO and a multiplier back to source pixel coords."""
    if width <= MAX_INFERENCE_WIDTH:
        return frame, 1.0
    ratio = MAX_INFERENCE_WIDTH / float(width)
    infer_w = MAX_INFERENCE_WIDTH
    infer_h = max(1, int(round(height * ratio)))
    resized = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
    return resized, width / float(infer_w)


def extract_pose_from_video(
    video_path: str | Path,
    *,
    subject: str,
    device: str | None = None,
) -> PoseExtraction:
    path = Path(video_path)
    if not path.is_file():
        raise VideoDecodeError(f"Video not found: {path}", subject=subject)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoDecodeError(f"Unable to read video: {path.name}", subject=subject)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 1e-3:
        fps = 25.0
    duration = frame_count / fps if frame_count > 0 else 0.0
    if width <= 0 or height <= 0:
        cap.release()
        raise VideoDecodeError(f"Video has invalid dimensions: {path.name}", subject=subject)

    yolo = _load_yolo()
    predict_kwargs = {"verbose": False, "imgsz": YOLO_IMGSZ}
    if device:
        predict_kwargs["device"] = device

    if width > MAX_INFERENCE_WIDTH:
        logger.info(
            "Downscaling %s from %dx%d to max width %d before pose inference",
            path.name,
            width,
            height,
            MAX_INFERENCE_WIDTH,
        )

    scale = CASIA_IMAGE_WIDTH / float(width)
    target_dt = 1.0 / TARGET_FPS
    next_t = 0.0
    sampled = 0
    detected_xy: list[np.ndarray] = []
    pose_frames: list[dict] = []
    prev_box: np.ndarray | None = None
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_index / fps
            frame_index += 1
            if timestamp + 1e-6 < next_t:
                continue
            next_t += target_dt
            sampled += 1

            infer_frame, coord_back = _downscale_for_inference(frame, width, height)
            results = yolo.predict(infer_frame, **predict_kwargs)
            result = results[0]
            boxes_obj = result.boxes
            kps_obj = result.keypoints

            selected = None
            if boxes_obj is not None and kps_obj is not None and len(boxes_obj) > 0:
                boxes = boxes_obj.xyxy.cpu().numpy() * coord_back
                xy = kps_obj.xy.cpu().numpy() * coord_back
                conf = (
                    kps_obj.conf.cpu().numpy()
                    if kps_obj.conf is not None
                    else np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)
                )
                if xy.ndim == 3 and xy.shape[-1] >= 2:
                    selected = _select_person(boxes, xy, conf, prev_box)

            joints_viz: list[dict] = []
            if selected is None:
                prev_box = None
                for _ in range(NUM_JOINTS):
                    joints_viz.append({"x": 0.0, "y": 0.0, "confidence": 0.0})
                pose_frames.append(
                    {
                        "timestamp": float(timestamp),
                        "detected": False,
                        "joints": joints_viz,
                    }
                )
                continue

            box, xy, conf = selected
            prev_box = box
            if xy.shape[0] < NUM_JOINTS:
                padded = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
                padded[: xy.shape[0]] = xy[:, :2]
                xy = padded
                conf_p = np.zeros((NUM_JOINTS,), dtype=np.float32)
                conf_p[: conf.shape[0]] = conf[: NUM_JOINTS]
                conf = conf_p
            else:
                xy = xy[:NUM_JOINTS, :2].astype(np.float32)
                conf = conf[:NUM_JOINTS].astype(np.float32)

            model_xy = xy * scale
            detected_xy.append(model_xy)

            for j in range(NUM_JOINTS):
                joints_viz.append(
                    {
                        "x": float(xy[j, 0] / width),
                        "y": float(xy[j, 1] / height),
                        "confidence": float(conf[j]),
                    }
                )
            pose_frames.append(
                {
                    "timestamp": float(timestamp),
                    "detected": True,
                    "joints": joints_viz,
                }
            )
    finally:
        cap.release()

    if sampled == 0:
        raise VideoDecodeError(f"No frames decoded from {path.name}", subject=subject)

    frames_detected = len(detected_xy)
    coverage = frames_detected / sampled
    if frames_detected < MIN_DETECTED_FRAMES or coverage < MIN_DETECTION_COVERAGE:
        raise InsufficientGaitDataError(
            "Unable to establish a reliable 17-joint gait sequence.",
            subject=subject,
        )

    skeleton = np.stack(detected_xy, axis=0).astype(np.float32)
    return PoseExtraction(
        skeleton=skeleton,
        pose_frames=pose_frames,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        sampled_frames=sampled,
        frames_detected=frames_detected,
        frames_used=int(skeleton.shape[0]),
        coverage=float(coverage),
        source_name=path.name,
    )
