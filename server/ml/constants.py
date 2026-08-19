"""Trained DeepGait contract. Values match reference/best_gait_verifier.pth."""

from pathlib import Path

NUM_JOINTS = 17
IN_CHANNELS = 8
SEQUENCE_LENGTH = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
DECISION_THRESHOLD = 0.5

# CASIA-B native capture. Pose xy is isotropically scaled so image width → 320.
CASIA_IMAGE_WIDTH = 320.0
TARGET_FPS = 25.0

# Pose inference runs on frames downscaled to this width (aspect ratio kept). Full-resolution
# decode of 1080p/4K clips dominates RAM; YOLO already uses imgsz=640 internally.
MAX_INFERENCE_WIDTH = 1280
YOLO_IMGSZ = 640

MIN_DETECTED_FRAMES = 16
MIN_DETECTION_COVERAGE = 0.25

# Physical body edges for visualization (no self-loops). Same links as the ST-GCN graph.
SKELETON_EDGES = [
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

ML_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ML_DIR / "weights" / "best_gait_verifier.pth"
YOLO_POSE_WEIGHTS = str(ML_DIR / "weights" / "yolo11n-pose.pt")
