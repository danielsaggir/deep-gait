"""CLI: two videos (or skeleton pickles) → one JSON document on stdout."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ml.constants import (
    DECISION_THRESHOLD,
    DEFAULT_CHECKPOINT,
    EMBEDDING_DIM,
    IN_CHANNELS,
    NUM_JOINTS,
    SEQUENCE_LENGTH,
    SKELETON_EDGES,
)
from ml.errors import CheckpointError, DeepGaitMLError
from ml.model import SiameseGaitVerifier
from ml.pose import extract_pose_from_video
from ml.preprocessing import (
    feature_composition,
    load_raw_skeleton,
    preprocess_skeleton,
    temporal_features,
)

logger = logging.getLogger("ml.inference")


def _setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_verifier(checkpoint: Path, device: torch.device) -> SiameseGaitVerifier:
    if not checkpoint.is_file():
        raise CheckpointError(f"Checkpoint not found: {checkpoint}")

    model = SiameseGaitVerifier(
        num_nodes=NUM_JOINTS,
        in_channels=IN_CHANNELS,
        embedding_dim=EMBEDDING_DIM,
    )
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise CheckpointError(f"Checkpoint is incompatible with SiameseGaitVerifier: {exc}") from exc

    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint %s on %s", checkpoint, device)
    return model


def _pose_quality(extraction) -> dict:
    return {
        "framesDetected": extraction.frames_detected,
        "framesUsed": extraction.frames_used,
        "framesSampled": extraction.sampled_frames,
        "coverage": round(extraction.coverage, 4),
    }


def _subject_payload(extraction, features: torch.Tensor, embedding: list[float]) -> dict:
    gait = temporal_features(features)
    return {
        "metadata": {
            "source": extraction.source_name,
            "duration": round(extraction.duration, 3),
            "width": extraction.width,
            "height": extraction.height,
            "fps": round(extraction.fps, 3),
            "format": Path(extraction.source_name).suffix.lstrip(".").upper() or "UNKNOWN",
        },
        "poseQuality": _pose_quality(extraction),
        "poseFrames": extraction.pose_frames,
        "gaitSignature": gait,
        "featureComposition": feature_composition(features),
        "embedding": embedding,
        "skeletonEdges": SKELETON_EDGES,
    }


CASIA_FALLBACK_W = 320.0
CASIA_FALLBACK_H = 240.0


def _pickle_extraction(pkl_path: Path, _subject: str):
    from ml.pose import PoseExtraction

    seq = load_raw_skeleton(pkl_path)
    t = int(seq.shape[0])
    pose_frames = []
    for i in range(t):
        joints = []
        for j in range(min(NUM_JOINTS, seq.shape[1])):
            joints.append(
                {
                    "x": float(seq[i, j, 0] / max(CASIA_FALLBACK_W, 1.0)),
                    "y": float(seq[i, j, 1] / max(CASIA_FALLBACK_H, 1.0)),
                    "confidence": 1.0,
                }
            )
        while len(joints) < NUM_JOINTS:
            joints.append({"x": 0.0, "y": 0.0, "confidence": 0.0})
        pose_frames.append({"timestamp": i / 25.0, "detected": True, "joints": joints})

    return PoseExtraction(
        skeleton=seq.astype(np.float32),
        pose_frames=pose_frames,
        width=int(CASIA_FALLBACK_W),
        height=int(CASIA_FALLBACK_H),
        fps=25.0,
        duration=t / 25.0,
        sampled_frames=t,
        frames_detected=t,
        frames_used=t,
        coverage=1.0,
        source_name=pkl_path.name,
    )


def run_analysis(
    path_a: Path,
    path_b: Path,
    checkpoint: Path,
    *,
    skeleton_mode: bool = False,
) -> dict:
    t0 = time.perf_counter()
    device = resolve_device()
    device_name = "cuda" if device.type == "cuda" else "cpu"

    t_pose = time.perf_counter()
    if skeleton_mode:
        ext_a = _pickle_extraction(path_a, "A")
        ext_b = _pickle_extraction(path_b, "B")
    else:
        ext_a = extract_pose_from_video(path_a, subject="A", device=device_name)
        ext_b = extract_pose_from_video(path_b, subject="B", device=device_name)
    pose_s = time.perf_counter() - t_pose

    t_pre = time.perf_counter()
    x1 = preprocess_skeleton(ext_a.skeleton, sequence_length=SEQUENCE_LENGTH, is_training=False)
    x2 = preprocess_skeleton(ext_b.skeleton, sequence_length=SEQUENCE_LENGTH, is_training=False)
    pre_s = time.perf_counter() - t_pre

    t_inf = time.perf_counter()
    model = load_verifier(checkpoint, device)
    with torch.no_grad():
        logits, z1, z2 = model(x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device))
        prob = float(torch.sigmoid(logits)[0, 0].item())
        cosine = float(torch.nn.functional.cosine_similarity(z1, z2)[0].item())
        emb1 = z1.squeeze(0).detach().cpu().float().tolist()
        emb2 = z2.squeeze(0).detach().cpu().float().tolist()
    inf_s = time.perf_counter() - t_inf

    verdict = "LIKELY_MATCH" if prob >= DECISION_THRESHOLD else "LIKELY_DIFFERENT"
    total = time.perf_counter() - t0

    return {
        "result": {
            "samePersonProbability": prob,
            "cosineSimilarity": cosine,
            "threshold": DECISION_THRESHOLD,
            "verdict": verdict,
        },
        "subjectA": _subject_payload(ext_a, x1, emb1),
        "subjectB": _subject_payload(ext_b, x2, emb2),
        "model": {
            "architecture": "SiameseGaitVerifier",
            "embeddingDimension": EMBEDDING_DIM,
            "inputChannels": IN_CHANNELS,
            "sequenceLength": SEQUENCE_LENGTH,
            "joints": NUM_JOINTS,
            "device": device_name,
        },
        "timing": {
            "poseExtraction": round(pose_s, 4),
            "preprocessing": round(pre_s, 4),
            "inference": round(inf_s, 4),
            "total": round(total, 4),
        },
    }


def main() -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(description="DeepGait Siamese inference")
    parser.add_argument("--video-a", type=Path, default=None)
    parser.add_argument("--video-b", type=Path, default=None)
    parser.add_argument("--skeleton-a", type=Path, default=None)
    parser.add_argument("--skeleton-b", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    skeleton_mode = args.skeleton_a is not None or args.skeleton_b is not None
    if skeleton_mode:
        if args.skeleton_a is None or args.skeleton_b is None:
            print(
                json.dumps(
                    {
                        "error": {
                            "code": "INVALID_REQUEST",
                            "message": "Both --skeleton-a and --skeleton-b are required.",
                        }
                    }
                ),
                flush=True,
            )
            return 2
        path_a, path_b = args.skeleton_a, args.skeleton_b
    else:
        if args.video_a is None or args.video_b is None:
            print(
                json.dumps(
                    {
                        "error": {
                            "code": "INVALID_REQUEST",
                            "message": "Both --video-a and --video-b are required.",
                        }
                    }
                ),
                flush=True,
            )
            return 2
        path_a, path_b = args.video_a, args.video_b

    try:
        result = run_analysis(path_a, path_b, args.checkpoint, skeleton_mode=skeleton_mode)
    except DeepGaitMLError as exc:
        logger.error("%s", exc)
        print(json.dumps({"error": exc.to_dict()}), flush=True)
        return 2
    except Exception as exc:
        logger.exception("Inference failed")
        print(
            json.dumps({"error": {"code": "INFERENCE_FAILURE", "message": "Analysis failed."}}),
            flush=True,
        )
        return 1

    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
