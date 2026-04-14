"""CLI: video + checkpoint -> one-line JSON signature (for Node bridge)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from ml.config_io import load_config
from ml.model import STGCN
from ml.processor import GaitProcessor

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(h)
    root.setLevel(logging.INFO)


def load_model(
    checkpoint: Path | str,
    config: dict,
    device: torch.device,
) -> STGCN:
    joints = int(config.get("joints", 17))
    window = int(config.get("window_size", 64))
    emb = int(config.get("embedding_dim", 128))
    temporal_kernel = int(config.get("temporal_kernel", 9))
    hidden = int(config.get("hidden_channels", 64))

    model = STGCN(
        num_nodes=joints,
        in_channels=2,
        embedding_dim=emb,
        hidden_channels=hidden,
        temporal_kernel=temporal_kernel,
    )
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    logger.info("Loaded weights from %s", checkpoint)
    return model


def run_inference(
    video: Path,
    checkpoint: Path,
    config_path: Path,
    device: torch.device | None = None,
) -> list[float]:
    cfg = load_config(config_path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proc = GaitProcessor(
        window_size=int(cfg.get("window_size", 64)),
        num_joints=int(cfg.get("joints", 17)),
        visibility_threshold=float(cfg.get("visibility_threshold", 0.5)),
    )
    x = proc.process_video(video)

    model = load_model(checkpoint, cfg, device)
    x = x.to(device)
    with torch.no_grad():
        emb = model(x)
    vec = emb.squeeze(0).detach().cpu().float().tolist()
    return vec


def main() -> None:
    _setup_logging()
    p = argparse.ArgumentParser(description="DeepGait ST-GCN inference")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config.yaml (default: repo root config.yaml next to cwd)",
    )
    args = p.parse_args()

    root = Path.cwd()
    config_path = args.config
    if config_path is None:
        for candidate in (root / "config.yaml", root.parent / "config.yaml"):
            if candidate.is_file():
                config_path = candidate
                break
        if config_path is None:
            raise SystemExit("Pass --config or place config.yaml in cwd or parent")

    vec = run_inference(args.video, args.checkpoint, Path(config_path))
    print(json.dumps(vec), flush=True)


if __name__ == "__main__":
    main()
