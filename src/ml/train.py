"""Minimal training loop for research / Jupyter (not used by the Express server)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.config_io import load_config
from ml.dataset import CasiaPoseDataset
from ml.model import STGCN

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    root.addHandler(h)
    root.setLevel(logging.INFO)


def main() -> None:
    _setup_logging()
    ap = argparse.ArgumentParser(description="DeepGait ST-GCN training (research)")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", type=Path, default=Path("checkpoint.pth"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    joints = int(cfg.get("joints", 17))
    window = int(cfg.get("window_size", 64))
    emb = int(cfg.get("embedding_dim", 128))
    num_classes = int(cfg.get("num_train_classes", 100))

    ds = CasiaPoseDataset(
        root=args.data_root,
        window_size=window,
        joints=joints,
    )
    if len(ds) == 0:
        logger.warning("Empty dataset — generating dummy batch for smoke train")
        dummy_x = torch.randn(4, 2, window, joints)
        dummy_y = torch.randint(0, num_classes, (4,))

        model = STGCN(
            num_nodes=joints,
            embedding_dim=emb,
            temporal_kernel=int(cfg.get("temporal_kernel", 9)),
            hidden_channels=int(cfg.get("hidden_channels", 64)),
        ).to(device)
        head = nn.Linear(emb, num_classes).to(device)
        opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=args.lr)
        crit = nn.CrossEntropyLoss()
        model.train()
        for _ in range(args.epochs):
            opt.zero_grad()
            z = model(dummy_x.to(device))
            logits = head(z)
            loss = crit(logits, dummy_y.to(device))
            loss.backward()
            opt.step()
            logger.info("loss=%.4f", loss.item())
        torch.save({"model_state_dict": model.state_dict()}, args.save)
        logger.info("Saved %s", args.save)
        return

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    model = STGCN(
        num_nodes=joints,
        embedding_dim=emb,
        temporal_kernel=int(cfg.get("temporal_kernel", 9)),
        hidden_channels=int(cfg.get("hidden_channels", 64)),
    ).to(device)
    head = nn.Linear(emb, num_classes).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total = 0.0
        n = 0
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device) % num_classes
            opt.zero_grad()
            z = model(batch_x)
            logits = head(z)
            loss = crit(logits, batch_y)
            loss.backward()
            opt.step()
            total += loss.item() * batch_x.size(0)
            n += batch_x.size(0)
        logger.info("epoch %d loss=%.4f", epoch + 1, total / max(n, 1))

    torch.save({"model_state_dict": model.state_dict()}, args.save)
    logger.info("Saved %s", args.save)


if __name__ == "__main__":
    main()
