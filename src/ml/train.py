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


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    crit: nn.Module,
) -> tuple[float, float]:
    """Returns (mean loss, accuracy)."""
    model.eval()
    head.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        z = model(batch_x)
        logits = head(z)
        loss = crit(logits, batch_y)
        total_loss += loss.item() * batch_x.size(0)
        pred = logits.argmax(dim=-1)
        correct += (pred == batch_y).sum().item()
        n += batch_x.size(0)
    mean_loss = total_loss / max(n, 1)
    acc = correct / max(n, 1)
    return mean_loss, acc


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
    ap.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of data for validation (0 disables val split).",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    joints = int(cfg.get("joints", 17))
    window = int(cfg.get("window_size", 64))
    emb = int(cfg.get("embedding_dim", 128))
    cfg_classes = int(cfg.get("num_train_classes", 100))

    ds = CasiaPoseDataset(
        root=args.data_root,
        window_size=window,
        joints=joints,
    )
    if len(ds) == 0:
        logger.warning("Empty dataset — generating dummy batch for smoke train")
        num_classes = cfg_classes
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

    max_label = max(ds._label_from_path(p) for p in ds.paths)
    num_classes = max(cfg_classes, max_label + 1)
    if num_classes > cfg_classes:
        logger.info(
            "num_train_classes raised from config %d to %d (max label in data=%d)",
            cfg_classes,
            num_classes,
            max_label,
        )

    n_total = len(ds)
    val_frac = float(args.val_fraction)
    if val_frac > 0 and n_total >= 2:
        n_val = max(1, int(round(n_total * val_frac)))
        n_val = min(n_val, n_total - 1)
        n_train = n_total - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        logger.info(
            "train/val split: %d train, %d val (val_fraction=%.3f)",
            n_train,
            n_val,
            val_frac,
        )
    else:
        train_ds = ds
        val_ds = None
        if val_frac > 0:
            logger.info("validation disabled (need >=2 samples and val-fraction>0)")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if val_ds is not None
        else None
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
        model.train()
        head.train()
        total = 0.0
        correct = 0
        n = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            opt.zero_grad()
            z = model(batch_x)
            logits = head(z)
            loss = crit(logits, batch_y)
            loss.backward()
            opt.step()
            total += loss.item() * batch_x.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == batch_y).sum().item()
            n += batch_x.size(0)
        train_loss = total / max(n, 1)
        train_acc = correct / max(n, 1)

        if val_loader is not None:
            val_loss, val_acc = _eval_epoch(model, head, val_loader, device, crit)
            logger.info(
                "epoch %d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                epoch + 1,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )
        else:
            logger.info(
                "epoch %d train_loss=%.4f train_acc=%.4f",
                epoch + 1,
                train_loss,
                train_acc,
            )

    torch.save({"model_state_dict": model.state_dict()}, args.save)
    logger.info("Saved %s", args.save)


if __name__ == "__main__":
    main()
