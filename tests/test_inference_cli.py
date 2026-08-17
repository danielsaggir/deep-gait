from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"


def test_pickle_cli_inference():
    tree = ROOT / "data" / "processed" / "casia_b_hrnet"
    pkls = sorted(tree.rglob("*.pkl")) if tree.exists() else []
    usable = [p for p in pkls if p.stat().st_size > 1000][:2]
    if len(usable) < 2:
        pytest.skip("CASIA-B pickles not present")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.inference",
            "--skeleton-a",
            str(usable[0]),
            "--skeleton-b",
            str(usable[1]),
            "--checkpoint",
            str(SERVER / "ml" / "weights" / "best_gait_verifier.pth"),
        ],
        cwd=str(ROOT),
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(SERVER)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "result" in payload
    prob = payload["result"]["samePersonProbability"]
    assert 0.0 <= prob <= 1.0
    assert payload["result"]["threshold"] == 0.5
    assert payload["result"]["verdict"] in ("LIKELY_MATCH", "LIKELY_DIFFERENT")
    assert payload["model"]["inputChannels"] == 8
    assert len(payload["subjectA"]["embedding"]) == 128
    assert len(payload["subjectA"]["gaitSignature"]["velocityMagnitude"]) == 64
