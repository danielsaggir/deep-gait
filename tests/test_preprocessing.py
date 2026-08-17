"""Unit tests for production preprocessing and checkpoint loading."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from ml.constants import (  # noqa: E402
    DECISION_THRESHOLD,
    DEFAULT_CHECKPOINT,
    EMBEDDING_DIM,
    IN_CHANNELS,
    NUM_JOINTS,
    SEQUENCE_LENGTH,
)
from ml.model import SiameseGaitVerifier  # noqa: E402
from ml.pose import _iou, _select_person  # noqa: E402
from ml.preprocessing import load_raw_skeleton, preprocess_skeleton  # noqa: E402

REF_PREPROCESS = ROOT / "reference" / "dataset-2.py"


def _load_reference_preprocess():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ref_dataset", REF_PREPROCESS)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.preprocess_skeleton


def test_preprocess_output_shape():
    seq = np.random.uniform(20, 300, size=(80, 17, 2)).astype(np.float32)
    x = preprocess_skeleton(seq, is_training=False)
    assert x.shape == (IN_CHANNELS, SEQUENCE_LENGTH, NUM_JOINTS)


def test_short_sequence_zero_padded():
    seq = np.random.uniform(20, 300, size=(20, 17, 2)).astype(np.float32)
    x = preprocess_skeleton(seq, is_training=False)
    assert x.shape == (8, 64, 17)
    assert torch.all(x[:, 20:, :] == 0)


def test_long_sequence_center_crop():
    seq = np.zeros((100, 17, 2), dtype=np.float32)
    for t in range(100):
        seq[t, :, 0] = float(t)
        seq[t, 11, 0] = 0.0
        seq[t, 12, 0] = 0.0
    x = preprocess_skeleton(seq, is_training=False)
    start = (100 - 64) // 2
    assert x.shape[1] == 64
    assert float(x[0, 0, 0]) == pytest.approx(float(start))
    assert float(x[0, -1, 0]) == pytest.approx(float(start + 63))


def test_inference_preprocess_is_deterministic():
    seq = np.random.uniform(20, 300, size=(70, 17, 2)).astype(np.float32)
    a = preprocess_skeleton(seq, is_training=False)
    b = preprocess_skeleton(seq, is_training=False)
    assert torch.equal(a, b)


def test_joint_padding_to_17():
    seq = np.random.uniform(20, 300, size=(40, 12, 2)).astype(np.float32)
    x = preprocess_skeleton(seq, is_training=False)
    assert x.shape == (8, 64, 17)


def test_matches_reference_preprocess():
    ref = _load_reference_preprocess()
    rng = np.random.default_rng(0)
    seq = rng.uniform(20, 300, size=(90, 17, 2)).astype(np.float32)
    prod = preprocess_skeleton(seq, is_training=False)
    research = ref(seq, sequence_length=64, is_training=False)
    assert torch.allclose(prod, research, atol=1e-6)


def test_strict_checkpoint_load():
    assert DEFAULT_CHECKPOINT.is_file()
    model = SiameseGaitVerifier(num_nodes=17, in_channels=8, embedding_dim=128)
    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    assert missing == []
    assert unexpected == []


def test_default_eleven_channels_cannot_load():
    model = SiameseGaitVerifier(num_nodes=17, in_channels=11, embedding_dim=128)
    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    with pytest.raises(RuntimeError):
        model.load_state_dict(ckpt["model_state_dict"], strict=True)


def test_forward_shapes_and_probability_range():
    model = SiameseGaitVerifier()
    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    x1 = torch.randn(1, 8, 64, 17)
    x2 = torch.randn(1, 8, 64, 17)
    with torch.no_grad():
        logits, z1, z2 = model(x1, x2)
        prob = torch.sigmoid(logits)
    assert logits.shape == (1, 1)
    assert z1.shape == (1, EMBEDDING_DIM)
    assert z2.shape == (1, EMBEDDING_DIM)
    assert 0.0 <= float(prob) <= 1.0
    assert pytest.approx(1.0, abs=1e-5) == float(z1.norm(p=2))
    assert DECISION_THRESHOLD == 0.5


def test_casia_pickle_roundtrip_if_present():
    tree = ROOT / "data" / "processed" / "casia_b_hrnet"
    pkls = list(tree.rglob("*.pkl")) if tree.exists() else []
    if len(pkls) < 2:
        pytest.skip("CASIA-B pickles not present")
    s1 = load_raw_skeleton(pkls[0])
    s2 = load_raw_skeleton(pkls[1])
    assert s1.ndim == 3 and s1.shape[1] == 17 and s1.shape[2] == 2
    x1 = preprocess_skeleton(s1, is_training=False)
    x2 = preprocess_skeleton(s2, is_training=False)
    model = SiameseGaitVerifier()
    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        logits, z1, z2 = model(x1.unsqueeze(0), x2.unsqueeze(0))
        prob = float(torch.sigmoid(logits)[0, 0])
    assert 0.0 <= prob <= 1.0
    assert z1.shape[-1] == 128


def test_person_selection_prefers_iou_then_largest():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 90, 90]], dtype=np.float32)
    xy = np.zeros((2, 17, 2), dtype=np.float32)
    conf = np.ones((2, 17), dtype=np.float32)
    prev = np.array([48, 48, 88, 88], dtype=np.float32)
    chosen = _select_person(boxes, xy, conf, prev)
    assert chosen is not None
    np.testing.assert_array_equal(chosen[0], boxes[1])
    assert _iou(boxes[0], boxes[1]) == 0.0


def test_angles_written_at_vertex_joints():
    seq = np.zeros((8, 17, 2), dtype=np.float32)
    seq[:, 5] = [0, 0]
    seq[:, 7] = [1, 0]
    seq[:, 9] = [1, 1]
    x = preprocess_skeleton(seq, is_training=False)
    assert float(x[2, 0, 7]) > 0
    assert float(x[2, 0, 0]) == 0.0
