# DeepGait

**A cinematic, forensic-style gait-verification workstation.** DeepGait takes two walking videos, extracts 2D human pose (skeleton) sequences from each, and uses a Siamese Spatial-Temporal Graph Convolutional Network (ST-GCN) to estimate the probability that both clips show the same person walking — a biometric technique known as **gait recognition** or **gait verification**.

This repository contains the full product: a trained PyTorch model, a Python inference runtime, a Node.js/Express API, and a React single-page application that presents the analysis as a live, narrated "workstation" experience rather than a plain upload-and-result form.

```text
Video A + Video B
  → YOLO11-pose            (COCO-17 keypoints, pixel coordinates)
  → isotropic rescale      (CASIA-B reference width: 320 px, ~25 fps)
  → preprocess_skeleton()  (pelvis-centering, 8 channels/joint, 64 frames)
  → SiameseGaitVerifier    (twin ST-GCN encoders + fusion classifier)
  → sigmoid(logit) ≥ 0.5   → "LIKELY MATCH" / "LIKELY DIFFERENT"
```

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [The Machine Learning Pipeline](#the-machine-learning-pipeline)
5. [Model Architecture in Detail](#model-architecture-in-detail)
6. [Training & Evaluation](#training--evaluation)
7. [Repository Structure](#repository-structure)
8. [Tech Stack](#tech-stack)
9. [Getting Started](#getting-started)
10. [API Reference](#api-reference)
11. [Frontend Application](#frontend-application)
12. [Testing](#testing)
13. [Deploying to Render](#deploying-to-render)
14. [Known Limitations](#known-limitations)
15. [Future Work](#future-work)
16. [Academic Context](#academic-context)

---

## Overview

Gait — the way a person walks — is a **behavioral biometric**: it can be observed at a distance, without cooperation, and without the subject removing a mask or looking at a camera. DeepGait is a proof-of-concept system that demonstrates the full pipeline for gait-based person verification, from raw video to a calibrated probability score, wrapped in a polished, explainable product.

Rather than treating the model as a black box, the application is built around **transparency**: every stage of the pipeline (pose extraction, normalization, channel construction, graph encoding, embedding, fusion, and decision) is visualized in the UI, and an interactive 13-step "education debrief" lets a user step through exactly how the two verdicts were produced.

**What DeepGait is:**
- A two-video **verification** system: "are these the same person?" (1:1 comparison)
- A demonstration of a **Siamese ST-GCN** architecture trained on the CASIA-B gait dataset
- A full-stack reference implementation (ML → API → UI) suitable for a research/capstone project

**What DeepGait is not:**
- Not a **gallery search** / identification system (1:N) — it does not look up an identity in a database
- Not a production-grade forensic tool — there is no authentication, persistence, audit trail, or multi-subject tracking
- Not claiming legal-grade certainty — the output is a model probability, not proof of identity

---

## Key Features

### Analysis workflow
- **Dual video upload** with drag-and-drop or file picker for "Subject A" and "Subject B"
- **Client-side preview** of both clips with extracted metadata (duration, resolution, format, fps)
- **Real pose overlay** — COCO-17 skeleton joints and bones drawn over the video during and after analysis
- **Staged analysis console** that mirrors the real pipeline: *Decoding clips → Extracting pose → Normalising skeletons → Building channels → Encoding signatures → Fusing the pair → Scoring*
- **Verdict card** — `LIKELY MATCH` / `LIKELY DIFFERENT`, match probability, cosine similarity, and decision threshold
- **Synchronized replay** of both subjects after a result is produced

### Explainability & detail views
- **Readout tab** — full score breakdown, pose coverage per subject, and a timing bar (pose extraction / preprocessing / inference)
- **Motion tab** — per-frame velocity-based gait signatures plotted for both subjects (`GaitSignature`)
- **Fingerprint tab** — 128-dimensional embedding comparison (`EmbeddingCompare`) and per-channel feature composition (`FeatureComposition`)
- **13-step "Education Flow"** debrief (opened with `?`) that walks through the entire pipeline with custom SVG diagrams: frame sampling, joint detection, pelvis centering, motion traces, the 8-channel tensor, the skeleton-as-graph adjacency matrix, twin encoders, embedding fingerprints, cosine similarity, and the sigmoid decision curve

### Product polish
- **Cinematic boot sequence** on first load and an ambient animated HUD background
- **Procedural Web Audio** feedback (accept, analyze, success/failure, stage-advance tones) with a persisted mute toggle
- **Full keyboard control** — `Enter` to analyze, `R` to reset, `M` to mute, `?` to open the debrief, plus ARIA live regions and focus-trapped modals for accessibility
- **Live health strip** showing whether Python, PyTorch, the checkpoint, and the compute device (CPU/CUDA) are ready
- **Quality gating** — the API refuses to score a clip with too little detected motion (`INSUFFICIENT_GAIT_DATA`) instead of returning a misleading number

---

## System Architecture

DeepGait is a three-tier system: a browser SPA, a thin Node.js API, and a Python ML runtime invoked as a subprocess. There is **no database, authentication, or job queue** — every request is stateless and self-contained.

```text
┌─────────────────────────────┐
│   React SPA (TypeScript)    │  Vite dev server :5173
│   Workstation UI, charts,   │  fetch() + multipart/form-data
│   HUD, audio, education     │
└──────────────┬──────────────┘
               │  GET /api/health
               │  POST /api/analysis  (videoA, videoB)
               ▼
┌─────────────────────────────┐
│  Express API (TypeScript)   │  Node.js :3001
│  routes → controllers →     │  Multer disk upload, temp-dir
│  mlService (spawns Python)  │  cleanup, structured error codes
└──────────────┬──────────────┘
               │  spawn("python", ["-m", "ml.inference",
               │         "--video-a", ..., "--video-b", ...,
               │         "--checkpoint", ...])
               │  JSON on stdout, logs on stderr
               ▼
┌─────────────────────────────┐
│   Python ML runtime         │  server/ml/
│   pose.py   → YOLO11-pose   │
│   preprocessing.py → tensor │
│   model.py  → ST-GCN        │
│   inference.py → orchestr.  │
└─────────────────────────────┘
```

**Why a subprocess instead of a long-lived Python service?** Simplicity and process isolation for a research-grade demo: each request gets a fresh, deterministic run with no shared mutable state, at the cost of paying model-load latency per request. The `PYTHONPATH` is set to `server/`, so the `ml` package resolves identically whether invoked from the CLI (`python -m ml.inference`) or by Node.

**Request lifecycle:**
1. Client submits `multipart/form-data` with fields `videoA` and `videoB` to `POST /api/analysis`.
2. `attachTempDir` middleware creates a per-request temp directory; `uploadPair` (Multer) validates MIME/extension, enforces the size limit, and streams both files to disk.
3. `analysisController.analyze` calls `mlService.runAnalysis`, which spawns the Python CLI with the two file paths and the resolved checkpoint path, and awaits a single line of JSON on stdout (with a configurable timeout, default 180 s).
4. On success, the full JSON payload (scores, both subjects' pose/embedding/gait data, model metadata, timings) is returned as-is to the client.
5. On failure, Python emits a structured `{ "error": { "code", "message", "subject?" } }` object, which is translated into an HTTP status (e.g. `422` for `INSUFFICIENT_GAIT_DATA`, `503` for a missing checkpoint or Python runtime, `504` for a timeout).
6. The temp directory is always removed in a `finally` block, regardless of outcome.

---

## The Machine Learning Pipeline

The production pipeline in `server/ml/` reproduces, at inference time, the exact preprocessing contract that the model was trained under (`reference/dataset-2.py`), while replacing the training-time pose source (offline HRNet keypoints from the CASIA-B dataset) with a live, general-purpose pose detector (YOLO11-pose) so that the model can run on arbitrary user-supplied video.

### 1. Pose extraction (`server/ml/pose.py`)
- Loads the video with OpenCV and samples frames at a target rate of **~25 fps** (matching CASIA-B), regardless of the source video's native frame rate.
- Runs **Ultralytics YOLO11-pose** (`yolo11n-pose.pt`) per sampled frame, which natively predicts the same **17-keypoint COCO** joint layout used by the training data.
- **Single-subject tracking**: when multiple people are detected, the box with the highest IoU against the previous frame's box is kept (a lightweight tracker); if no previous box exists or IoU tracking fails, the largest bounding box is chosen.
- Keypoints are kept even at modest confidence, mirroring how the original HRNet-labeled training data behaves.
- All `x, y` pixel coordinates are **isotropically scaled** so that the frame width maps to CASIA-B's reference width of 320 px — aligning coordinate scale between the live detector and the training distribution.
- **Quality gate**: if fewer than 16 frames contain a detected person, or detection coverage is below 25% of sampled frames, extraction raises `InsufficientGaitDataError` rather than feeding noise into the model.

### 2. Preprocessing (`server/ml/preprocessing.py`)
`preprocess_skeleton()` reproduces the training-time feature engineering exactly (`is_training=False` — no flips, scale jitter, or noise augmentation at inference):

| Channel(s) | Feature | Description |
|---|---|---|
| 0–1 | **Position** | (x, y) of each of the 17 joints, re-centered so the pelvis midpoint (mean of joints 11 & 12) is the origin |
| 2 | **Joint angles** | Interior angles at six joints (elbows, knees, hips) computed from vector cosine similarity between adjacent bones |
| 3 | **Bone proportions** | Length of eight limb segments, normalized by torso length (shoulder-midpoint to hip-midpoint distance) |
| 4–5 | **Velocity** | Frame-to-frame difference of position channels |
| 6–7 | **Acceleration** | Frame-to-frame difference of the velocity channels |

The result is an `8 × T × 17` tensor. Sequences longer than the model's fixed window are **center-cropped** to 64 frames (at inference); shorter sequences are **zero-padded**. During training, the equivalent step instead uses a random crop position plus horizontal-flip, scale-jitter, and Gaussian-noise augmentation.

### 3. Inference (`server/ml/inference.py`)
Both subjects' tensors are batched through the same `SiameseGaitVerifier`, producing two independent 128-D embeddings, a fused **match probability** (sigmoid of the classifier logit), and a **cosine similarity** between embeddings (reported as supporting evidence, not the decision variable). The CLI also derives UI-facing artifacts for each subject — a velocity-based "gait signature," a per-channel feature-composition summary, and the raw pose overlay frames — so the frontend never has to recompute anything the model already produced.

---

## Model Architecture in Detail

**`SiameseGaitVerifier`** (`server/ml/model.py`, architecture identical to `reference/model.py`) is composed of two weight-shared **ST-GCN** encoders and a small fusion classifier.

### Graph construction
The 17 COCO joints are treated as nodes of a graph. A fixed **physical adjacency matrix** connects anatomically adjacent joints (e.g. shoulder–elbow, hip–knee, shoulder–hip) plus self-loops, then row-normalizes it. This adjacency is registered as a **learnable parameter**, so the network can adapt edge weights during training beyond the fixed skeletal prior.

### `DeepGait_STGCN` encoder
Three stacked **ST-GCN blocks**, each combining:
- **Spatial graph convolution** — a 1×1 `Conv2d` across channels, followed by graph propagation (`einsum` against the learnable adjacency) to mix information between anatomically connected joints
- **Temporal convolution** — a `Conv1d` (kernel size 9) applied independently per joint across the time axis, with batch norm, ReLU, and dropout (0.3)
- **Residual connection** — a 1×1 projection when channel dimensions change, else identity

Channel progression: **8 → 64 → 128 → 256**, followed by global adaptive average pooling over (time, joints) and a final linear layer to a **128-D embedding**, which is L2-normalized.

### Siamese fusion & classification
Both subjects are passed through the *same* encoder (`forward_once`) to produce embeddings `z1`, `z2`. These are combined into a fusion vector:

```text
fused = concat([ |z1 - z2|,   z1 * z2,   cosine_similarity(z1, z2) ])
        (128 dims)  (128 dims)  (1 dim)             = 257 dims
```

The fusion vector passes through a small MLP classifier (`Linear(257→256) → ReLU → Dropout(0.2) → Linear(256→1)`) producing a single logit. `sigmoid(logit)` is the reported `samePersonProbability`; a threshold of **0.5** separates `LIKELY_MATCH` from `LIKELY_DIFFERENT`.

| Hyperparameter | Value |
|---|---|
| Input channels | 8 |
| Joints (nodes) | 17 (COCO) |
| Sequence length | 64 frames |
| Embedding dimension | 128 |
| Classifier hidden dimension | 256 |
| Temporal kernel size | 9 |
| Dropout (ST-GCN blocks / classifier) | 0.3 / 0.2 |
| Decision threshold | 0.5 |

> **Note:** `reference/model.py` defaults to `in_channels=11` (an earlier research configuration). The shipped checkpoint (`best_gait_verifier.pth`) was trained with **8 channels**, so `server/ml/constants.py` and `server/ml/model.py` pin `IN_CHANNELS = 8` to match the weights exactly — loading is done with `strict=True` to guarantee this.

---

## Training & Evaluation

- **Dataset**: [CASIA-B](http://www.cbsr.ia.ac.cn/china/Gait%20Databases%20CH.asp), a well-known gait recognition benchmark, with 2D skeleton keypoints pre-extracted via **HRNet** and stored as pickled `(T, 17, 3)` arrays (x, y, confidence) at CASIA-B's native ~320×240 resolution.
- **Task formulation**: pairs of sequences are sampled and labeled as *same subject* or *different subject*; the network is trained with a binary cross-entropy loss on the fused classifier logit (see `reference/train.ipynb`, `reference/dataset-2.py`).
- **Data augmentation** (training only): random horizontal flip, random scale jitter (0.9×–1.1×), Gaussian coordinate noise, and random temporal crop position.
- **Reported test-set performance** (`reference/Test.ipynb`):

  | Split | Accuracy |
  |---|---|
  | Cross-class (unseen identities) | **86.87%** |
  | Within-class (seen identities, held-out sequences) | **83.33%** |

  These figures describe aggregate performance on the held-out CASIA-B test split — they are **not** a per-video confidence guarantee for arbitrary real-world footage (see [Known Limitations](#known-limitations)).
- Training curves (`reference/training vs val loss.png`, `reference/train vs val accuracy.png`) and the full training/evaluation notebooks are preserved in `reference/` as the research source of truth and are **not** modified for application convenience — the production `server/ml/` code adapts only what is necessary (default `in_channels`, live pose extraction) while keeping the preprocessing math bit-for-bit identical.

---

## Repository Structure

```text
Deep Gait/
├── client/                      React + TypeScript SPA (Vite)
│   └── src/
│       ├── App.tsx                     Renders <Workstation />
│       ├── audio/engine.ts             Procedural Web Audio UI sounds
│       ├── hooks/workstationReducer.ts Central state machine (phases, actions)
│       ├── services/api.ts             fetchHealth(), runAnalysis()
│       ├── types/analysis.ts           Shared client-side types, STAGES
│       ├── utils/                      skeleton.ts, series.ts, theme.ts
│       ├── styles/                     global.css, cinematic.css, ambient.css
│       ├── components/
│       │   ├── workstation/            Workstation.tsx, Header.tsx (shell + shortcuts)
│       │   ├── video/                  SubjectPanel, SkeletonOverlay
│       │   ├── analysis/               Console, StageRail, AnalysisDetail, FitToHeight
│       │   ├── charts/                 GaitSignature, EmbeddingCompare, FeatureComposition
│       │   ├── cinematic/              BootSequence, EducationFlow, EduDiagrams
│       │   ├── hud/                    AmbientField, AmbientCanvas, ReactorCore, HudFrame, TickRing
│       │   └── brand/                  GaitMark
│       └── test/                       Vitest + Testing Library specs
│
├── server/                      Express + TypeScript API
│   ├── src/
│   │   ├── index.ts                    App entry (cors, json, routes, error handler)
│   │   ├── config/env.ts               .env loading, PORT/PYTHON_BIN/checkpoint resolution
│   │   ├── routes/index.ts             /api/health, /api/analysis
│   │   ├── controllers/                healthController, analysisController
│   │   ├── middleware/                 upload.ts (Multer), errors.ts
│   │   ├── services/                   mlService.ts (Python bridge), tempFiles.ts
│   │   └── types/analysis.ts           Shared server-side response/error types
│   ├── tests/api.test.ts               Supertest integration tests (mocked ML layer)
│   └── ml/                      Production Python ML runtime
│       ├── inference.py                CLI entry: python -m ml.inference
│       ├── model.py                    SiameseGaitVerifier, ST-GCN blocks
│       ├── pose.py                     YOLO11-pose extraction + person tracking
│       ├── preprocessing.py            8-channel skeleton preprocessing
│       ├── constants.py                Model/pipeline constants (checkpoint contract)
│       ├── errors.py                   Domain error types (→ API error codes)
│       └── weights/
│           ├── best_gait_verifier.pth  Trained Siamese checkpoint (epoch 16)
│           └── yolo11n-pose.pt         YOLO11-pose weights (auto-downloaded if absent)
│
├── reference/                   Research source of truth — not modified for the app
│   ├── model.py                        Original ST-GCN architecture (in_channels=11 default)
│   ├── dataset-2.py                    CASIA-B dataset loader + preprocess_skeleton (training)
│   ├── train.ipynb                     Model training notebook
│   ├── Test.ipynb                      Evaluation notebook (accuracy figures)
│   ├── best_gait_verifier.pth          Original trained checkpoint
│   └── *.png                           Training/validation curves
│
├── tests/                       Python pytest suite (preprocessing, pose, CLI)
├── scripts/                     dev.sh (concurrent client+server), free-api-port.sh
├── artifacts/e2e/                Playwright E2E scripts + captured screenshots (gitignored)
├── pyproject.toml               Python package definition (ml → server/ml)
├── requirements.txt             pip install -e ".[inference]"
└── package.json                 npm workspaces root (client, server)
```

---

## Tech Stack

### Frontend (`client/`)
| Category | Technology |
|---|---|
| Framework | React 18 (function components, hooks, no router — single view) |
| Language | TypeScript 5.7 |
| Build tool | Vite 5 (`@vitejs/plugin-react`) |
| State | Plain `useReducer` (`workstationReducer.ts`) — no external state library |
| Testing | Vitest, React Testing Library, jsdom |
| Styling | Hand-written CSS (`global.css`, `cinematic.css`, `ambient.css`) — no CSS framework |
| Audio | Native Web Audio API (synthesized tones, no audio files) |

### Backend (`server/`)
| Category | Technology |
|---|---|
| Runtime | Node.js, Express 4 |
| Language | TypeScript 5.7, run via `tsx` |
| File uploads | Multer (disk storage, per-request temp directories) |
| Config | `dotenv` |
| Testing | Vitest, Supertest |

### Machine learning (`server/ml/`, `reference/`)
| Category | Technology |
|---|---|
| Framework | PyTorch ≥ 2.0 |
| Pose estimation | Ultralytics YOLO11-pose (`yolo11n-pose.pt`) |
| Video I/O | OpenCV (`opencv-python-headless`) |
| Numerics | NumPy |
| Testing | pytest |
| Packaging | `pyproject.toml` (setuptools), editable install, Python ≥ 3.10 |

### Tooling
- **npm workspaces** (`client`, `server`) orchestrated from the repository root
- **Playwright** for end-to-end browser testing (`artifacts/e2e/`)
- No ESLint/Prettier configured — linting is enforced via `tsc --noEmit` on both packages
- A root `Dockerfile` (Node + Python in one image) and `render.yaml` blueprint support deploying to Render; see [Deploying to Render](#deploying-to-render)

---

## Getting Started

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** and npm
- GPU (CUDA) is optional — CPU inference is fully supported, just slower

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[inference]"
```

This installs PyTorch, OpenCV, and Ultralytics, and registers the `ml` package (mapped to `server/ml/` via `pyproject.toml`).

Pose extraction uses **YOLO11-pose**. If `server/ml/weights/yolo11n-pose.pt` is absent, Ultralytics downloads it automatically on first run. The trained gait-verification checkpoint (`server/ml/weights/best_gait_verifier.pth`) already ships in the repository; override its location with the `CHECKPOINT_PATH` environment variable if needed.

### 2. Node environment

```bash
npm install
cp server/.env.example server/.env
```

Edit `server/.env` so `PYTHON_BIN` points at your virtual environment's interpreter:

```dotenv
PORT=3001
PYTHON_BIN=/absolute/path/to/Deep Gait/.venv/bin/python
CHECKPOINT_PATH=server/ml/weights/best_gait_verifier.pth
MAX_UPLOAD_MB=80
```

### 3. Run in development

```bash
npm run dev
```

This runs `scripts/dev.sh`, launching the Express API and the Vite dev server concurrently:
- API → `http://127.0.0.1:3001`
- UI → `http://127.0.0.1:5173` (proxies `/api/*` to the API)

### 4. Production build

```bash
npm run build
```

Typechecks both workspaces (`tsc --noEmit`) and builds the optimized Vite client bundle.

---

## API Reference

### `GET /api/health`
Reports readiness of the Python/PyTorch/model stack without running an analysis.

**Response**
```json
{
  "status": "online",
  "pythonAvailable": true,
  "torchAvailable": true,
  "modelAvailable": true,
  "device": "cpu"
}
```
`status` is `"online"` only when Python is reachable **and** the checkpoint file exists on disk; otherwise `"degraded"`.

### `POST /api/analysis`
Runs a full pairwise gait comparison.

**Request** — `multipart/form-data`
| Field | Type | Notes |
|---|---|---|
| `videoA` | file | Subject A footage |
| `videoB` | file | Subject B footage |

Accepted extensions: `.mp4`, `.webm`, `.mov`, `.avi`, `.mkv`. Max size: `MAX_UPLOAD_MB` per file (default 80 MB). Default request timeout: `ANALYSIS_TIMEOUT_MS` (default 180,000 ms).

**Response** (`200`, abbreviated)
```json
{
  "result": {
    "samePersonProbability": 0.9123,
    "cosineSimilarity": 0.87,
    "threshold": 0.5,
    "verdict": "LIKELY_MATCH"
  },
  "subjectA": {
    "metadata": { "source": "clip-a.mp4", "duration": 4.2, "width": 1920, "height": 1080, "fps": 29.97, "format": "MP4" },
    "poseQuality": { "framesDetected": 98, "framesUsed": 64, "framesSampled": 105, "coverage": 0.933 },
    "poseFrames": [ { "timestamp": 0.0, "detected": true, "joints": [ { "x": 0.51, "y": 0.22, "confidence": 0.94 }, "..." ] } ],
    "gaitSignature": { "velocityMagnitude": ["..."], "lowerBodyMotion": ["..."] },
    "featureComposition": { "position": 0.14, "angles": 0.32, "proportions": 0.61, "velocity": 0.02, "acceleration": 0.01 },
    "embedding": ["...128 floats..."],
    "skeletonEdges": [[0, 1], [0, 2], "..."]
  },
  "subjectB": { "...": "same shape as subjectA" },
  "model": {
    "architecture": "SiameseGaitVerifier",
    "embeddingDimension": 128,
    "inputChannels": 8,
    "sequenceLength": 64,
    "joints": 17,
    "device": "cpu"
  },
  "timing": { "poseExtraction": 4.21, "preprocessing": 0.03, "inference": 0.11, "total": 4.35 }
}
```

**Error codes** (`{ "error": { "code", "message", "subject?" } }`)
| Code | HTTP status | Meaning |
|---|---|---|
| `MISSING_VIDEO` | 400 | One or both video fields were not provided |
| `UNSUPPORTED_VIDEO` | 400 | File extension/MIME type not accepted |
| `FILE_TOO_LARGE` | 413 | File exceeds `MAX_UPLOAD_MB` |
| `VIDEO_DECODE_ERROR` | 500 | OpenCV could not read the video |
| `INSUFFICIENT_GAIT_DATA` | 422 | Fewer than 16 detected-person frames, or <25% detection coverage |
| `CHECKPOINT_ERROR` | 503 | Model checkpoint missing or incompatible with the architecture |
| `PYTHON_UNAVAILABLE` | 503 | The Python process could not be spawned |
| `INFERENCE_TIMEOUT` | 504 | Analysis exceeded `ANALYSIS_TIMEOUT_MS` |
| `INFERENCE_FAILURE` | 500 | Unclassified Python-side failure (stderr tail is attached to the message) |

The Python CLI (`server/ml/inference.py`) can also be run standalone for debugging or batch evaluation:

```bash
python -m ml.inference --video-a clip1.mp4 --video-b clip2.mp4 --checkpoint server/ml/weights/best_gait_verifier.pth
# or, against pre-extracted CASIA-B-style skeleton pickles:
python -m ml.inference --skeleton-a a.pkl --skeleton-b b.pkl
```

---

## Frontend Application

The entire UI is a single component tree rooted at `Workstation`, driven by one reducer (`workstationReducer.ts`) with an explicit state machine:

```text
READY → PARTIAL_UPLOAD → READY_TO_ANALYZE → ANALYZING → RESULT
                                                  └──────→ ERROR
```

| State field | Purpose |
|---|---|
| `phase` | Current step in the state machine above |
| `subjectA` / `subjectB` | `{ file, objectUrl, metadata }` for each uploaded clip |
| `analysis` | The full `POST /api/analysis` response once available |
| `stageIndex` | Index into the 7-step analysis console (`STAGES`) for the choreographed progress animation |
| `muted` | Persisted (sessionStorage) audio preference |
| `overlayEnabled` | Whether the skeleton overlay is drawn on the video |

**Component layers:**
- **Workstation shell** (`workstation/`) — top-level layout, header, global keyboard shortcuts
- **Video** (`video/`) — upload panels and canvas-based skeleton overlay rendering
- **Analysis** (`analysis/`) — the live console, staged progress rail, and the tabbed detail view (Readout / Motion / Fingerprint)
- **Charts** (`charts/`) — `GaitSignature` (velocity over time), `EmbeddingCompare` (128-D vector diff), `FeatureComposition` (per-channel magnitude)
- **Cinematic** (`cinematic/`) — `BootSequence` (app entry animation) and `EducationFlow` + `EduDiagrams` (the 13-step, SVG-illustrated explainer)
- **HUD** (`hud/`) — ambient particle field, ticking ring, and a "reactor core" motif shown while analysis is running
- **Brand** (`brand/`) — the `GaitMark` logo mark

All communication with the backend goes through `services/api.ts` (`fetchHealth()`, `runAnalysis()`), which wraps `fetch()`; there is no other client-server transport (no WebSockets, no polling beyond the initial health check).

---

## Testing

**Python** (pytest — preprocessing determinism/shape, pose extraction & person-selection logic, insufficient-data handling, end-to-end CLI inference against fixture skeleton pickles):
```bash
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests -q
```

**TypeScript / React** (Vitest — reducer transitions, component rendering for `Console`, `AnalysisDetail`, `EducationFlow`, keyboard shortcut behavior, and a Supertest-based API integration test with the ML layer mocked):
```bash
npm test
```

**End-to-end** (Playwright scripts under `artifacts/e2e/` drive the running app with real sample clips and capture screenshots/perf traces — used for manual QA rather than CI).

---

## Deploying to Render

DeepGait is one logical application that needs **both** Node (Express API + built React SPA) and Python (PyTorch + YOLO11-pose) at runtime. Render's native "Node" and "Python" environments each ship only one language, so the project deploys as a single **Docker** web service instead of either native runtime.

1. Push the repo to a Git remote Render can access.
2. In Render, choose **New → Blueprint** and select this repo — it will pick up `render.yaml` and the root `Dockerfile` automatically. (Alternatively, create a Web Service manually with environment **Docker** and Dockerfile path `./Dockerfile`.)
3. Choose a plan with enough CPU/RAM. CPU inference with PyTorch + YOLO11-pose on multi-second HD clips needs more headroom than Render's free/starter tier.
4. Render injects `PORT` automatically and the server already reads `process.env.PORT`; `PYTHON_BIN` is baked into the image, so no extra env vars are required beyond what's in `render.yaml` (`CHECKPOINT_PATH`, `MAX_UPLOAD_MB`, `ANALYSIS_TIMEOUT_MS`).
5. The image build installs Node deps, creates a Python venv (`pip install -e ".[inference]"`), builds the Vite client, and pre-downloads the YOLO11-pose weights so cold starts don't hit the network.
6. At runtime, `server/src/index.ts` serves the built `client/dist` SPA alongside `/api/*`, so the whole product is reachable on the single Render URL — no separate static site is needed.

---

## Known Limitations

- **Detector domain shift**: the model was trained on offline HRNet keypoints; production uses YOLO11-pose. Joint convention (COCO-17) and coordinate scale are explicitly aligned, but residual distribution shift between detectors is not eliminated.
- **Verification, not identification**: the product answers "do these two clips match?" — it does not search a gallery of known identities or assign a name.
- **Aggregate accuracy ≠ per-video confidence**: the reported 86.87% / 83.33% figures come from the CASIA-B held-out test split; a single real-world video pair's actual reliability depends heavily on footage quality, viewing angle, occlusion, and clothing.
- **Single-subject assumption**: multi-person scenes are reduced to one tracked subject via IoU/box-area heuristics; crowded or occluded footage can fail the quality gate or silently track the wrong person.
- **CPU latency**: pose extraction on two HD clips can take tens of seconds on CPU; a CUDA-capable GPU is recommended for interactive use but not required.
- **No persistence or audit trail**: uploaded videos live only in a per-request temp directory and are deleted immediately after the response is sent.

---

## Future Work

- Swap the live detector for an HRNet-based inference path to close the training/production pose-source gap entirely
- Support 1:N "gallery" search over a set of enrolled gait signatures instead of pairwise-only comparison
- Multi-person-aware tracking (e.g. persistent IDs) for cluttered scenes
- GPU-backed deployment (containerization) for lower end-to-end latency
- Confidence calibration (e.g. temperature scaling) so `samePersonProbability` better reflects true per-video reliability
- Batch/offline evaluation mode against additional public gait datasets beyond CASIA-B

---

## Academic Context

DeepGait began as a computer-vision/biometrics capstone project centered on training a Siamese Spatial-Temporal Graph Convolutional Network for gait verification on the CASIA-B dataset (see `reference/train.ipynb`, `reference/Test.ipynb`, and `reference/Project_Book.pdf`). This repository extends that research artifact into a complete, demonstrable application: a production-oriented inference runtime that adapts the trained model to arbitrary user-supplied video (rather than only pre-extracted benchmark skeletons), served through a documented REST API and presented through an explanatory, cinematic user interface designed to make a non-trivial ML pipeline legible to a non-technical audience.
