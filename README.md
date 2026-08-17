# DeepGait

Cinematic gait-verification workstation. Two walking videos are converted into 17-joint COCO skeletons, preprocessed with the trained research pipeline, and compared by a Siamese ST-GCN. The classifier probability is the decision; cosine similarity is supporting evidence.

```text
Video A + Video B
  → YOLO11-pose (COCO-17, pixel coords)
  → isotropic scale to CASIA-B width 320, ~25 fps
  → preprocess_skeleton (pelvis-center, 8 channels, 64 frames)
  → SiameseGaitVerifier (reference/best_gait_verifier.pth)
  → sigmoid(logit) ≥ 0.5 → LIKELY MATCH
```

## Architecture

```text
React (TypeScript SPA)
  → Express (TypeScript)
    → Python process (`python -m ml.inference`)
      → pose extraction + authoritative preprocessing + PyTorch
```

There is no database, authentication, or job queue.

| Path | Role |
|------|------|
| `client/` | React workstation |
| `server/src/` | Express API |
| `server/ml/` | Production Python runtime |
| `reference/` | Research source of truth (do not modify for app convenience) |
| `server/ml/weights/best_gait_verifier.pth` | Trained Siamese checkpoint (epoch 16) |

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm

GPU (CUDA) is optional. CPU inference is supported.

## Python setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[inference]"
```

Pose extraction uses Ultralytics **YOLO11-pose** (`yolo11n-pose.pt`). Weights are stored at `server/ml/weights/yolo11n-pose.pt` after the first download (gitignored). If that file is missing, Ultralytics downloads it automatically.

Checkpoint (already in the repo):

```text
server/ml/weights/best_gait_verifier.pth
```

Override with `CHECKPOINT_PATH` if needed.

## Node setup

```bash
npm install
cp server/.env.example server/.env
# PYTHON_BIN should point at the venv, e.g. .venv/bin/python
```

Example `server/.env`:

```text
PORT=3001
PYTHON_BIN=/absolute/path/to/Deep Gait/.venv/bin/python
CHECKPOINT_PATH=server/ml/weights/best_gait_verifier.pth
MAX_UPLOAD_MB=80
```

## Development

From the repository root:

```bash
npm run dev
```

- API: `http://127.0.0.1:3001`
- UI: `http://127.0.0.1:5173` (proxies `/api`)

## Tests

```bash
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests -q
npm test
```

## Production build

```bash
npm run build
```

Typechecks both packages and builds the Vite client.

## API

`GET /api/health` — Python / checkpoint / device readiness.

`POST /api/analysis` — multipart fields `videoA`, `videoB`. Returns match probability, cosine similarity, pose frames, gait signatures, embeddings, and timings.

## Raw video → model

Training skeletons are CASIA-B HRNet pickles: `(T, 17, 3)` pixel coordinates on ~320×240 frames. Production therefore:

1. Decodes the video
2. Samples near 25 fps
3. Runs YOLO11-pose (native COCO-17)
4. Selects one person (IoU track, else largest box)
5. Keeps detected `x,y` even at modest confidence (training did the same)
6. Scales keypoints isotropically so image width maps to 320
7. Runs `preprocess_skeleton(..., is_training=False)`: pelvis-center, angles, torso-normalized bones, velocity, acceleration, center-crop or zero-pad to 64
8. Loads the checkpoint with `in_channels=8` and `strict=True`

If fewer than 16 frames contain a person, or coverage is below 25%, the API returns `INSUFFICIENT_GAIT_DATA` instead of scoring noise.

## Known limitations

- Detector family differs from training (HRNet → YOLO11-pose). Joint convention and coordinate scale are aligned; residual domain shift remains.
- The product compares two sequences. It does not search a gallery or claim identity certainty.
- Evaluation on the research test split was 86.87% cross-class and 83.33% within-class. Those figures are not per-video confidence.
- Multi-person footage uses a single tracked subject. Crowded scenes can fail or mix identities.
- CPU pose extraction of two HD videos can take tens of seconds.

## Compute device

Python uses CUDA when `torch.cuda.is_available()`, otherwise CPU. The UI health strip shows the device reported by `/api/health`.
