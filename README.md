# Deep Gait

Gait recognition starter: **ST-GCN** (128-D embeddings), **MediaPipe/BlazePose** pose extraction, **CASIA-B** path helpers, a **Python inference CLI** for Node, and a **React + Express + MongoDB** web dashboard for uploads and similarity search.

**GitHub:** [github.com/danielsaggir/deep-gait](https://github.com/danielsaggir/deep-gait)

---

## Infrastructure split

| Where | Role |
|-------|------|
| **University Jupyter Server** | Heavy **training** (`python -m ml.train`, notebooks, CASIA data, GPU). Save checkpoints (e.g. `models/checkpoint.pth`) and copy or sync them to your machine. |
| **Local machine** | **Web app** (Node + React + MongoDB) and **inference** only (`python -m ml.inference`). |

---

## Repository layout

```text
Deep Gait/
├── config.yaml              # window_size, joints, embedding_dim, checkpoint path hints
├── package.json             # npm workspaces (webapp/server + webapp/client)
├── pyproject.toml           # deep_gait + ml packages; optional extras [inference], [train], [dev]
├── requirements.txt         # pip install -r → editable install + inference deps
├── data/
│   ├── raw/                 # CASIA .tar (gitignored); .gitkeep keeps the folder
│   └── processed/           # extracted data (gitignored)
├── notebooks/
│   └── DeepGait.ipynb       # Original Colab-style walkthrough
├── src/ml/                  # Production / research ML code
│   ├── model.py             # STGCN → 128-D
│   ├── processor.py         # GaitProcessor (video → tensor)
│   ├── dataset.py           # CasiaPoseDataset (.npy / .npz)
│   ├── inference.py         # CLI JSON signature (for Express bridge)
│   └── train.py             # Minimal training (Jupyter-oriented)
├── deep_gait/               # CASIA helpers only (`dataset.py` + `__init__.py`)
├── models/                  # Put checkpoints here (e.g. checkpoint.pth; gitignored)
├── tests/                   # pytest smoke tests
├── scripts/
│   └── run-all.sh           # Mongo (Docker) + API + Vite — see “Run everything (local)”
└── webapp/
    ├── server/              # Express API, uploads/, MongoDB, spawns Python inference
    └── client/              # Vite + React dashboard
```

---

## Python setup

1. **Python 3.10+** and a virtual environment:

   ```bash
   cd "/path/to/Deep Gait"
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install** (local inference + web bridge):

   ```bash
   pip install -r requirements.txt
   ```

   This installs `deep-gait` in editable mode with **`[inference]`** extras (PyTorch, OpenCV, MediaPipe).

   For **tests only**: `pip install -e ".[dev]"`.

   For **training on Jupyter** (same extras as inference): `pip install -e ".[train]"`.

3. **Checkpoint** — place a trained file at `models/checkpoint.pth` (or set `CHECKPOINT_PATH`). To generate a **dummy** checkpoint for wiring tests (not useful for accuracy):

   ```bash
   python -m ml.train --config config.yaml --save models/checkpoint.pth --epochs 1
   ```

4. **Inference CLI** (prints one JSON line: 128 floats):

   ```bash
   python -m ml.inference --video /path/to/video.mp4 --checkpoint models/checkpoint.pth --config config.yaml
   ```

---

## Web app (local)

**Prerequisites:** MongoDB running locally (or set `MONGODB_URI`), Python env as above, checkpoint at `models/checkpoint.pth`.

1. **Install JS dependencies** — from the repo root (installs server + client via npm workspaces):

   ```bash
   npm install
   ```

   Or install each app separately under `webapp/server/` and `webapp/client/` as before.

2. **Server** — from `webapp/server/`:

   ```bash
   cp .env.example .env
   # Edit .env: REPO_ROOT=absolute path to this repo, MONGODB_URI, CHECKPOINT_PATH if needed
   npm start
   ```

   Default API: `http://127.0.0.1:3001`. The server sets `PYTHONPATH` to `src` and runs `python -m ml.inference` on uploaded videos.

3. **Client** — from `webapp/client/`:

   ```bash
   npm run dev
   ```

   Open the printed URL (Vite proxies `/api` to port 3001).

### Run everything (local)

Use this when you want **one command** for the full stack: ensure MongoDB is up, then run the **Express API** and **Vite client** together (with labeled, merged logs). Training is **not** started; inference uses `CHECKPOINT_PATH` in `webapp/server/.env` as usual.

From the repo root (after `npm install` and a configured `webapp/server/.env`):

```bash
npm run run:all
```

What it does:

| Step | Behavior |
|------|----------|
| MongoDB | If nothing is listening on `127.0.0.1:27017`, the script starts (or reuses) a Docker container named `deepgait-mongo` (`mongo:7`). If you already run MongoDB yourself, it skips Docker. |
| Remote DB | If you use Atlas or another host, set `MONGODB_URI` in `.env` and run with `SKIP_MONGO_DOCKER=1 npm run run:all` so the script does not try to start Docker. |
| Processes | Runs `npm run dev:all`, which starts **server** and **client** in parallel; **Ctrl+C** stops both (`concurrently -k`). |

Equivalent manual steps: start Mongo, then `npm run start:server` in one terminal and `npm run dev:client` in another. To run only the two Node processes without the Mongo helper (Mongo already running): `npm run dev:all`.

---

## CASIA-B data (unchanged helpers)

Download/extract helpers live in [`deep_gait/dataset.py`](deep_gait/dataset.py). Typical flow:

- `mount_google_drive_if_colab()` (no-op locally)
- `download_casia_if_local()`
- `extract_casia_archive()` → `data/processed/casia_b_hrnet/`

`CasiaPoseDataset` scans that tree (or a custom `--data-root`) for `.npy` / `.npz` sequences shaped like `(2, T, 17)` or compatible layouts; see [`src/ml/dataset.py`](src/ml/dataset.py).

---

## Environment variables

| Variable | Meaning |
|----------|--------|
| `DEEP_GAIT_ROOT` | Project root if auto-detection fails. |
| `CASIA_B_HRNET_TAR` | Path to CASIA `.tar` if not under `data/raw/`. |
| `GOOGLE_DRIVE_FILE_URL_OR_ID` | For `download_casia_if_local()`. |
| `MONGODB_URI` | Mongo connection string (web server). |
| `REPO_ROOT` | Absolute path to repo root (web server; defaults to three levels above `webapp/server/src`). |
| `PYTHON_BIN` | Python executable for inference (default `python3`). |
| `CHECKPOINT_PATH` | Override path to `.pth` for inference. |

---

## Tests

```bash
pip install -e ".[dev,inference]"
pytest tests/ -q
```

---

## Git and large files

- `data/raw/*.tar`, `data/processed/casia_b_hrnet/` (extracted CASIA), `models/*.pth`, `webapp/server/uploads/` are ignored.
- `__pycache__/`, `.pytest_cache/`, build artifacts, and root `node_modules/` (npm workspaces) are ignored.
- Do not commit datasets or checkpoints; use Drive/`gdown` for archives.

---

## Where to change behavior

- **Model / window / embedding size:** [`config.yaml`](config.yaml) and [`src/ml/model.py`](src/ml/model.py).
- **Pose + normalization:** [`src/ml/processor.py`](src/ml/processor.py).
- **Dataset paths:** [`deep_gait/dataset.py`](deep_gait/dataset.py) and [`src/ml/dataset.py`](src/ml/dataset.py).
