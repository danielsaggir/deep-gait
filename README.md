# Deep Gait

POC project for gait recognition (ST-GCN and related experiments). The main entry point is the Jupyter notebook at the repository root. Dataset download, path resolution, and extraction live in a small Python package so the notebook stays short and imports work from any working directory.

---

## Repository layout

```text
Deep Gait/
├── DeepGait.ipynb          # Main notebook (model code + dataset cells at the top)
├── README.md               # This file
├── pyproject.toml          # Package metadata and dependencies (deep-gait, gdown)
├── requirements.txt        # Editable install: pip install -r requirements.txt → installs -e .
├── .gitignore              # Ignores venv, bytecode, large data, build metadata
│
├── deep_gait/              # Installable Python package (pip install -e .)
│   ├── __init__.py         # Re-exports public dataset helpers
│   └── dataset.py          # Colab/Drive, gdown download, paths, tar extract
│
└── data/                   # Local data (not committed; see .gitignore)
    ├── raw/                # CASIA-B_HRNet.tar (or other raw archives)
    └── processed/          # Extracted dataset (e.g. casia_b_hrnet/…)
```

After you run `pip install -e .`, setuptools also creates **`deep_gait.egg-info/`** next to `pyproject.toml`. That folder is build metadata; it can be regenerated and is listed in `.gitignore`.

---

## What each piece is for

| Path | Purpose |
|------|--------|
| [`DeepGait.ipynb`](DeepGait.ipynb) | Walkthrough: mount Drive (Colab), download CASIA archive (local), extract, then ST-GCN / PyTorch sections. |
| [`deep_gait/dataset.py`](deep_gait/dataset.py) | **Single source of truth** for archive filename, Google Drive defaults, `data/raw` vs `data/processed`, Colab mounted path, and `gdown` logic. |
| [`deep_gait/__init__.py`](deep_gait/__init__.py) | Lets you `from deep_gait import …` for the same functions as `deep_gait.dataset`. |
| [`pyproject.toml`](pyproject.toml) | Declares the `deep-gait` package and dependency on `gdown`. |
| [`requirements.txt`](requirements.txt) | `-e .` installs the project in editable mode and pulls dependencies from `pyproject.toml`. |
| `data/raw/` | Holds the **`.tar`** from Google Drive (large; gitignored). |
| `data/processed/casia_b_hrnet/` | **Extracted** tree (`tar.extractall` target; gitignored). The archive usually contains a top-level `CASIA-B_HRNet/` directory. |

---

## First-time setup (new developers)

1. **Python 3.10+** recommended (see `pyproject.toml`).

2. **Create and activate a virtual environment** (from the project root, the folder that contains `pyproject.toml`):

   ```bash
   cd "/path/to/Deep Gait"
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install the project in editable mode** (installs `gdown` and registers `deep_gait`):

   ```bash
   pip install -r requirements.txt
   # equivalent: pip install -e .
   ```

4. **Open the notebook** in Cursor/VS Code or Jupyter and select the **interpreter** from `.venv`. Then run the first dataset cells in order:
   - `mount_google_drive_if_colab()` (no-op locally)
   - `download_casia_if_local()` (skips if the archive already exists)
   - `extract_casia_archive()`

---

## Environment variables (optional)

| Variable | Meaning |
|----------|--------|
| `DEEP_GAIT_ROOT` | Absolute path to the project root if auto-detection fails (e.g. unusual kernel `cwd`). |
| `CASIA_B_HRNET_TAR` | Absolute path to the `.tar` file if it is not under `data/raw/`. |
| `GOOGLE_DRIVE_FILE_URL_OR_ID` | Drive share URL or file ID for `download_casia_if_local()` when you do not pass an argument in code. |

Project root is normally detected from the installed package location (`pip install -e .`), then from `DEEP_GAIT_ROOT`, `PWD`, the VS Code/Cursor notebook path, and walking parents for `deep_gait/dataset.py` or existing data markers.

---

## Google Colab

- Run **`pip install -e .`** from a clone/upload of this repo (or ensure `deep_gait` is on `PYTHONPATH`), then run the same three cells.
- The **mount** cell mounts Drive; **download** is a no-op on Colab (archive is read from the mounted Drive path); **extract** writes under `data/processed/casia_b_hrnet/` relative to the notebook’s current working directory (typically `/content`).

---

## Git and large files

- **`data/raw/*.tar`** and **`data/processed/`** are ignored so datasets are not committed.
- Add archives via Drive download or `gdown`, not via Git.

---

## Where to change dataset behavior

- Default Drive file ID, archive name, Colab tar path, and folder names: [`deep_gait/dataset.py`](deep_gait/dataset.py).
- Notebook calls only the public functions; avoid duplicating path logic in the notebook.
