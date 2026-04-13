"""CASIA-B HRNet archive: paths, optional Colab Drive, gdown download, tar extract."""

from __future__ import annotations

import importlib.util
import inspect
import os
import tarfile
from pathlib import Path

CASIA_ARCHIVE_NAME = "CASIA-B_HRNet.tar"
DEFAULT_DRIVE_FILE_ID = "1uqwkAbR0X3G6ECf0OWRmOpp282CqEjzb"
COLAB_MOUNTED_TAR = (
    "/content/drive/MyDrive/Deep Project/Data/CASIA-B Pose/CASIA-B_HRNet.tar"
)
REL_RAW_ARCHIVE = Path("data") / "raw" / CASIA_ARCHIVE_NAME
REL_LEGACY_ARCHIVE = Path("data") / CASIA_ARCHIVE_NAME
PROCESSED_SUBDIR = Path("data") / "processed" / "casia_b_hrnet"


def is_colab() -> bool:
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


def mount_google_drive_if_colab() -> None:
    if not is_colab():
        return
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except ImportError:
        pass


def _project_root_from_package() -> Path | None:
    """If this file lives in a repo checkout, parent of ``deep_gait/`` is the project root."""
    here = Path(__file__).resolve()
    pkg_dir = here.parent
    candidate = pkg_dir.parent
    if (candidate / "pyproject.toml").is_file() and (pkg_dir / "dataset.py").is_file():
        return candidate
    return None


def _is_project_root(path: Path) -> bool:
    p = path.resolve()
    if (p / "deep_gait" / "dataset.py").is_file():
        return True
    if (p / REL_RAW_ARCHIVE).is_file():
        return True
    if (p / REL_LEGACY_ARCHIVE).is_file():
        return True
    return False


def _notebook_dir_from_stack() -> Path | None:
    for frame_info in inspect.stack():
        vsc = frame_info.frame.f_globals.get("__vsc_ipynb_file__")
        if vsc:
            return Path(vsc).resolve().parent
    return None


def resolve_project_root() -> Path:
    from_pkg = _project_root_from_package()
    if from_pkg is not None:
        return from_pkg

    roots: list[Path] = []
    env_root = os.environ.get("DEEP_GAIT_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser().resolve())
    pwd = os.environ.get("PWD")
    if pwd:
        roots.append(Path(pwd).resolve())

    nb = _notebook_dir_from_stack()
    if nb is not None:
        roots.append(nb)

    cwd = Path.cwd().resolve()
    roots.append(cwd)

    seen: set[Path] = set()
    for r in roots:
        if r in seen:
            continue
        seen.add(r)
        cur = r
        for _ in range(24):
            if _is_project_root(cur):
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
    return cwd


def raw_archive_path() -> Path:
    """Preferred path to the ``.tar`` (may not exist yet)."""
    env = os.environ.get("CASIA_B_HRNET_TAR")
    if env:
        return Path(env).expanduser().resolve()
    return (resolve_project_root() / REL_RAW_ARCHIVE).resolve()


def processed_casia_dir() -> Path:
    return (resolve_project_root() / PROCESSED_SUBDIR).resolve()


def download_casia_if_local(url_or_id: str | None = None) -> Path:
    """Download with gdown when not on Colab. Returns path to the archive file."""
    if is_colab():
        return Path(COLAB_MOUNTED_TAR)

    root = resolve_project_root()
    env_path = os.environ.get("CASIA_B_HRNET_TAR")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_file():
            print(f"Archive already present (CASIA_B_HRNET_TAR): {p}")
            return p

    dest = (root / REL_RAW_ARCHIVE).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    legacy = (root / REL_LEGACY_ARCHIVE).resolve()
    if legacy.is_file() and not dest.is_file():
        legacy.replace(dest)
        print(f"Moved legacy archive to: {dest}")

    key = (
        (url_or_id or "").strip()
        or os.environ.get("GOOGLE_DRIVE_FILE_URL_OR_ID", "").strip()
        or DEFAULT_DRIVE_FILE_ID
    )

    if dest.is_file():
        print(f"Archive already present: {dest}")
        return dest
    if not key:
        print(
            "Set GOOGLE_DRIVE_FILE_URL_OR_ID or pass url_or_id= to download_casia_if_local()."
        )
        return dest

    import gdown  # noqa: PLC0415

    if key.startswith("http"):
        gdown.download(key, str(dest), quiet=False, fuzzy=True)
    else:
        gdown.download(
            f"https://drive.google.com/uc?id={key}", str(dest), quiet=False
        )
    print(f"Downloaded to {dest}")
    return dest


def _archive_path_for_extract() -> Path:
    if is_colab():
        return Path(COLAB_MOUNTED_TAR)
    env = os.environ.get("CASIA_B_HRNET_TAR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    root = resolve_project_root()
    raw_p = (root / REL_RAW_ARCHIVE).resolve()
    if raw_p.is_file():
        return raw_p
    leg = (root / REL_LEGACY_ARCHIVE).resolve()
    if leg.is_file():
        return leg
    return raw_p


def extract_casia_archive() -> Path:
    """Extract archive into ``data/processed/casia_b_hrnet``. Returns that directory."""
    archive = _archive_path_for_extract()
    out = processed_casia_dir()
    out.mkdir(parents=True, exist_ok=True)

    if not archive.is_file():
        raise FileNotFoundError(
            f"Archive not found: {archive}\n"
            "Fix: run download_casia_if_local(), set CASIA_B_HRNET_TAR, "
            "set DEEP_GAIT_ROOT, or `pip install -e .` from the project root."
        )

    print(f"Archive: {archive}")
    print(f"Extract to: {out}")

    with tarfile.open(archive, "r") as tar:
        try:
            tar.extractall(path=out, filter="data")
        except TypeError:
            tar.extractall(path=out)

    print(f"החילוץ הסתיים! הקבצים נמצאים ב-{out}")
    return out
