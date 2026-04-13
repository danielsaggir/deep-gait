"""Deep Gait project package."""

from deep_gait.dataset import (
    download_casia_if_local,
    extract_casia_archive,
    is_colab,
    mount_google_drive_if_colab,
    processed_casia_dir,
    raw_archive_path,
    resolve_project_root,
)

__all__ = [
    "download_casia_if_local",
    "extract_casia_archive",
    "is_colab",
    "mount_google_drive_if_colab",
    "processed_casia_dir",
    "raw_archive_path",
    "resolve_project_root",
]
