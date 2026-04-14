"""CLI: ``python -m deep_gait download`` | ``extract``."""

from __future__ import annotations

import argparse

from deep_gait.dataset import download_casia_if_local, extract_casia_archive


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m deep_gait",
        description="CASIA-B HRNet archive: download (gdown) or extract into data/processed/.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "download",
        help="Download CASIA-B_HRNet.tar when missing (needs gdown + Drive id or URL).",
    )
    sub.add_parser(
        "extract",
        help="Extract the .tar into data/processed/casia_b_hrnet (needs archive on disk).",
    )
    args = p.parse_args()
    if args.cmd == "download":
        download_casia_if_local()
    else:
        extract_casia_archive()


if __name__ == "__main__":
    main()
