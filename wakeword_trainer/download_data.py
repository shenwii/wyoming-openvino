#!/usr/bin/env python3
"""
Acoustic environment, room impulse responses (RIR), and negative feature dataset downloader (download_data.py).
Downloads and extracts:
  1. MIT Environmental Impulse Responses (MIT RIR)
  2. Ambient background noise dataset (ESC-50 / AudioSet)
  3. openWakeWord precomputed negative features (ACAV100M 2000 hours)
  4. False positive benchmark evaluation features (validation_set_features.npy)
"""

import argparse
import io
import logging
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile
import yaml

from i18n import t

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_data")


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_with_resume(url: str, dest_path: Path) -> bool:
    """Download a file with HTTP Range resume support and progress output."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    initial_size = 0
    if temp_path.exists():
        initial_size = temp_path.stat().st_size

    headers = {"User-Agent": "Wyoming-Wakeword-Trainer/1.0"}
    if initial_size > 0:
        headers["Range"] = f"bytes={initial_size}-"
        logger.info(t("download.resume_detected", name=dest_path.name, size=initial_size))

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.status
            content_range = resp.headers.get("Content-Range")
            content_len = resp.headers.get("Content-Length")

            if status == 206 and content_range:
                # 206 Partial Content
                total_size = int(content_range.split("/")[-1])
                mode = "ab"
            else:
                total_size = int(content_len) if content_len else 0
                mode = "wb"
                initial_size = 0

            logger.info(t("download.start_download", name=dest_path.name, size=total_size / 1024 / 1024 if total_size else 0))

            with open(temp_path, mode) as out_f:
                downloaded = initial_size
                chunk_size = 1024 * 1024  # 1MB
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(
                            f"\r  [{dest_path.name}] {downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB ({pct:.1f}%)",
                            end="",
                            flush=True,
                        )
                print()

        temp_path.rename(dest_path)
        logger.info(t("download.complete", path=dest_path))
        return True

    except Exception as exc:
        logger.error(t("download.download_error", url=url, error=str(exc)))
        return False


def extract_zip(archive_path: Path, extract_to: Path) -> bool:
    """Extract a ZIP archive into specified destination directory."""
    logger.info(t("download.extracting", name=archive_path.name, dest=extract_to))
    extract_to.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
        logger.info(t("download.extract_complete", dest=extract_to))
        return True
    except Exception as exc:
        logger.error(f"Failed to extract {archive_path}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description=t("cli.download_help"))
    parser.add_argument("--config", default="config.yaml", help="Path to config file [default: config.yaml]")
    parser.add_argument("--all", action="store_true", help="Download all datasets (including ~2.5GB ACAV100M)")
    parser.add_argument("--rirs", action="store_true", help="Download only MIT RIR impulse responses")
    parser.add_argument("--noise", action="store_true", help="Download only background noise datasets")
    parser.add_argument("--negatives", action="store_true", help="Download negative features")
    parser.add_argument("--quick-mode", action="store_true", help="Quick mode: download only validation and ambient")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        cfg_path = Path(__file__).resolve().parent / "config.yaml"
    cfg = load_config(cfg_path)

    data_dir = (Path(__file__).resolve().parent / cfg["data"]["data_dir"]).resolve()
    downloads_cfg = cfg["data"]["downloads"]

    download_all = args.all or (not args.rirs and not args.noise and not args.negatives and not args.quick_mode)

    print("======================================================================")
    print("📦 Acoustic Environment, RIR, & Negative Dataset Downloader")
    print(f"   - Target directory: {data_dir}")
    print("======================================================================")

    # 1. MIT RIR Dataset
    if download_all or args.rirs or args.quick_mode:
        rirs_dir = data_dir / "mit_rirs"
        rirs_zip = data_dir / "mit_rirs.zip"
        if not rirs_dir.exists() or len(list(rirs_dir.glob("**/*.wav"))) == 0:
            ok = download_with_resume(downloads_cfg["mit_rirs_url"], rirs_zip)
            if ok and rirs_zip.exists():
                extract_zip(rirs_zip, rirs_dir)
        else:
            logger.info(t("setup.already_exists", path=rirs_dir))

    # 2. Ambient Noise Dataset
    if download_all or args.noise or args.quick_mode:
        noise_dir = data_dir / "noise"
        noise_zip = data_dir / "noise_dataset.zip"
        if not noise_dir.exists() or len(list(noise_dir.glob("**/*.wav"))) == 0:
            ok = download_with_resume(downloads_cfg["noise_dataset_url"], noise_zip)
            if ok and noise_zip.exists():
                extract_zip(noise_zip, noise_dir)
        else:
            logger.info(t("setup.already_exists", path=noise_dir))

    # 3. openWakeWord Negative Features
    if download_all or args.negatives or args.quick_mode:
        neg_dir = data_dir / "negative_features"
        neg_dir.mkdir(parents=True, exist_ok=True)

        val_target = neg_dir / "validation_set_features.npy"
        if not val_target.exists():
            download_with_resume(downloads_cfg["validation_features_url"], val_target)
        else:
            logger.info(t("setup.already_exists", path=val_target))

        if not args.quick_mode or args.negatives or download_all:
            acav_target = neg_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
            if not acav_target.exists():
                download_with_resume(downloads_cfg["acav100m_features_url"], acav_target)
            else:
                logger.info(t("setup.already_exists", path=acav_target))

    print(f"\n{t('download.dataset_ready')}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
