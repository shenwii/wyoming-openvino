#!/usr/bin/env python3
"""
Third-party libraries and pre-trained feature extraction models setup script (setup_libs.py).
Downloads openWakeWord and MicroWakeWord components and models into the local lib/ directory.
"""

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

from i18n import t

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("setup_libs")

# Base feature model and repository URLs
DEFAULT_MELSPEC_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx"
DEFAULT_EMBEDDING_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"
OPENWAKEWORD_REPO_URL = "https://github.com/dscripka/openWakeWord.git"
MICROWAKEWORD_REPO_URL = "https://github.com/JohnnyPrimus/MicroWakeWordV2Trainer.git"


def download_file_with_progress(url: str, dest_path: Path) -> bool:
    """Download a file with progress reporting."""
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        logger.info(t("setup.already_exists", path=dest_path))
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    logger.info(t("setup.downloading", url=url, path=dest_path))

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Wyoming-Wakeword-Trainer/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp, open(temp_path, "wb") as out:
            total_size = int(resp.headers.get("content-length", 0))
            chunk_size = 1024 * 1024  # 1MB
            downloaded = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(
                        f"\r  {t('setup.download_progress', downloaded=downloaded / 1024 / 1024, total=total_size / 1024 / 1024, pct=pct)}",
                        end="",
                        flush=True,
                    )
            print()
        temp_path.rename(dest_path)
        logger.info(t("setup.download_complete", path=dest_path))
        return True
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(t("setup.download_failed", url=url, error=str(exc)))
        return False


def setup_feature_models(models_dir: Path) -> bool:
    """Download and verify openWakeWord official feature extraction ONNX models."""
    models_dir.mkdir(parents=True, exist_ok=True)
    melspec_target = models_dir / "melspectrogram.onnx"
    embedding_target = models_dir / "embedding_model.onnx"

    # Try to reuse existing models from parent workspace
    parent_models_dir = Path(__file__).resolve().parent.parent / "models"
    for target, name, url in [
        (melspec_target, "melspectrogram.onnx", DEFAULT_MELSPEC_URL),
        (embedding_target, "embedding_model.onnx", DEFAULT_EMBEDDING_URL),
    ]:
        candidate = parent_models_dir / name
        if candidate.exists() and not target.exists():
            try:
                target.symlink_to(candidate)
                logger.info(t("setup.symlink_created", target=target, candidate=candidate))
                continue
            except Exception:
                pass

        if not target.exists():
            ok = download_file_with_progress(url, target)
            if not ok:
                logger.warning(t("setup.manual_download", name=name, path=models_dir))
                return False

    return True


def clone_git_repo(repo_url: str, target_dir: Path) -> bool:
    """Clone a third-party git repository into local lib/ directory."""
    if target_dir.exists() and (target_dir / ".git").exists():
        logger.info(t("setup.already_exists", path=target_dir))
        return True

    logger.info(t("setup.downloading", url=repo_url, path=target_dir))
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = ["git", "clone", "--depth", "1", repo_url, str(target_dir)]
        subprocess.run(cmd, check=True)
        logger.info(t("setup.repo_cloned", url=repo_url, path=target_dir.name))
        return True
    except Exception as exc:
        logger.error(t("setup.repo_failed", url=repo_url))
        return False


def main():
    parser = argparse.ArgumentParser(description=t("cli.setup_help"))
    parser.add_argument("--models-only", action="store_true", help="Download only base feature models")
    parser.add_argument("--lib-dir", default="./lib", help="Library save directory [default: ./lib]")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    lib_dir = (base_dir / args.lib_dir).resolve()
    models_dir = lib_dir / "models"

    print("======================================================================")
    print("🚀 openWakeWord & MicroWakeWord Library & Model Setup")
    print("======================================================================")

    # 1. Prepare feature extraction models
    setup_feature_models(models_dir)

    if not args.models_only:
        # 2. Clone upstream repositories into lib/
        clone_git_repo(OPENWAKEWORD_REPO_URL, lib_dir / "openwakeword")
        clone_git_repo(MICROWAKEWORD_REPO_URL, lib_dir / "MicroWakeWordV2Trainer")

    print(f"\n{t('setup.download_complete', path=models_dir)}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
