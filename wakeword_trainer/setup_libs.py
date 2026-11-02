#!/usr/bin/env python3
"""
第三方开源库与预训练特征提取模型配置脚本 (setup_libs.py)
======================================================
将 openWakeWord 与 MicroWakeWordV2Trainer 官方开源组件/模型下载至本地 lib/ 目录，
避免重复造轮子，方便训练脚本直接调用与复用。
"""

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("setup_libs")

# 基础模型与仓库链接
DEFAULT_MELSPEC_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx"
DEFAULT_EMBEDDING_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"
OPENWAKEWORD_REPO_URL = "https://github.com/dscripka/openWakeWord.git"
MICROWAKEWORD_REPO_URL = "https://github.com/JohnnyPrimus/MicroWakeWordV2Trainer.git"


def download_file_with_progress(url: str, dest_path: Path) -> bool:
    """带进度指示的文件下载函数。"""
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        logger.info("文件已存在，跳过下载: %s", dest_path)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    logger.info("正在下载: %s -> %s", url, dest_path)

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
                    print(f"\r  下载进度: {downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB ({pct:.1f}%)", end="", flush=True)
            print()
        temp_path.rename(dest_path)
        logger.info("下载完成: %s", dest_path)
        return True
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        logger.error("下载失败 %s: %s", url, exc)
        return False


def setup_feature_models(models_dir: Path) -> bool:
    """下载并校验 openWakeWord 官方特征提取 ONNX 模型。"""
    models_dir.mkdir(parents=True, exist_ok=True)
    melspec_target = models_dir / "melspectrogram.onnx"
    embedding_target = models_dir / "embedding_model.onnx"

    # 1. 尝试从本地工作区现有路径复用
    parent_models_dir = Path(__file__).resolve().parent.parent / "models"
    for target, name, url in [
        (melspec_target, "melspectrogram.onnx", DEFAULT_MELSPEC_URL),
        (embedding_target, "embedding_model.onnx", DEFAULT_EMBEDDING_URL),
    ]:
        candidate = parent_models_dir / name
        if candidate.exists() and not target.exists():
            try:
                target.symlink_to(candidate)
                logger.info("已从主仓库建立符号链接: %s -> %s", target, candidate)
                continue
            except Exception:
                pass

        if not target.exists():
            ok = download_file_with_progress(url, target)
            if not ok:
                logger.warning("未能自动下载特征模型 %s，请手动下载或放置在 %s", name, models_dir)
                return False

    return True


def clone_git_repo(repo_url: str, target_dir: Path) -> bool:
    """克隆第三方开源仓库至本地 lib/ 目录。"""
    if target_dir.exists() and (target_dir / ".git").exists():
        logger.info("仓库已存在，跳过克隆: %s", target_dir)
        return True

    logger.info("正在克隆仓库: %s -> %s", repo_url, target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = ["git", "clone", "--depth", "1", repo_url, str(target_dir)]
        subprocess.run(cmd, check=True)
        logger.info("仓库克隆完成: %s", target_dir.name)
        return True
    except Exception as exc:
        logger.error("克隆仓库失败: %s (%s)", repo_url, exc)
        return False


def main():
    parser = argparse.ArgumentParser(description="下载配置 openWakeWord 与 MicroWakeWord 基础库与模型")
    parser.add_argument("--models-only", action="store_true", help="仅下载基础特征提取 ONNX 模型")
    parser.add_argument("--lib-dir", default="./lib", help="第三方依赖库保存目录 (默认: ./lib)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    lib_dir = (base_dir / args.lib_dir).resolve()
    models_dir = lib_dir / "models"

    print("======================================================================")
    print("🚀 openWakeWord & MicroWakeWord 第三方库与基础模型同步")
    print("======================================================================")

    # 1. 准备特征提取模型
    logger.info("检查并下载 openWakeWord 官方特征提取 ONNX 模型...")
    models_ok = setup_feature_models(models_dir)

    if not args.models_only:
        # 2. 克隆 openWakeWord 与 MicroWakeWordV2Trainer 仓库源码到 lib
        logger.info("正在同步 upstream 源码库至 %s...", lib_dir)
        clone_git_repo(OPENWAKEWORD_REPO_URL, lib_dir / "openwakeword")
        clone_git_repo(MICROWAKEWORD_REPO_URL, lib_dir / "MicroWakeWordV2Trainer")

    print("\n✅ 基础组件准备完毕！")
    print(f"   - 特征模型目录: {models_dir}")
    print(f"   - 第三方库目录: {lib_dir}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
