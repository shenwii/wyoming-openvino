#!/usr/bin/env python3
"""
声学环境、房间混响 (RIR) 与负样本特征数据集下载器 (download_data.py)
======================================================================
参考 MicroWakeWordV2Trainer 与 openWakeWord 官方数据集规范，
自动下载并解压:
  1. MIT Environmental Impulse Responses (MIT RIR 房间冲激响应)
  2. 背景环境噪声数据集 (AudioSet / ESC-50 环境录音)
  3. openWakeWord 官方预计算负样本特征 (ACAV100M 2000 小时语音特征)
  4. 假阳性测试基准特征集 (validation_set_features.npy)
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_data")


def load_config(config_path: Path) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_with_resume(url: str, dest_path: Path) -> bool:
    """
    支持断点续传与进度展示的文件下载函数。
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    initial_size = 0
    if temp_path.exists():
        initial_size = temp_path.stat().st_size

    headers = {"User-Agent": "Wyoming-Wakeword-Trainer/1.0"}
    if initial_size > 0:
        headers["Range"] = f"bytes={initial_size}-"
        logger.info("发现未完成的下载，尝试断点续传: %s (%d bytes 已下载)", dest_path.name, initial_size)

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

            logger.info("开始下载 %s (总大小: %.2f MB)", dest_path.name, total_size / 1024 / 1024 if total_size else 0)

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
        logger.info("✅ 下载完成: %s", dest_path)
        return True

    except Exception as exc:
        logger.error("❌ 下载 %s 异常: %s", url, exc)
        return False


def extract_zip(archive_path: Path, extract_to: Path) -> bool:
    """解压 ZIP 压缩包至指定目录。"""
    logger.info("正在解压 %s -> %s ...", archive_path.name, extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
        logger.info("✅ 解压完成: %s", extract_to)
        return True
    except Exception as exc:
        logger.error("解压失败 %s: %s", archive_path, exc)
        return False


def main():
    parser = argparse.ArgumentParser(description="下载唤醒词模型训练所需的噪声、混响与负样本数据")
    parser.add_argument("--config", default="config.yaml", help="训练配置文件路径 (默认: config.yaml)")
    parser.add_argument("--all", action="store_true", help="下载全部数据集 (包含 2.5GB ACAV100M 特征)")
    parser.add_argument("--rirs", action="store_true", help="仅下载 MIT RIR 混响冲激响应数据集")
    parser.add_argument("--noise", action="store_true", help="仅下载背景噪声数据集")
    parser.add_argument("--negatives", action="store_true", help="仅下载官方负样本特征 (ACAV100M + Validation)")
    parser.add_argument("--quick-mode", action="store_true", help="快速轻量模式: 仅下载验证集与基础环境音")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        cfg_path = Path(__file__).resolve().parent / "config.yaml"
    cfg = load_config(cfg_path)

    data_dir = (Path(__file__).resolve().parent / cfg["data"]["data_dir"]).resolve()
    downloads_cfg = cfg["data"]["downloads"]

    download_all = args.all or (not args.rirs and not args.noise and not args.negatives and not args.quick_mode)

    print("======================================================================")
    print("📦 声学环境、RIR 混响与负样本数据下载管理")
    print(f"   - 目标数据目录: {data_dir}")
    print("======================================================================")

    # 1. 下载 MIT RIR 房间冲激响应数据集
    if download_all or args.rirs or args.quick_mode:
        rirs_dir = data_dir / "mit_rirs"
        rirs_zip = data_dir / "mit_rirs.zip"
        if not rirs_dir.exists() or len(list(rirs_dir.glob("**/*.wav"))) == 0:
            logger.info("准备下载 MIT RIR 房间冲激响应数据...")
            ok = download_with_resume(downloads_cfg["mit_rirs_url"], rirs_zip)
            if ok and rirs_zip.exists():
                extract_zip(rirs_zip, rirs_dir)
        else:
            logger.info("MIT RIR 冲激响应数据已存在，跳过: %s", rirs_dir)

    # 2. 下载背景噪声数据集
    if download_all or args.noise or args.quick_mode:
        noise_dir = data_dir / "noise"
        noise_zip = data_dir / "noise_dataset.zip"
        if not noise_dir.exists() or len(list(noise_dir.glob("**/*.wav"))) == 0:
            logger.info("准备下载背景环境噪声数据集...")
            ok = download_with_resume(downloads_cfg["noise_dataset_url"], noise_zip)
            if ok and noise_zip.exists():
                extract_zip(noise_zip, noise_dir)
        else:
            logger.info("背景噪声数据集已存在，跳过: %s", noise_dir)

    # 3. 下载 openWakeWord 负样本与验证集特征
    if download_all or args.negatives or args.quick_mode:
        neg_dir = data_dir / "negative_features"
        neg_dir.mkdir(parents=True, exist_ok=True)

        val_target = neg_dir / "validation_set_features.npy"
        if not val_target.exists():
            logger.info("正在下载假阳性验证集特征 (validation_set_features.npy)...")
            download_with_resume(downloads_cfg["validation_features_url"], val_target)
        else:
            logger.info("验证集特征已存在: %s", val_target)

        # ACAV100M 特征较庞大 (~2.5GB)，非 quick 模式下下载
        if not args.quick_mode or args.negatives or download_all:
            acav_target = neg_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
            if not acav_target.exists():
                logger.info("正在下载 2000 小时语音负样本特征 (openwakeword_features_ACAV100M_2000_hrs_16bit.npy, ~2.5GB)...")
                download_with_resume(downloads_cfg["acav100m_features_url"], acav_target)
            else:
                logger.info("ACAV100M 负样本特征已存在: %s", acav_target)

    print("\n✅ 数据集准备完成！")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
