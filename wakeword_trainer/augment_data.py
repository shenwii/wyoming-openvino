#!/usr/bin/env python3
"""
唤醒词声学环境仿真与数据增强合成器 (augment_data.py)
======================================================
参考 MicroWakeWord 与 openWakeWord 官方声学混音增强管线:
  1. RIR (Room Impulse Response) 真实房间混响模拟 (MIT RIR 数据集)
  2. 背景环境噪声混音 (以 5~25dB 随机 SNR 混叠生活环境噪声)
  3. 随机时间平移与静音边界扰动 (模拟语音在滑动窗口中的非定常位置)
  4. 动态范围与增益扰动 (Volume Jittering)
将干净的 TTS 语音转换为具有高鲁棒性的真实环境合成语音样本。
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import wave
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal
import soundfile as sf
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("augment_data")

TARGET_SAMPLE_RATE = 16000


def load_wav_16k(wav_path: Path) -> np.ndarray:
    """读取 WAV 文件并确保为 16kHz 单声道 float32 (-1.0 ~ 1.0)。"""
    try:
        data, sr = sf.read(str(wav_path), dtype="float32")
    except Exception:
        # 降级使用 wave 库
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(frames, dtype=np.int16)
            data = pcm.astype(np.float32) / 32768.0

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != TARGET_SAMPLE_RATE:
        num_samples = int(round(len(data) * float(TARGET_SAMPLE_RATE) / sr))
        data = signal.resample(data, num_samples)

    return data


def save_wav_16k(audio_float: np.ndarray, out_path: Path) -> None:
    """将 float32 音频保存为 16kHz 单声道 16-bit WAV。"""
    max_val = np.max(np.abs(audio_float))
    if max_val > 0.99:
        audio_float = audio_float / max_val * 0.98

    pcm16 = (audio_float * 32767.0).astype(np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())


def apply_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """使用 RIR 冲激响应进行快速 FFT 卷积以添加真实房间混响。"""
    # 归一化 RIR 能量
    rir_energy = np.sqrt(np.mean(rir ** 2)) + 1e-9
    rir_norm = rir / rir_energy * 0.05

    # 卷积
    convolved = signal.fftconvolve(audio, rir_norm, mode="full")
    # 截取与原音频长度匹配 (保持尾部轻微衰减)
    target_len = len(audio) + int(TARGET_SAMPLE_RATE * 0.2)
    convolved = convolved[:target_len]

    # 保持原音频 RMS 能量
    orig_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    conv_rms = np.sqrt(np.mean(convolved ** 2)) + 1e-9
    return convolved * (orig_rms / conv_rms)


def mix_noise_at_snr(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """以指定的信噪比 (SNR) 混合背景噪声。"""
    audio_len = len(audio)
    if len(noise) < audio_len:
        # 重复平铺噪声
        repeats = int(math.ceil(audio_len / len(noise)))
        noise = np.tile(noise, repeats)

    # 随机裁剪一段与 audio 等长的噪声
    max_start = max(0, len(noise) - audio_len)
    start_idx = random.randint(0, max_start) if max_start > 0 else 0
    noise_slice = noise[start_idx : start_idx + audio_len]

    audio_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise_slice ** 2)) + 1e-9

    # 计算目标噪声比例: SNR = 20 * log10(audio_rms / (target_noise_rms))
    target_noise_rms = audio_rms / (10.0 ** (snr_db / 20.0))
    scale = target_noise_rms / noise_rms

    mixed = audio + noise_slice * scale
    return mixed


def time_shift_and_pad(audio: np.ndarray, max_shift_sec: float) -> np.ndarray:
    """在音频前后随机添加静音填充，实现时间平移。"""
    max_shift_samples = int(max_shift_sec * TARGET_SAMPLE_RATE)
    if max_shift_samples <= 0:
        return audio

    front_pad = random.randint(0, max_shift_samples)
    back_pad = random.randint(0, max_shift_samples)

    padded = np.pad(audio, (front_pad, back_pad), mode="constant", constant_values=0.0)
    return padded


def augment_single_audio(
    clean_audio: np.ndarray,
    rir_files: List[Path],
    noise_files: List[Path],
    rir_prob: float,
    snr_range: Tuple[float, float],
    max_shift_sec: float,
) -> np.ndarray:
    """对单条音频执行完整的声学增强流水线。"""
    augmented = clean_audio.copy()

    # 1. 随机添加 RIR 房间冲激混响
    if rir_files and random.random() < rir_prob:
        rir_path = random.choice(rir_files)
        try:
            rir_audio = load_wav_16k(rir_path)
            augmented = apply_rir(augmented, rir_audio)
        except Exception as e:
            logger.debug("应用 RIR %s 失败: %s", rir_path.name, e)

    # 2. 随机混合环境背景噪声
    if noise_files:
        noise_path = random.choice(noise_files)
        try:
            noise_audio = load_wav_16k(noise_path)
            snr = random.uniform(snr_range[0], snr_range[1])
            augmented = mix_noise_at_snr(augmented, noise_audio, snr)
        except Exception as e:
            logger.debug("混合噪声 %s 失败: %s", noise_path.name, e)

    # 3. 随机时间平移与边界填充
    augmented = time_shift_and_pad(augmented, max_shift_sec=max_shift_sec)

    # 4. 随机增益扰动 (Volume Jittering: 0.7 ~ 1.3)
    gain = random.uniform(0.7, 1.3)
    augmented = augmented * gain

    return augmented


def main():
    parser = argparse.ArgumentParser(description="唤醒词与负样本声学仿真与环境混音增强")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径 (默认: config.yaml)")
    parser.add_argument("--rounds", type=int, default=1, help="每个样本生成的增强轮数 (默认: 1)")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        cfg_path = Path(__file__).resolve().parent / "config.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / cfg["data"].get("data_dir", "./data")).resolve()
    aug_cfg = cfg["data"].get("augmentation", {})

    synth_dir = (base_dir / cfg["tts"].get("output_dir", "./data/synthetic_speech")).resolve()
    aug_out_dir = data_dir / "augmented"

    pos_in_dir = synth_dir / "positive"
    neg_in_dir = synth_dir / "negative"

    pos_out_dir = aug_out_dir / "positive"
    neg_out_dir = aug_out_dir / "negative"

    # 收集 RIR 与噪声文件列表
    rirs_dir = data_dir / "mit_rirs"
    noise_dir = data_dir / "noise"

    rir_files = list(rirs_dir.glob("**/*.wav")) if rirs_dir.exists() else []
    noise_files = list(noise_dir.glob("**/*.wav")) if noise_dir.exists() else []

    snr_range = tuple(aug_cfg.get("snr_db_range", [5.0, 25.0]))
    rir_prob = float(aug_cfg.get("rir_probability", 0.75))
    max_shift_sec = float(aug_cfg.get("time_shift_max_sec", 0.35))

    print("======================================================================")
    print("🎛️  唤醒词声学环境仿真与数据混音增强 (Augmentation)")
    print(f"   - 干净语音输入: {synth_dir}")
    print(f"   - 增强输出目录: {aug_out_dir}")
    print(f"   - 找到 RIR 冲激响应: {len(rir_files)} 个")
    print(f"   - 找到环境噪声文件: {len(noise_files)} 个")
    print(f"   - SNR 范围: {snr_range[0]} ~ {snr_range[1]} dB")
    print(f"   - RIR 混响概率: {rir_prob * 100:.0f}%")
    print("======================================================================")

    if not rir_files:
        logger.warning("未找到 RIR 混响音频，将跳过混响模拟。可执行 python download_data.py --rirs 下载。")
    if not noise_files:
        logger.warning("未找到背景噪声文件，将跳过背景混音。可执行 python download_data.py --noise 下载。")

    pos_files = list(pos_in_dir.glob("*.wav")) if pos_in_dir.exists() else []
    neg_files = list(neg_in_dir.glob("*.wav")) if neg_in_dir.exists() else []

    if not pos_files:
        logger.error("❌ 未在 %s 发现正样本 WAV 文件！请先运行 python generate_tts.py 生成合成音频。", pos_in_dir)
        sys.exit(1)

    total_tasks = [("positive", pos_files, pos_out_dir), ("negative", neg_files, neg_out_dir)]
    manifest = []

    for split_name, file_list, out_dir in total_tasks:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("正在增强 [%s] 类别音频 (共 %d 个文件, 每文件 %d 轮)...", split_name, len(file_list), args.rounds)

        count = 0
        for f in file_list:
            clean_audio = load_wav_16k(f)
            for r in range(args.rounds):
                aug_audio = augment_single_audio(
                    clean_audio=clean_audio,
                    rir_files=rir_files,
                    noise_files=noise_files,
                    rir_prob=rir_prob,
                    snr_range=snr_range,
                    max_shift_sec=max_shift_sec,
                )
                out_name = f"aug_{f.stem}_r{r}.wav"
                out_path = out_dir / out_name
                save_wav_16k(aug_audio, out_path)

                manifest.append({
                    "filename": out_name,
                    "split": split_name,
                    "is_positive": (split_name == "positive"),
                    "source": f.name,
                })
                count += 1
                if count % 100 == 0:
                    print(f"\r  [{split_name.upper()}] 已完成增强: {count}/{len(file_list) * args.rounds}", end="", flush=True)

        print()
        logger.info("✅ [%s] 增强完成，共输出 %d 个仿真音频文件", split_name, count)

    # 写入清单
    manifest_path = aug_out_dir / "augmented_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("✅ 全部数据增强完成！增强清单已保存至: %s", manifest_path)


if __name__ == "__main__":
    main()
