#!/usr/bin/env python3
"""
Wake word acoustic simulation and audio augmentation module (augment_data.py).
Applies:
  1. Room Impulse Response (RIR) reverberation simulation (MIT RIR dataset)
  2. Ambient noise background mixing with random SNR (5 to 25 dB)
  3. Random time shifting and silence padding
  4. Dynamic range volume perturbation (Volume Jittering)
Transforms clean TTS speech into robust real-world training samples.
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

from i18n import t

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("augment_data")

TARGET_SAMPLE_RATE = 16000


def load_wav_16k(wav_path: Path) -> np.ndarray:
    """Read a WAV file and ensure 16kHz mono float32 (-1.0 to 1.0)."""
    try:
        data, sr = sf.read(str(wav_path), dtype="float32")
    except Exception:
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
    """Save float32 audio as 16kHz mono 16-bit WAV."""
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
    """Fast FFT convolution with room impulse response (RIR) to simulate room reverberation."""
    rir_energy = np.sqrt(np.mean(rir ** 2)) + 1e-9
    rir_norm = rir / rir_energy * 0.05

    convolved = signal.fftconvolve(audio, rir_norm, mode="full")
    target_len = len(audio) + int(TARGET_SAMPLE_RATE * 0.2)
    convolved = convolved[:target_len]

    orig_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    conv_rms = np.sqrt(np.mean(convolved ** 2)) + 1e-9
    return convolved * (orig_rms / conv_rms)


def mix_noise_at_snr(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix background noise at target signal-to-noise ratio (SNR in dB)."""
    audio_len = len(audio)
    if len(noise) < audio_len:
        repeats = int(math.ceil(audio_len / len(noise)))
        noise = np.tile(noise, repeats)

    max_start = max(0, len(noise) - audio_len)
    start_idx = random.randint(0, max_start) if max_start > 0 else 0
    noise_slice = noise[start_idx : start_idx + audio_len]

    audio_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise_slice ** 2)) + 1e-9

    target_noise_rms = audio_rms / (10.0 ** (snr_db / 20.0))
    scale = target_noise_rms / noise_rms

    mixed = audio + noise_slice * scale
    return mixed


def time_shift_and_pad(audio: np.ndarray, max_shift_sec: float) -> np.ndarray:
    """Add random silence padding before and after audio to implement temporal shift."""
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
    """Execute complete acoustic augmentation pipeline on a single audio clip."""
    augmented = clean_audio.copy()

    # 1. Random RIR room reverberation
    if rir_files and random.random() < rir_prob:
        rir_path = random.choice(rir_files)
        try:
            rir_audio = load_wav_16k(rir_path)
            augmented = apply_rir(augmented, rir_audio)
        except Exception as e:
            logger.debug("Failed to apply RIR %s: %s", rir_path.name, e)

    # 2. Random background noise mixing
    if noise_files:
        noise_path = random.choice(noise_files)
        try:
            noise_audio = load_wav_16k(noise_path)
            snr = random.uniform(snr_range[0], snr_range[1])
            augmented = mix_noise_at_snr(augmented, noise_audio, snr)
        except Exception as e:
            logger.debug("Failed to mix noise %s: %s", noise_path.name, e)

    # 3. Random time shifting and silence padding
    augmented = time_shift_and_pad(augmented, max_shift_sec=max_shift_sec)

    # 4. Random volume jittering (0.7 to 1.3)
    gain = random.uniform(0.7, 1.3)
    augmented = augmented * gain

    return augmented


def main():
    parser = argparse.ArgumentParser(description=t("cli.augment_help"))
    parser.add_argument("--config", default="config.yaml", help="Path to config file [default: config.yaml]")
    parser.add_argument("--rounds", type=int, default=1, help="Augmentation rounds per sample [default: 1]")
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

    # Collect RIR and noise files
    rirs_dir = data_dir / "mit_rirs"
    noise_dir = data_dir / "noise"

    rir_files = list(rirs_dir.glob("**/*.wav")) if rirs_dir.exists() else []
    noise_files = list(noise_dir.glob("**/*.wav")) if noise_dir.exists() else []

    snr_range = tuple(aug_cfg.get("snr_db_range", [5.0, 25.0]))
    rir_prob = float(aug_cfg.get("rir_probability", 0.75))
    max_shift_sec = float(aug_cfg.get("time_shift_max_sec", 0.35))

    print("======================================================================")
    print("🎛️  Acoustic Environment Simulation & Audio Augmentation")
    print(f"   - Input directory: {synth_dir}")
    print(f"   - Output directory: {aug_out_dir}")
    print(f"   - RIR files found: {len(rir_files)}")
    print(f"   - Noise files found: {len(noise_files)}")
    print(f"   - SNR range: {snr_range[0]} ~ {snr_range[1]} dB")
    print(f"   - RIR probability: {rir_prob * 100:.0f}%")
    print("======================================================================")

    pos_files = list(pos_in_dir.glob("*.wav")) if pos_in_dir.exists() else []
    neg_files = list(neg_in_dir.glob("*.wav")) if neg_in_dir.exists() else []

    if not pos_files:
        logger.error("No positive WAV files found in %s! Run generate_tts.py first.", pos_in_dir)
        sys.exit(1)

    total_tasks = [("positive", pos_files, pos_out_dir), ("negative", neg_files, neg_out_dir)]
    manifest = []

    for split_name, file_list, out_dir in total_tasks:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(t("augment.augmenting", desc=split_name, count=len(file_list)))

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
                    print(f"\r  [{split_name.upper()}] {count}/{len(file_list) * args.rounds}", end="", flush=True)

        print()
        logger.info(t("augment.augment_complete", desc=split_name, path=out_dir, count=count))

    manifest_path = aug_out_dir / "augmented_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nManifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
