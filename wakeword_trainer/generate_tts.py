#!/usr/bin/env python3
"""
Dual-engine wake word and hard negative synthetic speech generator (generate_tts.py).
Combines Kokoro OpenVINO TTS (local offline acceleration) and Microsoft Edge-TTS (cloud neural voices),
generating:
  1. Positive wake word samples (primary phrase + aliases) -> label 1.0
  2. Hard negative samples (confusable near-misses & distractors) -> label 0.0
All outputs are normalized to standard 16kHz, 16-bit mono PCM WAV.
"""

import argparse
import asyncio
import io
import json
import logging
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
import yaml

from i18n import t

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_tts")

TARGET_SAMPLE_RATE = 16000


def resample_and_save_wav(
    audio_data: np.ndarray,
    orig_sr: int,
    out_path: Path,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> None:
    """Resample numpy audio array to 16kHz mono 16-bit PCM WAV."""
    if audio_data.dtype in (np.float32, np.float64):
        max_val = np.max(np.abs(audio_data))
        if max_val > 0.99:
            audio_data = audio_data / max_val * 0.98
        pcm16 = (audio_data * 32767.0).astype(np.int16)
    elif audio_data.dtype == np.int16:
        pcm16 = audio_data
    else:
        pcm16 = audio_data.astype(np.int16)

    if orig_sr != target_sr:
        num_samples = int(round(len(pcm16) * float(target_sr) / orig_sr))
        pcm16 = signal.resample(pcm16.astype(np.float32), num_samples).astype(np.int16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        wf.writeframes(pcm16.tobytes())


def convert_audio_file_to_16k_wav(src_file: Path, dst_wav: Path) -> bool:
    """Use ffmpeg to convert arbitrary audio (e.g. MP3 from Edge-TTS) to 16kHz 16-bit mono WAV."""
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(src_file),
        "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        str(dst_wav),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as exc:
        logger.error(f"ffmpeg conversion failed: {exc}")
        return False


class KokoroTTSSynthesizer:
    """Wrapper for local Kokoro OpenVINO TTS engine."""

    def __init__(self, model_dir_str: str, device: str = "CPU", language: str = "zh") -> None:
        self.device = device
        self.language = language
        self.available = False

        candidate_dirs = [
            Path(model_dir_str).resolve(),
            Path(__file__).resolve().parent.parent / "Kokoro-82M-FP16-OpenVINO",
            Path(__file__).resolve().parent / "models" / "Kokoro-82M-FP16-OpenVINO",
            Path("Kokoro-82M-FP16-OpenVINO").resolve(),
        ]
        self.model_dir = None
        for cd in candidate_dirs:
            if cd.exists() and (cd / "openvino_model.xml").exists():
                self.model_dir = cd
                break

        if not self.model_dir:
            logger.warning("Kokoro-82M OpenVINO model directory not found. Kokoro engine disabled.")
            return

        try:
            parent_dir = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(parent_dir))
            import importlib.util

            tts_script = parent_dir / "wyoming-tts" / "wyoming-tts.py"
            spec = importlib.util.spec_from_file_location("wyoming_tts", tts_script)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            self.resolve_model_and_voices = mod.resolve_model_and_voices
            self.KokoroEngine = mod.KokoroEngine
            self.MultiLangG2P = mod.MultiLangG2P
            self.get_or_blend_voice = mod.get_or_blend_voice
            self.split_chunks = mod.split_text_into_streaming_chunks

            model_file, self.voices, vocab = self.resolve_model_and_voices(model_dir=str(self.model_dir))
            self.engine = self.KokoroEngine(model_file=model_file, device=device)
            self.g2p = self.MultiLangG2P(fallback_language=language, vocab=vocab)
            self.available = True
            logger.info(t("tts.kokoro_ready", count=len(self.voices)))
        except Exception as exc:
            logger.warning(t("tts.kokoro_fallback", error=str(exc)))
            self.available = False

    def synthesize(self, text: str, voice_name: str, speed: float, out_wav: Path) -> bool:
        if not self.available:
            return False

        try:
            voice_data = self.get_or_blend_voice(self.voices, voice_name)
            chunks = self.split_chunks(text)
            all_pcm = bytearray()

            for chunk_text in chunks:
                tokens = self.g2p.text_to_tokens(chunk_text, self.language)
                if tokens.shape[1] <= 2:
                    continue
                pcm_chunk = self.engine.synthesize_tokens(tokens, voice_data, speed=speed)
                if pcm_chunk:
                    all_pcm.extend(pcm_chunk)

            if not all_pcm:
                return False

            raw_audio = np.frombuffer(all_pcm, dtype=np.int16)
            resample_and_save_wav(raw_audio, orig_sr=24000, out_path=out_wav, target_sr=TARGET_SAMPLE_RATE)
            return True
        except Exception as exc:
            logger.debug(f"Kokoro synthesis failed ({text}): {exc}")
            return False


class EdgeTTSSynthesizer:
    """Multilingual synthesizer using Microsoft Edge-TTS."""

    DEFAULT_VOICES = {
        "zh": [
            "zh-CN-XiaoxiaoNeural",
            "zh-CN-YunxiNeural",
            "zh-CN-YunjianNeural",
            "zh-CN-XiaoyiNeural",
            "zh-CN-liaoning-XiaobeiNeural",
            "zh-CN-shaanxi-XiaoniNeural",
            "zh-TW-HsiaoChenNeural",
        ],
        "en": [
            "en-US-JennyNeural",
            "en-US-GuyNeural",
            "en-US-AriaNeural",
            "en-US-ChristopherNeural",
            "en-GB-SoniaNeural",
            "en-GB-RyanNeural",
        ],
        "ja": [
            "ja-JP-NanamiNeural",
            "ja-JP-KeitaNeural",
        ],
        "ko": [
            "ko-KR-SunHiNeural",
            "ko-KR-InJoonNeural",
        ],
    }

    def __init__(self, language: str = "zh", custom_voices: Optional[List[str]] = None) -> None:
        self.language = language
        self.voices = custom_voices if (custom_voices and len(custom_voices) > 0) else self.DEFAULT_VOICES.get(language, self.DEFAULT_VOICES.get("en", []))
        try:
            import edge_tts
            self.edge_tts = edge_tts
            self.available = True
            logger.info(t("tts.edge_ready", count=len(self.voices)))
        except ImportError:
            logger.warning("Module 'edge-tts' not found. Run pip install edge-tts.")
            self.available = False

    async def _synth_async(self, text: str, voice: str, rate_str: str, pitch_str: str, volume_str: str, out_mp3: Path) -> bool:
        try:
            comm = self.edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate_str,
                pitch=pitch_str,
                volume=volume_str,
            )
            await comm.save(str(out_mp3))
            return True
        except Exception as exc:
            logger.debug(f"Edge-TTS request failed: {exc}")
            return False

    def synthesize(
        self,
        text: str,
        voice: str,
        rate_str: str,
        pitch_str: str,
        volume_str: str,
        out_wav: Path,
    ) -> bool:
        if not self.available:
            return False

        temp_mp3 = out_wav.with_suffix(".temp.mp3")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(
                self._synth_async(text, voice, rate_str, pitch_str, volume_str, temp_mp3)
            )
            loop.close()

            if success and temp_mp3.exists() and temp_mp3.stat().st_size > 100:
                convert_audio_file_to_16k_wav(temp_mp3, out_wav)
                temp_mp3.unlink(missing_ok=True)
                return True
        except Exception as exc:
            logger.debug(f"Edge-TTS synthesis error: {exc}")
        finally:
            temp_mp3.unlink(missing_ok=True)
        return False


def parse_rate_range(range_list: List[str]) -> Tuple[int, int]:
    """Parse percentage ranges like ['-20%', '+20%'] to (-20, 20)."""
    try:
        min_v = int(range_list[0].replace("%", "").replace("+", ""))
        max_v = int(range_list[1].replace("%", "").replace("+", ""))
        return (min_v, max_v)
    except Exception:
        return (-15, 15)


def parse_pitch_range(range_list: List[str]) -> Tuple[int, int]:
    """Parse pitch ranges like ['-15Hz', '+15Hz'] to (-15, 15)."""
    try:
        min_v = int(range_list[0].replace("Hz", "").replace("+", ""))
        max_v = int(range_list[1].replace("Hz", "").replace("+", ""))
        return (min_v, max_v)
    except Exception:
        return (-10, 10)


def generate_dataset_split(
    texts: List[str],
    target_count: int,
    is_positive: bool,
    out_dir: Path,
    kokoro_synth: KokoroTTSSynthesizer,
    edge_synth: EdgeTTSSynthesizer,
    cfg_tts: dict,
    manifest_records: list,
) -> int:
    """Generate synthesized speech for positive or negative dataset split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    use_kokoro = cfg_tts.get("kokoro", {}).get("enabled", True) and kokoro_synth.available
    use_edge = cfg_tts.get("edge_tts", {}).get("enabled", True) and edge_synth.available

    if not use_kokoro and not use_edge:
        logger.error("No TTS engine available! Install edge-tts or configure Kokoro.")
        return 0

    kokoro_cfg = cfg_tts.get("kokoro", {})
    k_speed_range = kokoro_cfg.get("speed_range", [0.75, 1.25])
    k_voices = kokoro_cfg.get("voices", [])
    if not k_voices and kokoro_synth.available:
        k_voices = list(kokoro_synth.voices.keys())

    edge_cfg = cfg_tts.get("edge_tts", {})
    e_rate_min, e_rate_max = parse_rate_range(edge_cfg.get("rate_range", ["-20%", "+20%"]))
    e_pitch_min, e_pitch_max = parse_pitch_range(edge_cfg.get("pitch_range", ["-15Hz", "+15Hz"]))
    e_voices = edge_cfg.get("voices", [])
    if not e_voices and edge_synth.available:
        e_voices = edge_synth.voices

    prefix = "pos" if is_positive else "neg"
    engine_desc = ("Kokoro + Edge-TTS" if (use_kokoro and use_edge) else ("Kokoro" if use_kokoro else "Edge-TTS"))
    logger.info(t("tts.synth_start", total=target_count, engine=engine_desc))

    retry_count = 0
    max_retries = target_count * 3

    while generated < target_count and retry_count < max_retries:
        retry_count += 1
        text = random.choice(texts)

        candidate_engines = []
        if use_kokoro:
            candidate_engines.append("kokoro")
        if use_edge:
            candidate_engines.append("edge_tts")

        chosen_engine = random.choice(candidate_engines)
        out_filename = f"{prefix}_{chosen_engine}_{generated:05d}.wav"
        out_wav = out_dir / out_filename

        ok = False
        meta = {
            "filename": out_filename,
            "path": str(out_wav.relative_to(out_dir.parent.parent)),
            "text": text,
            "is_positive": is_positive,
            "engine": chosen_engine,
        }

        if chosen_engine == "kokoro":
            speed = round(random.uniform(k_speed_range[0], k_speed_range[1]), 2)
            if kokoro_cfg.get("enable_voice_blending", True) and len(k_voices) >= 2 and random.random() < 0.4:
                v1, v2 = random.sample(k_voices, 2)
                w1 = round(random.uniform(0.3, 0.7), 2)
                voice_name = f"{v1}({w1})+{v2}({round(1.0 - w1, 2)})"
            else:
                voice_name = random.choice(k_voices)

            meta["voice"] = voice_name
            meta["speed"] = speed
            ok = kokoro_synth.synthesize(text=text, voice_name=voice_name, speed=speed, out_wav=out_wav)

        elif chosen_engine == "edge_tts":
            voice_name = random.choice(e_voices)
            rate_val = random.randint(e_rate_min, e_rate_max)
            pitch_val = random.randint(e_pitch_min, e_pitch_max)
            rate_str = f"{rate_val:+d}%"
            pitch_str = f"{pitch_val:+d}Hz"
            meta["voice"] = voice_name
            meta["rate"] = rate_str
            meta["pitch"] = pitch_str
            ok = edge_synth.synthesize(
                text=text,
                voice=voice_name,
                rate_str=rate_str,
                pitch_str=pitch_str,
                volume_str="+0%",
                out_wav=out_wav,
            )

        if ok and out_wav.exists() and out_wav.stat().st_size > 1000:
            generated += 1
            manifest_records.append(meta)
            if generated % 50 == 0 or generated == target_count:
                print(
                    f"\r  {t('tts.synth_progress', count=generated, total=target_count, pct=generated / target_count * 100)}",
                    end="",
                    flush=True,
                )

    print()
    return generated


def main():
    parser = argparse.ArgumentParser(description=t("cli.tts_help"))
    parser.add_argument("--config", default="config.yaml", help="Path to config file [default: config.yaml]")
    parser.add_argument("--positive-only", action="store_true", help="Generate only positive samples")
    parser.add_argument("--negative-only", action="store_true", help="Generate only hard negative samples")
    parser.add_argument("--count-scale", type=float, default=1.0, help="Sample count scale factor [default: 1.0]")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        cfg_path = Path(__file__).resolve().parent / "config.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wakeword_cfg = cfg["wakeword"]
    tts_cfg = cfg["tts"]

    base_dir = Path(__file__).resolve().parent
    out_dir = (base_dir / tts_cfg.get("output_dir", "./data/synthetic_speech")).resolve()
    pos_dir = out_dir / "positive"
    neg_dir = out_dir / "negative"

    lang = wakeword_cfg.get("language", "en")
    primary_phrase = wakeword_cfg.get("phrase", "hey jarvis")
    candidate_phrases = wakeword_cfg.get("candidate_phrases", [])
    positive_phrases = [primary_phrase] + [p for p in candidate_phrases if p != primary_phrase]
    hard_negatives = wakeword_cfg.get("hard_negatives", [])

    pos_target = int(tts_cfg.get("target_positive_samples", 2000) * args.count_scale)
    neg_target = int(tts_cfg.get("target_negative_samples", 1000) * args.count_scale)

    print("======================================================================")
    print("🎙️  Dual-Engine Synthetic Speech Generator (Kokoro + Edge-TTS)")
    print(f"   - Language: {lang}")
    print(f"   - Phrase: '{primary_phrase}'")
    print(f"   - Aliases: {candidate_phrases}")
    print(f"   - Hard negatives count: {len(hard_negatives)}")
    print(f"   - Targets: {pos_target} positive, {neg_target} negative")
    print(f"   - Output directory: {out_dir}")
    print("======================================================================")

    k_cfg = tts_cfg.get("kokoro", {})
    kokoro = KokoroTTSSynthesizer(
        model_dir_str=k_cfg.get("model_dir", "../Kokoro-82M-FP16-OpenVINO"),
        device=k_cfg.get("device", "CPU"),
        language=lang,
    )
    edge = EdgeTTSSynthesizer(
        language=lang,
        custom_voices=tts_cfg.get("edge_tts", {}).get("voices", []),
    )

    manifest = []

    if not args.negative_only:
        generate_dataset_split(
            texts=positive_phrases,
            target_count=pos_target,
            is_positive=True,
            out_dir=pos_dir,
            kokoro_synth=kokoro,
            edge_synth=edge,
            cfg_tts=tts_cfg,
            manifest_records=manifest,
        )

    if not args.positive_only and hard_negatives:
        generate_dataset_split(
            texts=hard_negatives,
            target_count=neg_target,
            is_positive=False,
            out_dir=neg_dir,
            kokoro_synth=kokoro,
            edge_synth=edge,
            cfg_tts=tts_cfg,
            manifest_records=manifest,
        )

    manifest_path = out_dir / "metadata.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(t("tts.synth_complete", path=manifest_path))


if __name__ == "__main__":
    main()
