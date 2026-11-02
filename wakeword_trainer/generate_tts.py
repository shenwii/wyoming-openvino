#!/usr/bin/env python3
"""
双引擎唤醒词与 NG 干扰词语音合成生成器 (generate_tts.py)
======================================================
结合本项目内置 Kokoro OpenVINO TTS (本地离线加速) 与 微软 Edge-TTS (云端高质量)，
以多样化语速、音调、音色及音色加权融合生成:
  1. 正向唤醒词样本 (Positive Samples: 目标词 + 候补/变体词) -> 标签 1.0
  2. NG 负向干扰词样本 (Hard Negatives / Near Misses)        -> 标签 0.0
所有音频输出统一为标准的 16kHz, 16-bit 单声道 WAV 文件。
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_tts")

TARGET_SAMPLE_RATE = 16000


# ==============================================================================
# 辅助函数: 音频重采样与标准 16kHz WAV 写入
# ==============================================================================
def resample_and_save_wav(
    audio_data: np.ndarray,
    orig_sr: int,
    out_path: Path,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> None:
    """将 numpy 音频数组统一重采样为 16kHz 单声道 16-bit PCM WAV。"""
    if audio_data.dtype in (np.float32, np.float64):
        # 归一化并防止削顶截断
        max_val = np.max(np.abs(audio_data))
        if max_val > 0.99:
            audio_data = audio_data / max_val * 0.98
        pcm16 = (audio_data * 32767.0).astype(np.int16)
    elif audio_data.dtype == np.int16:
        pcm16 = audio_data
    else:
        pcm16 = audio_data.astype(np.int16)

    # 重采样
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
    """使用 ffmpeg 将各类音频文件 (如 Edge-TTS 生成的 MP3) 转为 16kHz 16-bit 单声道 WAV。"""
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
        logger.error("ffmpeg 转换音频异常: %s", exc)
        return False


# ==============================================================================
# 引擎 A: 本地 Kokoro OpenVINO TTS 封装
# ==============================================================================
class KokoroTTSSynthesizer:
    """基于 wyoming-openvino 项目内置 KokoroEngine 的本地极速合成器。"""

    def __init__(self, model_dir_str: str, device: str = "CPU", language: str = "zh") -> None:
        self.device = device
        self.language = language
        self.available = False

        # 尝试寻找模型目录
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
            logger.warning("未找到 Kokoro-82M OpenVINO 模型目录，Kokoro TTS 引擎暂不可用。")
            return

        # 动态导入父目录的 wyoming-tts.py
        try:
            parent_dir = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(parent_dir))
            import importlib.util

            tts_script = parent_dir / "wyoming-tts.py"
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
            logger.info("✅ Kokoro OpenVINO TTS 引擎初始化就绪 (模型: %s, 音色数: %d)", self.model_dir.name, len(self.voices))
        except Exception as exc:
            logger.warning("初始化 Kokoro TTS 失败: %s。将仅使用 Edge-TTS 进行合成。", exc)
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
            # Kokoro 原始输出采样率为 24000Hz，重采样并保存为 16000Hz
            resample_and_save_wav(raw_audio, orig_sr=24000, out_path=out_wav, target_sr=TARGET_SAMPLE_RATE)
            return True
        except Exception as exc:
            logger.debug("Kokoro 合成出错 (%s): %s", text, exc)
            return False


# ==============================================================================
# 引擎 B: 微软 Edge-TTS 云端免费高保真合成器
# ==============================================================================
class EdgeTTSSynthesizer:
    """基于微软 edge-tts 的多语种高保真语音合成器。"""

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
    }

    def __init__(self, language: str = "zh", custom_voices: Optional[List[str]] = None) -> None:
        self.language = language
        self.voices = custom_voices if (custom_voices and len(custom_voices) > 0) else self.DEFAULT_VOICES.get(language, self.DEFAULT_VOICES["zh"])
        try:
            import edge_tts
            self.edge_tts = edge_tts
            self.available = True
            logger.info("✅ Edge-TTS 引擎初始化就绪 (可用音色: %d 款)", len(self.voices))
        except ImportError:
            logger.warning("未检测到 'edge-tts' 模块，请执行 pip install edge-tts 进行安装。")
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
            logger.debug("Edge-TTS 请求失败: %s", exc)
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
            # 运行异步合成
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
            logger.debug("Edge-TTS 合成执行异常: %s", exc)
        finally:
            temp_mp3.unlink(missing_ok=True)
        return False


# ==============================================================================
# 主生成调度器 (Coordinator)
# ==============================================================================
def parse_rate_range(range_list: List[str]) -> Tuple[int, int]:
    """解析如 ['-20%', '+20%'] 为 (-20, 20)。"""
    try:
        min_v = int(range_list[0].replace("%", "").replace("+", ""))
        max_v = int(range_list[1].replace("%", "").replace("+", ""))
        return (min_v, max_v)
    except Exception:
        return (-15, 15)


def parse_pitch_range(range_list: List[str]) -> Tuple[int, int]:
    """解析如 ['-15Hz', '+15Hz'] 为 (-15, 15)。"""
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
    """生成指定类别 (正样本 / NG 负样本) 的所有合成语音。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    use_kokoro = cfg_tts.get("kokoro", {}).get("enabled", True) and kokoro_synth.available
    use_edge = cfg_tts.get("edge_tts", {}).get("enabled", True) and edge_synth.available

    if not use_kokoro and not use_edge:
        logger.error("❌ 没有任何可用的 TTS 引擎！请安装 edge-tts 或配置 Kokoro 模型。")
        return 0

    # Kokoro 参数范围
    kokoro_cfg = cfg_tts.get("kokoro", {})
    k_speed_range = kokoro_cfg.get("speed_range", [0.75, 1.25])
    k_voices = kokoro_cfg.get("voices", [])
    if not k_voices and kokoro_synth.available:
        k_voices = list(kokoro_synth.voices.keys())

    # Edge-TTS 参数范围
    edge_cfg = cfg_tts.get("edge_tts", {})
    e_rate_min, e_rate_max = parse_rate_range(edge_cfg.get("rate_range", ["-20%", "+20%"]))
    e_pitch_min, e_pitch_max = parse_pitch_range(edge_cfg.get("pitch_range", ["-15Hz", "+15Hz"]))
    e_voices = edge_cfg.get("voices", [])
    if not e_voices and edge_synth.available:
        e_voices = edge_synth.voices

    prefix = "pos" if is_positive else "neg"
    logger.info(
        "开始生成 [%s] 类别音频 (目标: %d 个, 词条数: %d)...",
        "正向唤醒词" if is_positive else "NG负向干扰词",
        target_count,
        len(texts),
    )

    retry_count = 0
    max_retries = target_count * 3

    while generated < target_count and retry_count < max_retries:
        retry_count += 1
        text = random.choice(texts)

        # 决定使用 Kokoro 还是 Edge-TTS (优先轮换分配)
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
            # 是否尝试发音人融合
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
                print(f"\r  -> 已成功生成 [{prefix.upper()}]: {generated}/{target_count} ({generated / target_count * 100:.1f}%)", end="", flush=True)

    print()
    logger.info("完成 [%s] 生成: %d 个文件成功写入", prefix, generated)
    return generated


def main():
    parser = argparse.ArgumentParser(description="多参数多样化唤醒词与 NG 词 TTS 音频生成")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径 (默认: config.yaml)")
    parser.add_argument("--positive-only", action="store_true", help="仅生成正样本")
    parser.add_argument("--negative-only", action="store_true", help="仅生成 NG 负样本")
    parser.add_argument("--count-scale", type=float, default=1.0, help="样本数缩放因子 (例如 0.1 仅生成十分之一做快速测试)")
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

    lang = wakeword_cfg.get("language", "zh")
    primary_phrase = wakeword_cfg.get("phrase", "你好小智")
    candidate_phrases = wakeword_cfg.get("candidate_phrases", [])
    positive_phrases = [primary_phrase] + [p for p in candidate_phrases if p != primary_phrase]

    hard_negatives = wakeword_cfg.get("hard_negatives", [])

    pos_target = int(tts_cfg.get("target_positive_samples", 2000) * args.count_scale)
    neg_target = int(tts_cfg.get("target_negative_samples", 1000) * args.count_scale)

    print("======================================================================")
    print("🎙️  多引擎唤醒词数据生成流水线 (Kokoro OpenVINO + Microsoft Edge-TTS)")
    print(f"   - 语言模式: {lang}")
    print(f"   - 主唤醒词: '{primary_phrase}'")
    print(f"   - 候补变体词: {candidate_phrases}")
    print(f"   - NG 干扰词数: {len(hard_negatives)}")
    print(f"   - 正样本目标: {pos_target} | 负样本目标: {neg_target}")
    print(f"   - 输出目录: {out_dir}")
    print("======================================================================")

    # 1. 初始化两大 TTS 引擎
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

    # 2. 生成正样本音频
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

    # 3. 生成 NG 负样本音频
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

    # 4. 写入元数据索引
    manifest_path = out_dir / "metadata.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("✅ 合成数据生成完成！元数据记录已写入: %s (共 %d 条记录)", manifest_path, len(manifest))


if __name__ == "__main__":
    main()
