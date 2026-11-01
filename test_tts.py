#!/usr/bin/env python3
"""
TTS 快速测试与 WAV 音频生成脚本 (Kokoro OpenVINO 极速版)
======================================================
无需连接 Home Assistant，直接在本地测试 Kokoro TTS 合成并将音频保存为标准的 .wav 文件。
生成的 WAV 文件可以在任何浏览器、网页播放器或系统播放器中直接播放。

使用示例:
    # 中文合成测试
    python test_tts.py "今天天气真好，我们来测试 OpenVINO 语音合成服务的生成效果。" --model-dir ./Kokoro-82M-FP16-OpenVINO -o test_zh.wav

    # 日文合成测试
    python test_tts.py "こんにちは、今日はとても良い天気ですね。" --model-dir ./Kokoro-82M-FP16-OpenVINO -v jf_alpha -o test_ja.wav

    # 英文合成测试
    python test_tts.py "Hello, this is a fast text-to-speech test using OpenVINO." --model-dir ./Kokoro-82M-FP16-OpenVINO -v af_heart -o test_en.wav
"""

import io
import re
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import importlib.util

# 动态加载同目录下的 wyoming-tts.py
_tts_module_path = Path(__file__).parent / "wyoming-tts.py"
_spec = importlib.util.spec_from_file_location("wyoming_tts", _tts_module_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MultiLangG2P = _mod.MultiLangG2P
KokoroEngine = _mod.KokoroEngine
detect_language = _mod.detect_language
resolve_model_and_voices = _mod.resolve_model_and_voices
get_or_blend_voice = _mod.get_or_blend_voice
split_text_into_streaming_chunks = _mod.split_text_into_streaming_chunks
VOICE_PREFIX_TO_LANG = _mod.VOICE_PREFIX_TO_LANG

app = typer.Typer(help="Kokoro OpenVINO TTS 独立测试与 WAV 生成工具")


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """将原始 16-bit PCM 转换为通用的标准 WAV 文件字节流。"""
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)           # 单声道
        wf.setsampwidth(2)           # 16-bit PCM (2 bytes)
        wf.setframerate(sample_rate) # 24000 Hz
        wf.writeframes(pcm_bytes)
    return wav_io.getvalue()


@app.command()
def main(
    text: str = typer.Argument(..., help="待合成的文本内容"),
    model_dir: str = typer.Option(
        ...,
        "--model-dir",
        "-m",
        help="Kokoro 模型目录（含 openvino_model.xml 及 voices/ 目录）",
    ),
    voice_name: Optional[str] = typer.Option(
        None,
        "--voice-name",
        "-v",
        help="发音人名称或多音色融合表达式（如 zf_xiaobei, jf_alpha, af_heart, zf_xiaobei(0.7)+af_heart(0.3)；不填则自动推选）",
    ),
    language: Optional[str] = typer.Option(
        "auto",
        "--language",
        "-l",
        help="合成语言（auto / zh / ja / en / es / fr 等）",
    ),
    output: str = typer.Option(
        "output.wav",
        "--output",
        "-o",
        help="输出的 WAV 音频文件路径",
    ),
    device: str = typer.Option(
        "CPU",
        "--device",
        "-d",
        help="推理设备（默认 CPU，Kokoro 在 CPU 上极快且兼容性最佳）",
    ),
    speed: float = typer.Option(
        0.85,
        "--speed",
        "-s",
        help="语速倍率（默认 0.85；>1.0 加速，<1.0 减速）",
    ),
    threads: Optional[int] = typer.Option(
        None,
        "--threads",
        "-t",
        help="OpenVINO CPU 推理线程数（默认 None 自适应优化）",
    ),
):
    print(f"📖 待合成文本: {text}")

    # 1. 解析模型与声音
    model_file, voices, vocab = resolve_model_and_voices(model_dir=model_dir)

    all_voices = list(voices.keys())
    if not voice_name:
        preferred_voices = (
            [v for v in all_voices if v.startswith("zf_") or v.startswith("zm_")] +
            [v for v in all_voices if v.startswith("jf_") or v.startswith("jm_")] +
            [v for v in all_voices if v.startswith("af_") or v.startswith("am_")]
        )
        voice_name = preferred_voices[0] if preferred_voices else all_voices[0]

    voice_data = get_or_blend_voice(voices, voice_name)
    print(f"🎙️  使用声音: {voice_name}")

    # 2. 编译 OpenVINO 模型
    engine = KokoroEngine(model_file=model_file, device=device, threads=threads)

    # 3. 确定语言逻辑（假名必定为 ja；有显式指定或音色指定以指定为准）
    detected = detect_language(text, default_lang="zh")
    if language and language != "auto":
        target_lang = language
    elif voice_name:
        prefix = voice_name[0].lower()
        target_lang = VOICE_PREFIX_TO_LANG.get(prefix, detected)
    else:
        target_lang = detected

    print(f"🌐 确定语言: {target_lang} (文本检测: {detected})")

    # 4. 初始化 G2P 管理器
    g2p_manager = MultiLangG2P(
        fallback_language=target_lang,
        vocab=vocab,
    )

    # 5. 文本切句与流式生成
    chunks = split_text_into_streaming_chunks(text)
    print(f"✂️  文本已拆分为 {len(chunks)} 个流式语音从句")

    all_pcm = bytearray()

    for i, chunk_text in enumerate(chunks, 1):
        # 分句语言精细判定
        chunk_lang = target_lang
        if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", chunk_text):
            chunk_lang = "ja"
        elif re.search(r"[\u4E00-\u9FFF]", chunk_text) and chunk_lang not in ("ja", "zh"):
            chunk_lang = "zh"
        elif re.search(r"^[a-zA-Z0-9\s\.,!\?'-]+$", chunk_text):
            chunk_lang = "en"

        tokens = g2p_manager.text_to_tokens(chunk_text, chunk_lang)
        if tokens.shape[1] <= 2:
            continue

        pcm_chunk = engine.synthesize_tokens(tokens, voice_data, speed=speed)
        if pcm_chunk:
            all_pcm.extend(pcm_chunk)
            print(f"   [{i}/{len(chunks)}] 合成完成: {chunk_text} (tokens={tokens.shape[1]})")

    # 6. 转 16-bit PCM 并打包为 WAV 文件
    wav_bytes = pcm_to_wav(bytes(all_pcm), sample_rate=24000)

    out_path = Path(output)
    out_path.write_bytes(wav_bytes)

    print(f"✅ 生成成功！WAV 音频已保存至: {out_path.resolve()}")
    print("💡 该 WAV 文件可以在任何浏览器、播放器中直接播放。")


if __name__ == "__main__":
    app()
