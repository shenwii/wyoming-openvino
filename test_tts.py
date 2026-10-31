#!/usr/bin/env python3
"""
TTS 快速测试与 WAV 音频生成脚本
=============================
无需连接 Home Assistant，直接在本地测试 Kokoro TTS 合成并将音频保存为标准的 .wav 文件。
生成的 WAV 文件可以在任何浏览器、网页播放器或系统播放器中直接播放。

使用示例:
    python test_tts.py "我来测试下" --model-dir /models/Kokoro-82M-FP16-OpenVINO -o test.wav
    python test_tts.py "Hello, this is a test." --model-dir /models/Kokoro-82M-FP16-OpenVINO -v af_heart -o en.wav
"""

import io
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import openvino as ov
import typer

import importlib.util

# 动态加载同目录下的 wyoming-tts.py
_tts_module_path = Path(__file__).parent / "wyoming-tts.py"
_spec = importlib.util.spec_from_file_location("wyoming_tts", _tts_module_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MultiLangG2P = _mod.MultiLangG2P
process_text_with_misaki = _mod.process_text_with_misaki
resolve_model_and_voices = _mod.resolve_model_and_voices

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
        help="发音人名称（如 zf_xiaobei, af_heart, jf_teio 等；不填则自动推选）",
    ),
    language: Optional[str] = typer.Option(
        "auto",
        "--language",
        "-l",
        help="合成语言（auto / zh / en / ja 等）",
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
        1.0,
        "--speed",
        "-s",
        help="语速倍率",
    ),
):
    print(f"📖 待合成文本: {text}")

    # 1. 解析模型与声音
    model_file, voices, vocab = resolve_model_and_voices(model_dir=model_dir)

    all_voices = list(voices.keys())
    if not voice_name or voice_name not in voices:
        voice_name = all_voices[0]

    print(f"🎙️  使用声音: {voice_name}")

    # 2. 编译模型
    import os
    core = ov.Core()
    model = core.read_model(str(model_file))
    ov_config = {
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_NUM_THREADS": str(min(os.cpu_count() or 4, 8)),
    }
    compiled_model = core.compile_model(model, device, config=ov_config)

    # 3. 初始化 G2P 并转换音素
    g2p_manager = MultiLangG2P(fallback_language="zh" if language == "auto" else language)

    # 自动识别语言
    target_lang = language
    if not target_lang or target_lang == "auto":
        target_lang = g2p_manager.detect_language_from_text(text)

    print(f"🌐 确定语言: {target_lang}")
    g2p = g2p_manager.get_g2p(target_lang)

    # 4. 文本切句与分段推理
    _split_func = getattr(_mod, "split_text_into_chunks", lambda t: [t])
    chunks = _split_func(text)
    print(f"✂️  文本已拆分为 {len(chunks)} 个自然句子")

    all_pcm = bytearray()
    voice_data = voices[voice_name]
    speed_array = np.array([speed], dtype=np.float32)

    for i, chunk_text in enumerate(chunks, 1):
        tokens = process_text_with_misaki(g2p, chunk_text, vocab=vocab)
        token_len = tokens.shape[1]

        if voice_data.ndim == 2:
            idx = min(token_len, len(voice_data) - 1)
            style = voice_data[idx : idx + 1]
        elif voice_data.ndim == 3:
            idx = min(token_len, len(voice_data) - 1)
            style = voice_data[idx]
        elif voice_data.ndim == 1:
            style = voice_data[np.newaxis, :]
        else:
            style = voice_data

        if style.shape != (1, 256):
            style = style.reshape(1, 256)
        style = style.astype(np.float32)

        output_ov = compiled_model([tokens, style, speed_array])
        audio = output_ov[compiled_model.output(0)]
        audio = np.squeeze(audio)

        audio = np.clip(audio, -1.0, 1.0)
        pcm_chunk = (audio * 32767).astype(np.int16).tobytes()
        all_pcm.extend(pcm_chunk)
        print(f"   [{i}/{len(chunks)}] 合成完成: {chunk_text}")

    # 6. 转 16-bit PCM 并打包为 WAV 文件
    wav_bytes = pcm_to_wav(bytes(all_pcm), sample_rate=24000)

    out_path = Path(output)
    out_path.write_bytes(wav_bytes)

    print(f"✅ 生成成功！WAV 音频已保存至: {out_path.resolve()}")
    print("💡 该 WAV 文件可以在任何浏览器、播放器中直接播放。")


if __name__ == "__main__":
    app()
