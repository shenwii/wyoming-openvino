#!/usr/bin/env python3
"""
唤醒词快速测试工具 (Wake Word Standalone & Server Test)
=====================================================
提供两种便捷测试模式:
  1. 本地直接推理模式: 直接读取 .wav 音频文件送入 OpenVINO 模型进行唤醒词评估，输出置信度曲线
  2. Wyoming 客户端模式: 连接正在运行的 Wyoming 唤醒词服务，流式发送音频测试网络通信

使用示例:
    # 模式一: 直接测试 wav 文件在指定模型上的检测情况 (支持 NPU/GPU/CPU)
    python test_wakeword.py input.wav --model models/hey_jarvis_v0.1.onnx --device NPU

    # 模式二: 根据配置文件测试所有配置的唤醒词
    python test_wakeword.py input.wav --config wakeword_config.yaml

    # 模式三: 模拟向远程 Wyoming 服务发送音频并监听唤醒事件
    python test_wakeword.py input.wav --server 127.0.0.1:10400
"""

import asyncio
from pathlib import Path
import time
from typing import List, Optional
import wave

import numpy as np
import typer

app = typer.Typer(help="Wyoming 唤醒词测试与验证工具")


def read_wav_16k_mono(wav_path: Path) -> np.ndarray:
    """读取 WAV 文件并确保为 16kHz 单通道 16-bit PCM。"""
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if width != 2:
        raise ValueError(f"仅支持 16-bit PCM WAV 音频，当前音频 sample width = {width}")

    audio = np.frombuffer(frames, dtype=np.int16)

    # 若为立体声，转为单声道 (取通道平均)
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif channels > 2:
        raise ValueError(f"不支持 {channels} 声道音频，请使用单声道或双声道音频")

    if rate != 16000:
        logger_warn = f"警告: 音频采样率为 {rate}Hz，openWakeWord 标准采样率为 16000Hz！建议预先重采样。"
        print(f"\033[33m{logger_warn}\033[0m")

    return audio


async def test_against_wyoming_server(wav_path: Path, server_addr: str) -> None:
    """连接运行中的 Wyoming 服务并发送音频测试唤醒事件。"""
    from wyoming.audio import AudioChunk, AudioStop
    from wyoming.wake import Detection
    from wyoming.client import AsyncTcpClient

    host, port_str = server_addr.split(":")
    port = int(port_str)
    print(f"正在连接 Wyoming 唤醒词服务: tcp://{host}:{port} ...")

    client = AsyncTcpClient(host, port)
    await client.connect()
    print("✅ 已成功连接到 Wyoming 服务，正在发送音频流...")

    audio_samples = read_wav_16k_mono(wav_path)
    raw_bytes = audio_samples.tobytes()

    chunk_size = 2560  # 1280 采样点 * 2 字节 = 80ms
    timestamp = 0
    detections = []

    # 启动后台接收任务
    async def listen_events():
        while True:
            event = await client.read_event()
            if event is None:
                break
            if Detection.is_type(event.type):
                det = Detection.from_event(event)
                detections.append(det)
                print(f"\n\033[32m🔥 唤醒成功! 模型: '{det.name}' (时间戳: {det.timestamp})\033[0m")

    listener_task = asyncio.create_task(listen_events())

    # 按照 80ms 节奏切片发送音频
    for offset in range(0, len(raw_bytes), chunk_size):
        chunk = raw_bytes[offset : offset + chunk_size]
        chunk_event = AudioChunk(
            rate=16000,
            width=2,
            channels=1,
            audio=chunk,
            timestamp=timestamp,
        )
        await client.write_event(chunk_event.event())
        timestamp += len(chunk) // 2
        await asyncio.sleep(0.02)  # 略微等待，模拟真实网络推流

    await client.write_event(AudioStop().event())
    # 等待响应结算
    await asyncio.sleep(0.5)
    await client.disconnect()
    listener_task.cancel()

    print("\n---------------- 测试报告 ----------------")
    print(f"音频时长: {len(audio_samples) / 16000:.2f} 秒")
    print(f"触发唤醒次数: {len(detections)}")
    for d in detections:
        print(f"  - 唤醒词: {d.name}")
    print("------------------------------------------")


def test_local_direct(
    wav_path: Path,
    model_paths: List[str],
    device: str,
    cutoff: float,
    config_path: Optional[Path] = None,
) -> None:
    """在本地直接加载 OpenVINO 推理引擎评估音频。"""
    import importlib.util

    ww_module_path = Path(__file__).parent / "wyoming-wakeword.py"
    spec = importlib.util.spec_from_file_location("wyoming_wakeword", ww_module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    WakewordServerConfig = mod.WakewordServerConfig
    SingleModelConfig = mod.SingleModelConfig
    WakeWordEngine = mod.WakeWordEngine
    load_config_file = mod.load_config_file
    CHUNK_BYTES = mod.CHUNK_BYTES
    MEL_HISTORY_FRAMES = mod.MEL_HISTORY_FRAMES
    EMBEDDING_DIM = mod.EMBEDDING_DIM

    if config_path and config_path.exists():
        cfg = load_config_file(config_path)
    else:
        cfg = WakewordServerConfig(default_device=device, default_cutoff=cutoff)
        for m_p in model_paths:
            name = Path(m_p).stem.split("_v")[0]
            cfg.models.append(
                SingleModelConfig(
                    name=name,
                    model_path=m_p,
                    phrase=name.replace("_", " ").title(),
                    cutoff=cutoff,
                    device=device,
                )
            )

    print(f"正在初始化 OpenVINO 唤醒词引擎 (目标设备: {cfg.default_device})...")
    engine = WakeWordEngine(cfg)

    audio_samples = read_wav_16k_mono(wav_path)
    raw_bytes = audio_samples.tobytes()

    # 预先使用静音 (全零) 填充滑窗缓冲区，确保短音频 (如 1-2 秒) 也能无延迟检测
    melspec_buffer = np.zeros((MEL_HISTORY_FRAMES, 32), dtype=np.float32)
    embedding_buffer = np.zeros((DEFAULT_CONTEXT_LEN, EMBEDDING_DIM), dtype=np.float32)

    print(f"\n开始分析音频: {wav_path} (时长: {len(audio_samples) / 16000:.2f} 秒)")
    print(f"监听模型: {[m.name for m in engine.classifiers.values()]}")
    print("--------------------------------------------------")

    detected_count = 0
    now = time.time()

    for offset in range(0, len(raw_bytes), CHUNK_BYTES):
        chunk_slice = raw_bytes[offset : offset + CHUNK_BYTES]
        if len(chunk_slice) < CHUNK_BYTES:
            break

        chunk_audio = np.frombuffer(chunk_slice, dtype=np.int16)
        time_sec = (offset // 2) / 16000.0

        if engine.feature_pipeline:
            new_mels = engine.feature_pipeline.compute_melspectrogram(chunk_audio)
            n_new = len(new_mels)
            if n_new >= MEL_HISTORY_FRAMES:
                melspec_buffer[:] = new_mels[-MEL_HISTORY_FRAMES:]
            else:
                melspec_buffer = np.roll(melspec_buffer, -n_new, axis=0)
                melspec_buffer[-n_new:] = new_mels

            emb = engine.feature_pipeline.compute_embedding(melspec_buffer)
            embedding_buffer = np.roll(embedding_buffer, -1, axis=0)
            embedding_buffer[-1] = emb

        for model in engine.classifiers.values():
            score = 0.0
            if model.is_end_to_end:
                float_audio = chunk_audio.astype(np.float32).reshape(1, -1)
                score = model.predict(float_audio)
            else:
                inp = embedding_buffer[-model.context_len:].reshape(
                    1, model.context_len, EMBEDDING_DIM
                )
                score = model.predict(inp)

            if score >= model.cutoff:
                detected_count += 1
                print(
                    f"[{time_sec:6.2f}s] \033[32m🔥 唤醒成功! 模型: '{model.name}', 置信度: {score:.4f} (阈值: {model.cutoff})\033[0m"
                )
            elif score > 0.15:
                print(f"[{time_sec:6.2f}s] 模型: '{model.name}' 置信度上升: {score:.4f}")

    print("--------------------------------------------------")
    print(f"分析完成，共计命中唤醒词 {detected_count} 次。")


@app.command()
def test(
    wav_path: Path = typer.Argument(
        ...,
        help="待测试的 WAV 音频文件路径",
    ),
    model: Optional[List[str]] = typer.Option(
        None,
        "--model",
        "-m",
        help="测试单个或多个唤醒词模型路径",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="从配置文件读取模型列表",
    ),
    device: str = typer.Option(
        "NPU",
        "--device",
        "-d",
        help="本地推理设备 (优先 NPU，自动回退 GPU -> CPU)",
    ),
    cutoff: float = typer.Option(
        0.5,
        "--cutoff",
        help="唤醒阈值 (0.0 ~ 1.0)",
    ),
    server: Optional[str] = typer.Option(
        None,
        "--server",
        "-s",
        help="指定 Wyoming 服务地址 (如 127.0.0.1:10400) 进行网络流式测试",
    ),
) -> None:
    """唤醒词测试入口命令。"""
    if not wav_path.exists():
        print(f"\033[31m错误: 找不到音频文件 {wav_path}\033[0m")
        return

    if server:
        # 网络测试模式
        asyncio.run(test_against_wyoming_server(wav_path, server))
    else:
        # 本地直接推理模式
        model_list = model or []
        if not model_list and not config:
            # 尝试在 models 目录下寻找模型
            found = list(Path("models").glob("*.onnx"))
            found = [str(f) for f in found if f.name not in ["melspectrogram.onnx", "embedding_model.onnx"]]
            if found:
                print(f"自动选用检测到的模型: {found}")
                model_list = found
            else:
                print("\033[31m请通过 --model 或 --config 指定至少一个唤醒词模型！\033[0m")
                return

        test_local_direct(
            wav_path=wav_path,
            model_paths=model_list,
            device=device,
            cutoff=cutoff,
            config_path=config,
        )


if __name__ == "__main__":
    app()
