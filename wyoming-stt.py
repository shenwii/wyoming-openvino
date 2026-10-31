#!/usr/bin/env python3
"""
Wyoming STT 服务
================
基于 OpenVINO 加速的 Whisper 语音识别服务，通过 Wyoming 协议对接 Home Assistant。

功能:
    - 使用 OpenVINO GenAI 的 WhisperPipeline 进行高效语音转文字
    - 支持 Intel GPU / CPU / 混合推理 (AUTO / MULTI)
    - 通过 Wyoming 协议与 Home Assistant 无缝集成

使用示例:
    python wyoming-stt.py --model-dir ./whisper-openvino --device GPU --language zh
"""

import asyncio
import logging
from functools import partial

import numpy as np
import openvino_genai
import typer

from wyoming.audio import AudioChunk, AudioStop
from wyoming.asr import Transcribe, Transcript
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logger = logging.getLogger("wyoming-stt")

# ---------------------------------------------------------------------------
# Typer 命令行应用
# ---------------------------------------------------------------------------
app = typer.Typer(help="Wyoming STT 服务 - 基于 OpenVINO 的语音识别")


# ===========================================================================
# Wyoming 事件处理器
# ===========================================================================
class STTEventHandler(AsyncEventHandler):
    """
    Wyoming STT 事件处理器。

    处理来自 Wyoming 客户端的语音识别请求:
        1. 收到 Transcribe 事件 → 重置音频缓冲区
        2. 收到 AudioChunk 事件 → 追加音频数据到缓冲区
        3. 收到 AudioStop 事件  → 调用 OpenVINO 进行推理并返回识别结果
    """

    def __init__(
        self,
        *args,
        wyoming_info: Info,
        pipeline: openvino_genai.WhisperPipeline,
        language: str,
        **kwargs,
    ) -> None:
        """
        初始化 STT 事件处理器。

        参数:
            wyoming_info: Wyoming 服务描述信息（用于 Home Assistant 发现）
            pipeline:     OpenVINO Whisper 推理管线
            language:     识别语言代码（如 zh / en / ja）
        """
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.pipeline = pipeline
        self.language = language
        # 音频缓冲区，用于累积接收到的 PCM 音频数据
        self.audio_buffer = bytearray()

    async def handle_event(self, event: Event) -> bool:
        """处理 Wyoming 协议事件。"""

        if Describe.is_type(event.type):
            # ---- 服务描述请求 ----
            # 返回模型信息，供 Home Assistant 自动发现并注册服务
            await self.write_event(self.wyoming_info.event())

        elif Transcribe.is_type(event.type):
            # ---- 转录请求开始 ----
            # 清空缓冲区，准备接收新的音频流
            self.audio_buffer = bytearray()

        elif AudioChunk.is_type(event.type):
            # ---- 接收音频数据块 ----
            chunk = AudioChunk.from_event(event)
            self.audio_buffer.extend(chunk.audio)

        elif AudioStop.is_type(event.type):
            # ---- 音频接收完毕，执行语音识别 ----
            if not self.audio_buffer:
                logger.warning("收到空的音频数据，跳过识别")
                await self.write_event(Transcript(text="").event())
            else:
                text = await self.call_openvino(self.audio_buffer)
                logger.info("识别结果: %s", text)
                await self.write_event(Transcript(text=text).event())

        return True

    async def call_openvino(self, audio_bytes: bytearray) -> str:
        """
        使用 OpenVINO Whisper 模型进行语音识别。

        处理流程:
            1. 将 16-bit PCM 原始音频转换为 float32 归一化数组
            2. 配置 Whisper 生成参数（语言、最大 token 数等）
            3. 在线程池中执行推理（避免阻塞异步事件循环）
            4. 返回识别出的文本

        参数:
            audio_bytes: 16-bit PCM 格式的原始音频数据（16kHz 采样率）

        返回:
            识别出的文本字符串
        """
        # 将 16-bit PCM 音频转换为 float32，范围归一化到 [-1.0, 1.0]
        audio_array = (
            np.frombuffer(bytes(audio_bytes), dtype=np.int16).astype(np.float32)
            / 32768.0
        )

        # 配置 Whisper 生成参数
        config = self.pipeline.get_generation_config()
        config.max_new_tokens = 448  # Whisper 模型最大生成 token 数
        config.language = f"<|{self.language}|>"  # 指定识别语言
        config.task = "transcribe"  # 转录任务（非翻译）

        # 在线程池中执行推理，避免阻塞事件循环
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self.pipeline.generate, audio_array, config
        )

        return result.texts[0] if result.texts else ""


# ===========================================================================
# 命令行入口
# ===========================================================================
@app.command()
def main(
    model_dir: str = typer.Option(
        ...,
        "--model-dir",
        "-m",
        help="OpenVINO Whisper 模型目录路径（需包含导出的 .xml/.bin 及 tokenizer 文件）",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="服务监听地址",
    ),
    port: int = typer.Option(
        10020,
        "--port",
        "-p",
        help="服务监听端口",
    ),
    device: str = typer.Option(
        "GPU",
        "--device",
        "-d",
        help="OpenVINO 推理设备（CPU / GPU / AUTO / MULTI:CPU,GPU）",
    ),
    language: str = typer.Option(
        "zh",
        "--language",
        "-l",
        help="识别语言代码（如 zh / en / ja / ko / fr）",
    ),
    model_name: str = typer.Option(
        "whisper",
        "--model-name",
        help="模型名称，用于 Wyoming 服务发现和 Home Assistant 显示",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="日志级别（DEBUG / INFO / WARNING / ERROR）",
    ),
) -> None:
    """启动基于 OpenVINO 的 Wyoming STT（语音转文字）服务。"""

    # 初始化日志系统
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("正在加载 Whisper 模型: %s", model_dir)
    logger.info("推理设备: %s", device)

    # ---- 初始化 OpenVINO Whisper 推理管线 ----
    # WhisperPipeline 封装了完整的 Whisper 推理流程:
    # 音频特征提取 → 编码器 → 解码器 → token 解码
    pipeline = openvino_genai.WhisperPipeline(str(model_dir), device)
    logger.info("模型加载完毕")

    # ---- 构建 Wyoming 服务信息 ----
    # 该信息用于 Home Assistant 自动发现和注册 STT 服务
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="wyoming-openvino-stt",
                description=f"基于 OpenVINO 的 {model_name} 语音识别服务",
                models=[
                    AsrModel(
                        name=model_name,
                        description=f"OpenVINO {model_name}",
                        languages=[language],
                        attribution=Attribution(
                            name="OpenAI Whisper (OpenVINO)",
                            url="https://github.com/openai/whisper",
                        ),
                    )
                ],
                attribution=Attribution(
                    name="Wyoming OpenVINO",
                    url="https://github.com/wyoming-openvino",
                ),
            )
        ],
    )

    # ---- 启动 Wyoming 异步服务 ----
    server = AsyncServer.from_uri(f"tcp://{host}:{port}")
    logger.info("服务启动: %s:%d", host, port)

    asyncio.run(
        server.run(
            partial(
                STTEventHandler,
                wyoming_info=wyoming_info,
                pipeline=pipeline,
                language=language,
            )
        )
    )


if __name__ == "__main__":
    app()
