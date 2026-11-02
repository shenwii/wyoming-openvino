#!/usr/bin/env python3
"""
Wyoming STT 服务
================
基于 OpenVINO 加速的 Whisper 语音识别服务，通过 Wyoming 协议对接 Home Assistant。

多语言与简繁支持机制:
    - 语言优先采用 Wyoming 协议在请求中指定的语言（如 Home Assistant 语音助理配置的语言）
    - 若 Wyoming 协议未指定或命令行设为 auto/不传，Whisper 模型将自动检测输入的语言
    - 原生支持中文 (zh)、英语 (en)、日语 (ja) 以及 Whisper 的 90+ 种多国语言
    - 中文简繁转换控制: 传入 --to-simplified 参数时强制输出简体中文；未传入时默认输出繁体中文

使用示例:
    # 强制输出简体中文 (推荐普通话用户)
    python wyoming-stt.py --model-dir ./whisper-openvino --device GPU --to-simplified

    # 默认繁体中文输出
    python wyoming-stt.py --model-dir ./whisper-openvino --device GPU
"""

import asyncio
import ctypes
import logging
import re
from functools import partial
from typing import Optional

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
# 中文繁简转换 (优先使用 opencc 库，其次回退至系统 libopencc，均无则警告)
# ---------------------------------------------------------------------------
_opencc_t2s = None
_opencc_s2t = None
_opencc_warned = False


def _init_opencc() -> None:
    """初始化繁简转换器。"""
    global _opencc_t2s, _opencc_s2t

    # 1. 尝试导入 python 绑定库
    try:
        import opencc

        _opencc_t2s = opencc.OpenCC("t2s.json")
        _opencc_s2t = opencc.OpenCC("s2t.json")
        logger.debug("成功加载 Python opencc 繁简转换库")
        return
    except Exception:
        pass

    # 2. 尝试通过 ctypes 加载系统 libopencc 共享库
    for lib_name in (
        "libopencc.so.1.4",
        "libopencc.so.1",
        "libopencc.so",
        "opencc",
    ):
        try:
            lib = ctypes.cdll.LoadLibrary(lib_name)
            lib.opencc_open.restype = ctypes.c_void_p
            lib.opencc_open.argtypes = [ctypes.c_char_p]
            lib.opencc_convert_utf8.restype = ctypes.c_void_p
            lib.opencc_convert_utf8.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            if hasattr(lib, "opencc_convert_utf8_free"):
                lib.opencc_convert_utf8_free.restype = None
                lib.opencc_convert_utf8_free.argtypes = [ctypes.c_void_p]

            t2s_desc = lib.opencc_open(b"t2s.json")
            s2t_desc = lib.opencc_open(b"s2t.json")

            if t2s_desc and s2t_desc and t2s_desc != -1 and s2t_desc != -1:

                class _CtypesConverter:

                    def __init__(self, desc, lib_ref):
                        self._desc = desc
                        self._lib = lib_ref

                    def convert(self, text: str) -> str:
                        if not text:
                            return text
                        encoded = text.encode("utf-8")
                        ptr = self._lib.opencc_convert_utf8(
                            self._desc, encoded, len(encoded)
                        )
                        if not ptr:
                            return text
                        res = ctypes.string_at(ptr).decode("utf-8")
                        if hasattr(self._lib, "opencc_convert_utf8_free"):
                            self._lib.opencc_convert_utf8_free(ptr)
                        return res

                _opencc_t2s = _CtypesConverter(t2s_desc, lib)
                _opencc_s2t = _CtypesConverter(s2t_desc, lib)
                logger.debug("成功通过 ctypes 加载系统 libopencc 动态库 (%s)", lib_name)
                return
        except Exception:
            continue


_init_opencc()


def is_chinese_text(text: str) -> bool:
    """判断文本是否包含汉字且不含日文平假名/片假名。"""
    has_han = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_kana = bool(re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text))
    return has_han and not has_kana


def convert_chinese(text: str, to_simplified: bool = False) -> str:
    """
    对中文文本进行繁简转换。

    参数:
        text: 待转换的原始文本
        to_simplified: True 则强制转换为简体中文；False 则转换为繁体中文
    """
    global _opencc_warned
    if not text:
        return text

    converter = _opencc_t2s if to_simplified else _opencc_s2t
    if converter is not None:
        try:
            return converter.convert(text)
        except Exception as e:
            logger.warning("繁简转换执行异常: %s", e)
            return text

    if not _opencc_warned:
        logger.warning(
            "未检测到可用 opencc 库，繁简转换未生效。建议安装: pip install opencc-python-reimplemented"
        )
        _opencc_warned = True

    return text

# ---------------------------------------------------------------------------
# Typer 命令行应用
# ---------------------------------------------------------------------------
app = typer.Typer(help="Wyoming STT 服务 - 基于 OpenVINO 的多语言语音识别")

# Whisper 常用多语言列表（向 Home Assistant 宣告）
COMMON_LANGUAGES = [
    "zh", "en", "ja", "ko", "de", "fr", "es", "ru", "it", "pt",
    "nl", "tr", "pl", "sv", "vi", "th", "ar", "hi", "el", "cs",
]


# ===========================================================================
# Wyoming 事件处理器
# ===========================================================================
class STTEventHandler(AsyncEventHandler):
    """
    Wyoming STT 事件处理器。

    处理来自 Wyoming 客户端的语音识别请求:
        1. 收到 Describe 事件   → 返回服务及支持的语言列表
        2. 收到 Transcribe 事件 → 提取客户端指定的语言（若有），重置音频缓冲区
        3. 收到 AudioChunk 事件 → 追加音频数据到缓冲区
        4. 收到 AudioStop 事件  → 调用 OpenVINO 进行推理并返回识别结果
    """

    def __init__(
        self,
        *args,
        wyoming_info: Info,
        pipeline: openvino_genai.WhisperPipeline,
        default_language: Optional[str] = "auto",
        to_simplified: bool = False,
        **kwargs,
    ) -> None:
        """
        初始化 STT 事件处理器。

        参数:
            wyoming_info:     Wyoming 服务描述信息
            pipeline:         OpenVINO Whisper 推理管线
            default_language: 默认识别语言代码（如 auto / zh / en / ja）
            to_simplified:    中文识别模式下是否强制转换为简体中文（默认 False 为繁体中文）
        """
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.pipeline = pipeline
        self.default_language = (default_language or "auto").strip().lower()
        self.to_simplified = to_simplified
        self.current_language: str = self.default_language
        # 音频缓冲区，用于累积接收到的 PCM 音频数据
        self.audio_buffer = bytearray()

    async def handle_event(self, event: Event) -> bool:
        """处理 Wyoming 协议事件。"""

        if Describe.is_type(event.type):
            # ---- 服务描述请求 ----
            await self.write_event(self.wyoming_info.event())

        elif Transcribe.is_type(event.type):
            # ---- 转录请求开始 ----
            transcribe = Transcribe.from_event(event)
            # 优先使用客户端在 Wyoming 协议中指定的语言
            if transcribe.language:
                self.current_language = transcribe.language.strip().lower()
                logger.info("Wyoming 协议指定识别语言: %s", self.current_language)
            else:
                self.current_language = self.default_language
                logger.info("Wyoming 协议未指定语言，使用默认/自动检测: %s", self.current_language)

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
                text = await self.call_openvino(
                    self.audio_buffer,
                    language=self.current_language,
                )
                logger.info("识别结果 (语言=%s): %s", self.current_language, text)
                await self.write_event(Transcript(text=text).event())

        return True

    async def call_openvino(
        self,
        audio_bytes: bytearray,
        language: Optional[str] = None,
    ) -> str:
        """
        使用 OpenVINO Whisper 模型进行语音识别。

        处理流程:
            1. 将 16-bit PCM 原始音频转换为 float32 归一化数组
            2. 配置 Whisper 生成参数:
               - 若指定了语言代码（如 zh / en / ja），传入 <|zh|> 进行针对性识别
               - 若语言为 "auto" 或空，清空语言限制，由 Whisper 原生自动检测语言
               - 若为中文模式，注入 initial_prompt 进行简繁提示引导
            3. 在线程池中执行推理
            4. 对中文输出执行繁简规范化（--to-simplified 强制简体，否则默认繁体）
            5. 返回识别文本

        参数:
            audio_bytes: 16-bit PCM 格式的原始音频数据（16kHz 采样率）
            language:    识别语言（zh / en / ja / auto 等）

        返回:
            识别出的文本字符串
        """
        # 将 16-bit PCM 音频转换为 float32，范围归一化到 [-1.0, 1.0]
        audio_array = (
            np.frombuffer(bytes(audio_bytes), dtype=np.int16).astype(np.float32)
            / 32768.0
        )

        raw_lang = (language or self.default_language or "auto").strip().lower()
        # 兼容地区子标签（如 zh-cn, zh-tw 提取主语言 zh）
        target_lang = raw_lang.split("-")[0].split("_")[0]
        is_chinese = (target_lang == "zh")

        # 配置 Whisper 生成参数
        config = self.pipeline.get_generation_config()
        config.max_new_tokens = 448
        config.task = "transcribe"

        if target_lang and target_lang != "auto":
            # 客户端或配置显式指定了语言，例如 <|zh|>, <|en|>, <|ja|>
            config.language = f"<|{target_lang}|>"
        else:
            # 自动模式：清空语言指定，触发 Whisper 原生自动语种检测
            if hasattr(config, "language"):
                config.language = ""

        # 中文模式下注入 prompt 引导解码器
        if is_chinese:
            prompt = (
                "以下是普通话句子，请用简体中文输出。"
                if self.to_simplified
                else "以下是普通話句子，請用繁體中文輸出。"
            )
            if hasattr(config, "initial_prompt"):
                config.initial_prompt = prompt
            elif hasattr(config, "prompt"):
                config.prompt = prompt
        else:
            if hasattr(config, "initial_prompt"):
                config.initial_prompt = ""
            elif hasattr(config, "prompt"):
                config.prompt = ""

        # 在线程池中执行推理，避免阻塞事件循环
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self.pipeline.generate, audio_array, config
        )

        text = result.texts[0].strip() if result.texts else ""

        # 中文繁简处理逻辑：
        # 若指定了 zh，或者 auto 模式下检测到纯正中文（包含汉字且不含日文假名）
        if is_chinese or (target_lang == "auto" and is_chinese_text(text)):
            text = convert_chinese(text, to_simplified=self.to_simplified)

        return text


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
    language: Optional[str] = typer.Option(
        "auto",
        "--language",
        "-l",
        help="默认识别语言代码（默认为 auto 自动检测；若 Wyoming 协议传入了语言则优先遵循协议）",
    ),
    to_simplified: bool = typer.Option(
        False,
        "--to-simplified",
        help="中文识别模式下强制输出简体中文（未指定时默认为繁体中文）",
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
    """启动基于 OpenVINO 的 Wyoming STT 服务，支持自动语种检测、Wyoming 协议动态指定语言与中文简繁输出控制。"""

    # 初始化日志系统
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    lang_setting = language or "auto"
    logger.info("正在加载 Whisper 模型: %s", model_dir)
    logger.info("推理设备: %s", device)
    logger.info("默认语言策略: %s (优先遵循 Wyoming 协议请求)", lang_setting)
    logger.info(
        "中文简繁策略: %s",
        "强制简体 (--to-simplified)" if to_simplified else "默认繁体",
    )

    # ---- 初始化 OpenVINO Whisper 推理管线 ----
    pipeline = openvino_genai.WhisperPipeline(str(model_dir), device)
    logger.info("模型加载完毕")

    # ---- 构建 Wyoming 服务信息 ----
    # 宣告完整语言列表，使 Home Assistant 任意语言管线均可对接本服务
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="wyoming-openvino-stt",
                description=f"基于 OpenVINO 的 {model_name} 多语言语音识别服务 (支持自动检测及中/英/日等)",
                installed=True,
                version="1.0.0",
                models=[
                    AsrModel(
                        name=model_name,
                        description=f"OpenVINO {model_name} (Auto/Multilingual)",
                        languages=COMMON_LANGUAGES,
                        attribution=Attribution(
                            name="OpenAI Whisper (OpenVINO)",
                            url="https://github.com/openai/whisper",
                        ),
                        installed=True,
                        version="1.0.0",
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
    logger.info("STT 服务启动: %s:%d (默认: %s，支持协议指定与自动识别)", host, port, lang_setting)

    asyncio.run(
        server.run(
            partial(
                STTEventHandler,
                wyoming_info=wyoming_info,
                pipeline=pipeline,
                default_language=lang_setting,
                to_simplified=to_simplified,
            )
        )
    )


if __name__ == "__main__":
    app()
