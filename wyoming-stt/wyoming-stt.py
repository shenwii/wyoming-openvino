#!/usr/bin/env python3
"""
Wyoming STT Service
===================
OpenVINO-accelerated Whisper speech recognition service interfacing with Home Assistant
via the standard Wyoming protocol.

Multilingual and Simplified/Traditional Support:
    - Protocol Priority: Honors language specified by Wyoming client requests (e.g. Home Assistant voice assistant pipeline).
    - Auto-detection: If not specified by protocol or set to auto/none, Whisper automatically detects language.
    - Native support for English (en), Chinese (zh), Japanese (ja), Korean (ko), and Whisper's 90+ languages.
    - Chinese Conversion: Use --to-simplified to force Simplified Chinese output; defaults to Traditional Chinese.

Usage Examples:
    # Force Simplified Chinese output
    python wyoming-stt.py --model-dir ./whisper-openvino --device GPU --to-simplified

    # Default output
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

from i18n import t

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger("wyoming-stt")

# ---------------------------------------------------------------------------
# Chinese Simplified / Traditional conversion
# (Tries python-opencc first, falls back to system libopencc, or logs a warning)
# ---------------------------------------------------------------------------
_opencc_t2s = None
_opencc_s2t = None
_opencc_warned = False


def _init_opencc() -> None:
    """Initialize OpenCC converter for Simplified/Traditional Chinese."""
    global _opencc_t2s, _opencc_s2t

    # 1. Attempt to import python binding library
    try:
        import opencc

        _opencc_t2s = opencc.OpenCC("t2s.json")
        _opencc_s2t = opencc.OpenCC("s2t.json")
        logger.debug(t("log.opencc_python_loaded"))
        return
    except Exception:
        pass

    # 2. Attempt to load system libopencc via ctypes
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
                logger.debug(t("log.opencc_system_loaded", lib_name=lib_name))
                return
        except Exception:
            continue


_init_opencc()


def is_chinese_text(text: str) -> bool:
    """Check if the text contains Chinese characters and no Japanese Hiragana/Katakana."""
    has_han = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_kana = bool(re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text))
    return has_han and not has_kana


def convert_chinese(text: str, to_simplified: bool = False) -> str:
    """
    Convert Chinese text between Traditional and Simplified.

    Args:
        text: Original text to convert.
        to_simplified: True to force Simplified Chinese, False for Traditional.
    """
    global _opencc_warned
    if not text:
        return text

    converter = _opencc_t2s if to_simplified else _opencc_s2t
    if converter is not None:
        try:
            return converter.convert(text)
        except Exception as e:
            logger.warning(t("log.opencc_failed", error=str(e)))
            return text

    if not _opencc_warned:
        logger.warning(t("log.opencc_not_found"))
        _opencc_warned = True

    return text


# ---------------------------------------------------------------------------
# Typer CLI application
# ---------------------------------------------------------------------------
app = typer.Typer(help=t("cli.help"))

# Common Whisper languages announced to Home Assistant
COMMON_LANGUAGES = [
    "zh", "en", "ja", "ko", "de", "fr", "es", "ru", "it", "pt",
    "nl", "tr", "pl", "sv", "vi", "th", "ar", "hi", "el", "cs",
]


# ===========================================================================
# Wyoming Event Handler
# ===========================================================================
class STTEventHandler(AsyncEventHandler):
    """
    Wyoming STT Event Handler.

    Handles speech recognition requests from Wyoming clients:
        1. Describe event   -> Returns service info and supported languages
        2. Transcribe event -> Extracts client-requested language, resets audio buffer
        3. AudioChunk event -> Appends PCM audio chunks to buffer
        4. AudioStop event  -> Invokes OpenVINO Whisper inference and returns transcript
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
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.pipeline = pipeline
        self.default_language = (default_language or "auto").strip().lower()
        self.to_simplified = to_simplified
        self.current_language: str = self.default_language
        self.audio_buffer = bytearray()

    async def handle_event(self, event: Event) -> bool:
        """Process an incoming Wyoming protocol event."""

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())

        elif Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self.current_language = transcribe.language.strip().lower()
                logger.info(t("log.protocol_lang_specified", lang=self.current_language))
            else:
                self.current_language = self.default_language
                logger.info(t("log.protocol_lang_default", lang=self.current_language))

            self.audio_buffer = bytearray()

        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.audio_buffer.extend(chunk.audio)

        elif AudioStop.is_type(event.type):
            if not self.audio_buffer:
                logger.warning(t("log.empty_audio"))
                await self.write_event(Transcript(text="").event())
            else:
                text = await self.call_openvino(
                    self.audio_buffer,
                    language=self.current_language,
                )
                logger.info(t("log.transcription_result", lang=self.current_language, text=text))
                await self.write_event(Transcript(text=text).event())

        return True

    async def call_openvino(
        self,
        audio_bytes: bytearray,
        language: Optional[str] = None,
    ) -> str:
        """Execute speech recognition using OpenVINO Whisper model."""
        audio_array = (
            np.frombuffer(bytes(audio_bytes), dtype=np.int16).astype(np.float32)
            / 32768.0
        )

        raw_lang = (language or self.default_language or "auto").strip().lower()
        target_lang = raw_lang.split("-")[0].split("_")[0]
        is_chinese = (target_lang == "zh")

        config = self.pipeline.get_generation_config()
        config.max_new_tokens = 448
        config.task = "transcribe"

        if target_lang and target_lang != "auto":
            config.language = f"<|{target_lang}|>"
        else:
            if hasattr(config, "language"):
                config.language = ""

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

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self.pipeline.generate, audio_array, config
        )

        text = result.texts[0].strip() if result.texts else ""

        if is_chinese or (target_lang == "auto" and is_chinese_text(text)):
            text = convert_chinese(text, to_simplified=self.to_simplified)

        return text


# ===========================================================================
# CLI Entrypoint
# ===========================================================================
@app.command()
def main(
    model_dir: str = typer.Option(
        ...,
        "--model-dir",
        "-m",
        help=t("cli.options.model_dir"),
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help=t("cli.options.host"),
    ),
    port: int = typer.Option(
        10020,
        "--port",
        "-p",
        help=t("cli.options.port"),
    ),
    device: str = typer.Option(
        "GPU",
        "--device",
        "-d",
        help=t("cli.options.device"),
    ),
    language: Optional[str] = typer.Option(
        "auto",
        "--language",
        "-l",
        help=t("cli.options.language"),
    ),
    to_simplified: bool = typer.Option(
        False,
        "--to-simplified",
        help=t("cli.options.to_simplified"),
    ),
    model_name: str = typer.Option(
        "whisper",
        "--model-name",
        help=t("cli.options.model_name"),
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help=t("cli.options.log_level"),
    ),
) -> None:
    """Launch OpenVINO-based Wyoming STT service with automatic language detection, dynamic protocol language switching, and Chinese character conversion."""

    # Initialize logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    lang_setting = language or "auto"
    logger.info(t("log.loading_model", path=model_dir))
    logger.info(t("log.inference_device", device=device))
    logger.info(t("log.default_lang_strategy", lang=lang_setting))
    logger.info(
        t(
            "log.chinese_strategy",
            strategy=t("log.forced_simplified") if to_simplified else t("log.default_traditional"),
        )
    )

    # ---- Initialize OpenVINO Whisper Pipeline ----
    pipeline = openvino_genai.WhisperPipeline(str(model_dir), device)
    logger.info(t("log.model_loaded"))

    # ---- Build Wyoming Service Metadata ----
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="wyoming-openvino-stt",
                description=t("wyoming.program_description", model_name=model_name),
                installed=True,
                version="1.0.0",
                models=[
                    AsrModel(
                        name=model_name,
                        description=t("wyoming.model_description", model_name=model_name),
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

    # ---- Launch Wyoming Async Server ----
    server = AsyncServer.from_uri(f"tcp://{host}:{port}")
    logger.info(
        t("log.service_started", host=host, port=port, lang=lang_setting)
    )

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
