#!/usr/bin/env python3
"""
Wyoming TTS 服务
================
基于 OpenVINO 加速的 Kokoro 文字转语音服务，通过 Wyoming 协议对接 Home Assistant。

多语言与音色机制:
    - 语言优先采用 Wyoming 协议在请求中指定的语言（如 HA 配置的 voice.language）
    - 若协议未指定语言，自动依据请求的发音人名称前缀推断（如 zf_* -> zh, jf_* -> ja, af_* -> en）
    - 若仍未指定，根据待合成的文本内容自动推测（日文假名 -> ja, 汉字 -> zh, 字母 -> en）
    - 命令行 --language 默认为 auto，完全可不传
    - 启动时自动扫描并注册 voices.npy 中的全部发音人和语种供 Home Assistant 界面自由选择

使用示例:
    # 默认无需传 --language 和 --voice-name，全部由 HA 或自动策略控制
    python wyoming-tts.py \\
        --model-path ./kokoro.onnx \\
        --voices-path ./voices.npy \\
        --device GPU
"""

import asyncio
import logging
import re
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import openvino as ov
import typer

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logger = logging.getLogger("wyoming-tts")

# ---------------------------------------------------------------------------
# Typer 命令行应用
# ---------------------------------------------------------------------------
app = typer.Typer(help="Wyoming TTS 服务 - 基于 OpenVINO 的多语言语音合成 (支持中/英/日等)")


# ===========================================================================
# Kokoro 音素词汇表
# ===========================================================================
_PAD = "$"
_PUNCTUATION = ';:,.!?¡¿—…"«»"" '
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_LETTERS_IPA = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴ"
    "øɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃ"
    "ˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̈̃ˡ̩"
)
_SYMBOLS = [_PAD] + list(_PUNCTUATION) + list(_LETTERS) + list(_LETTERS_IPA)
VOCAB: Dict[str, int] = {s: i for i, s in enumerate(_SYMBOLS)}

# Kokoro 声音名称前缀与语言代码的对应
VOICE_PREFIX_TO_LANG: Dict[str, str] = {
    "a": "en",  # American English
    "b": "en",  # British English
    "z": "zh",  # Chinese (Mandarin)
    "j": "ja",  # Japanese
    "e": "es",  # Spanish
    "f": "fr",  # French
    "h": "hi",  # Hindi
    "i": "it",  # Italian
    "p": "pt",  # Portuguese
    "k": "ko",  # Korean
}


# ===========================================================================
# 多语言 G2P（字素到音素）转换管理器
# ===========================================================================
class MultiLangG2P:
    """
    多语言 G2P 转换管理器。
    按需延迟加载各语言的 G2P 引擎，支持同一服务内同时处理中、英、日等语言文本。
    """

    def __init__(self, fallback_language: str = "zh") -> None:
        self.fallback_language = fallback_language.split("-")[0].lower()
        self._g2p_instances: Dict[str, object] = {}

    def get_g2p(self, language: Optional[str] = None):
        """获取指定语言的 G2P 实例（按需加载并缓存）。"""
        lang = (language or self.fallback_language or "zh").split("-")[0].lower()
        if lang == "auto":
            lang = self.fallback_language or "zh"

        if lang in self._g2p_instances:
            return self._g2p_instances[lang]

        g2p = self._create_g2p(lang)
        self._g2p_instances[lang] = g2p
        return g2p

    def _create_g2p(self, lang: str):
        """创建对应语言的 misaki G2P 引擎。"""
        try:
            if lang == "zh":
                from misaki.zh import G2P
                logger.info("已加载中文 G2P 引擎 (misaki.zh)")
                return G2P()

            elif lang == "en":
                from misaki.en import G2P
                logger.info("已加载英文 G2P 引擎 (misaki.en)")
                return G2P(trf=False)

            elif lang == "ja":
                from misaki.ja import G2P
                logger.info("已加载日文 G2P 引擎 (misaki.ja)")
                return G2P()

            elif lang == "es":
                from misaki.es import G2P
                return G2P()

            elif lang == "fr":
                from misaki.fr import G2P
                return G2P()

            elif lang == "hi":
                from misaki.hi import G2P
                return G2P()

            elif lang == "it":
                from misaki.it import G2P
                return G2P()

            elif lang == "pt":
                from misaki.pt import G2P
                return G2P()

            elif lang == "ko":
                from misaki.ko import G2P
                return G2P()

            else:
                logger.warning("未找到专门的 '%s' G2P 引擎，回退到基础英文/通用 G2P", lang)
                from misaki.en import G2P
                return G2P(trf=False)

        except ImportError as exc:
            logger.error(
                "加载语言 '%s' 的 G2P 引擎失败，请确保安装了该语种依赖 (如 pypinyin/jieba/fugashi/unidic-lite): %s",
                lang,
                exc,
            )
            from misaki.en import G2P
            logger.warning("回退到 misaki.en 基础 G2P")
            return G2P(trf=False)

    def detect_language_from_text(self, text: str) -> str:
        """
        根据待合成文本内容自动识别语言:
            - 包含日文平假名/片假名 → 日语 (ja)
            - 包含中文字符且无假名 → 中文 (zh)
            - 纯拉丁字母/英文 → 英语 (en)
            - 无法判定（如数字或标点） → 回退到兜底语言
        """
        # 日文平假名/片假名范围
        if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", text):
            return "ja"

        # 中文字符范围
        if re.search(r"[\u4E00-\u9FFF]", text):
            return "zh"

        # 包含英文字母
        if re.search(r"[a-zA-Z]", text):
            return "en"

        return self.fallback_language or "zh"


def process_text_with_misaki(g2p, text: str) -> np.ndarray:
    """
    使用 misaki 将文本转换为 Kokoro 模型所需的 token ID 序列。
    """
    phoneme_parts: List[str] = []

    for _graphemes, phonemes in g2p(text):
        if phonemes is not None:
            phoneme_parts.append(phonemes)

    phoneme_str = "".join(phoneme_parts)

    if not phoneme_str:
        logger.warning("文本 '%s' 未能提取出有效音素", text)
        return np.array([[0, 0]], dtype=np.int64)

    # 逐字符查表转换为 token ID
    token_ids = [0] + [VOCAB.get(c, 0) for c in phoneme_str] + [0]
    return np.array([token_ids], dtype=np.int64)


# ===========================================================================
# Wyoming 事件处理器
# ===========================================================================
class TTSEventHandler(AsyncEventHandler):
    """
    Wyoming TTS 事件处理器。
    """

    def __init__(
        self,
        *args,
        wyoming_info: Info,
        compiled_model: ov.CompiledModel,
        g2p_manager: MultiLangG2P,
        voices: dict,
        default_voice_name: str,
        default_language: str,
        speed: float,
        sample_rate: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.compiled_model = compiled_model
        self.g2p_manager = g2p_manager
        self.voices = voices
        self.default_voice_name = default_voice_name
        self.default_language = default_language
        self.speed = speed
        self.sample_rate = sample_rate

    async def handle_event(self, event: Event) -> bool:
        """处理 Wyoming 协议事件。"""

        if Describe.is_type(event.type):
            # ---- 服务描述请求 ----
            await self.write_event(self.wyoming_info.event())

        elif Synthesize.is_type(event.type):
            # ---- 语音合成请求 ----
            synth = Synthesize.from_event(event)
            text = synth.text

            if not text or not text.strip():
                logger.warning("收到空的合成文本，发送空音频流")
                await self.write_event(
                    AudioStart(rate=self.sample_rate, width=2, channels=1).event()
                )
                await self.write_event(AudioStop().event())
                return True

            # 1. 确定声音：优先使用 Wyoming 协议指定的 voice
            voice_name = self.default_voice_name
            if synth.voice and synth.voice.name and synth.voice.name in self.voices:
                voice_name = synth.voice.name

            # 2. 确定语言：
            # 策略一：优先读取协议显式指定的语言
            lang = None
            if synth.voice and synth.voice.language:
                lang = synth.voice.language.split("-")[0].lower()

            # 策略二：若未指定，依据选定的声音名称前缀推导语言
            if not lang and voice_name:
                prefix = voice_name[0].lower()
                lang = VOICE_PREFIX_TO_LANG.get(prefix)

            # 策略三：若仍未确定（或设为 auto），依据合成文本内容自动检测（假名->ja, 汉字->zh, 字母->en）
            if not lang or lang == "auto":
                lang = self.g2p_manager.detect_language_from_text(text)

            logger.info("TTS 合成: 语言=%s, 声音=%s, 文本=%s", lang, voice_name, text)

            # 发送音频流开始事件
            await self.write_event(
                AudioStart(
                    rate=self.sample_rate,
                    width=2,       # 16-bit PCM
                    channels=1,    # 单声道
                ).event()
            )

            # 执行多语言 OpenVINO 语音合成
            audio_bytes = await self.call_openvino(
                text=text,
                language=lang,
                voice_name=voice_name,
            )

            # 分块发送音频数据
            chunk_size = 1024
            for i in range(0, len(audio_bytes), chunk_size):
                await self.write_event(
                    AudioChunk(
                        audio=audio_bytes[i : i + chunk_size],
                        rate=self.sample_rate,
                        width=2,
                        channels=1,
                    ).event()
                )

            # 发送音频流结束事件
            await self.write_event(AudioStop().event())

        return True

    async def call_openvino(
        self,
        text: str,
        language: Optional[str] = None,
        voice_name: Optional[str] = None,
    ) -> bytes:
        """
        使用 OpenVINO Kokoro 模型进行多语言语音合成。
        """
        target_lang = language or self.default_language or "zh"
        g2p = self.g2p_manager.get_g2p(target_lang)

        # 文本 → 音素 token IDs
        tokens = process_text_with_misaki(g2p, text)

        # 获取声音风格向量
        v_name = voice_name or self.default_voice_name
        style = self.voices.get(v_name)
        if style is None:
            first_key = next(iter(self.voices))
            logger.warning("未找到声音 '%s'，自动回退至 '%s'", v_name, first_key)
            style = self.voices[first_key]

        if style.ndim == 1:
            style = style[np.newaxis, :]

        speed_array = np.array([self.speed], dtype=np.float32)

        # 在线程池中执行推理
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._run_inference,
            tokens,
            style,
            speed_array,
        )

        return result

    def _run_inference(
        self,
        tokens: np.ndarray,
        style: np.ndarray,
        speed: np.ndarray,
    ) -> bytes:
        """执行 OpenVINO 模型推理。"""
        output = self.compiled_model(
            {"tokens": tokens, "style": style, "speed": speed}
        )

        audio = output[self.compiled_model.output(0)]
        audio = np.squeeze(audio)

        # float32 音频转换为 16-bit PCM
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        return audio_int16.tobytes()


# ===========================================================================
# 命令行入口
# ===========================================================================
@app.command()
def main(
    model_path: str = typer.Option(
        ...,
        "--model-path",
        "-m",
        help="Kokoro OpenVINO 模型文件路径（.xml 或 .onnx）",
    ),
    voices_path: str = typer.Option(
        ...,
        "--voices-path",
        "-v",
        help="声音风格文件路径（.npy，字典格式 {name: vector}）",
    ),
    voice_name: Optional[str] = typer.Option(
        None,
        "--voice-name",
        help="默认声音名称（不指定则自动选用声音文件中的第一个声音；HA 发起请求时优先遵循 HA 选定的声音）",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="服务监听地址",
    ),
    port: int = typer.Option(
        10021,
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
        help="默认合成语言（默认为 auto；优先遵循 Wyoming 协议指定，其次按声音前缀或文本内容自动推测）",
    ),
    speed: float = typer.Option(
        1.0,
        "--speed",
        "-s",
        help="语速倍率（1.0 为正常速度，>1.0 加速，<1.0 减速）",
    ),
    sample_rate: int = typer.Option(
        24000,
        "--sample-rate",
        help="音频采样率（Hz），Kokoro 默认 24000",
    ),
    model_name: str = typer.Option(
        "kokoro",
        "--model-name",
        help="模型名称，用于 Wyoming 服务发现和 Home Assistant 显示",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="日志级别（DEBUG / INFO / WARNING / ERROR）",
    ),
) -> None:
    """启动基于 OpenVINO 的 Wyoming TTS 服务，支持多语言自动推测与协议动态选择。"""

    # 初始化日志系统
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("正在加载 TTS 模型: %s", model_path)
    logger.info("推理设备: %s", device)

    # ---- 初始化 OpenVINO 推理引擎 ----
    core = ov.Core()
    logger.info("可用推理设备: %s", core.available_devices)

    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device)
    logger.info("模型编译完毕 (设备: %s)", device)

    # ---- 加载声音风格文件 ----
    logger.info("加载声音风格文件: %s", voices_path)
    voices_raw = np.load(voices_path, allow_pickle=True)

    if isinstance(voices_raw, np.ndarray) and voices_raw.ndim == 0:
        voices = voices_raw.item()
    elif hasattr(voices_raw, "files"):
        voices = {name: voices_raw[name] for name in voices_raw.files}
    else:
        voices = voices_raw

    if not isinstance(voices, dict):
        raise typer.BadParameter("voices 文件内容必须为声音字典对象")

    all_voice_names = list(voices.keys())
    logger.info("已加载 %d 个声音", len(all_voice_names))

    # 确定默认声音名称
    selected_voice = voice_name
    if not selected_voice or selected_voice not in voices:
        selected_voice = all_voice_names[0]
        logger.info("未指定或未匹配到默认声音，自动选用首个声音: %s", selected_voice)

    # ---- 初始化多语言 G2P 管理器 ----
    default_lang = language or "auto"
    g2p_manager = MultiLangG2P(fallback_language="zh" if default_lang == "auto" else default_lang)

    # ---- 自动扫描所有声音并构建 Wyoming 服务信息 ----
    tts_voices: List[TtsVoice] = []
    for v in all_voice_names:
        prefix = v[0].lower() if v else ""
        v_lang = VOICE_PREFIX_TO_LANG.get(prefix, "zh")
        tts_voices.append(
            TtsVoice(
                name=v,
                description=f"{v} ({v_lang})",
                languages=[v_lang],
            )
        )

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="wyoming-openvino-tts",
                description=f"基于 OpenVINO 的 {model_name} 多语言语音合成服务 (中/英/日等)",
                voices=tts_voices,
                attribution=Attribution(
                    name="Kokoro TTS (OpenVINO)",
                    url="https://github.com/hexgrad/kokoro",
                ),
            )
        ],
    )

    # ---- 启动 Wyoming 异步服务 ----
    server = AsyncServer.from_uri(f"tcp://{host}:{port}")
    logger.info("TTS 服务启动: %s:%d (共注册 %d 个声音，支持自动语言识别与 HA 动态指定)", host, port, len(tts_voices))

    asyncio.run(
        server.run(
            partial(
                TTSEventHandler,
                wyoming_info=wyoming_info,
                compiled_model=compiled_model,
                g2p_manager=g2p_manager,
                voices=voices,
                default_voice_name=selected_voice,
                default_language=default_lang,
                speed=speed,
                sample_rate=sample_rate,
            )
        )
    )


if __name__ == "__main__":
    app()
