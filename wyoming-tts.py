#!/usr/bin/env python3
"""
Wyoming TTS 服务
================
基于 OpenVINO 加速的 Kokoro 文字转语音服务，通过 Wyoming 协议对接 Home Assistant。

多语言支持:
    - 完整支持中文 (zh)、英语 (en)、日语 (ja)
    - 额外扩展支持西班牙语 (es)、法语 (fr)、印地语 (hi)、意大利语 (it)、巴西葡萄牙语 (pt)
    - 支持智能语言推断：优先按请求语言/声音前缀推断，亦可按文本字符（平假名/汉字/拉丁字母）自动匹配 G2P
    - 自动扫描声音字典（voices.npy），向 Home Assistant 宣告所有中/英/日可用声音及对应语种
    - 支持 Intel GPU / CPU / 混合推理 (AUTO / MULTI)

使用示例:
    python wyoming-tts.py \\
        --model-path ./kokoro.xml \\
        --voices-path ./voices.npy \\
        --voice-name af_heart \\
        --device GPU \\
        --language zh
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

# Kokoro 声音名称前缀与语言的映射关系
# 例: af_heart -> en, zf_xiaobei -> zh, jf_teio -> ja
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
    "k": "ko",  # Korean (若安装对应库)
}


# ===========================================================================
# 多语言 G2P（字素到音素）转换管理器
# ===========================================================================
class MultiLangG2P:
    """
    多语言 G2P 转换管理器。
    按需延迟加载各语言的 G2P 转换器，支持在同一服务实例内同时服务中、英、日等多语言请求。
    """

    def __init__(self, default_language: str = "zh") -> None:
        self.default_language = default_language.split("-")[0].lower()
        self._g2p_instances: Dict[str, object] = {}

    def get_g2p(self, language: str):
        """获取指定语言的 G2P 实例（按需加载并缓存）。"""
        lang = language.split("-")[0].lower()

        if lang in self._g2p_instances:
            return self._g2p_instances[lang]

        g2p = self._create_g2p(lang)
        self._g2p_instances[lang] = g2p
        return g2p

    def _create_g2p(self, lang: str):
        """创建指定语言的 misaki G2P 引擎。"""
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
                logger.warning("未找到专门的 '%s' G2P 引擎，回退到英文/默认 G2P", lang)
                from misaki.en import G2P

                return G2P(trf=False)

        except ImportError as exc:
            logger.error(
                "加载 '%s' 的 G2P 引擎失败，请确保安装了相应依赖: %s",
                lang,
                exc,
            )
            # 如果加载失败，尝试回退到英文规则转换
            try:
                from misaki.en import G2P

                logger.warning("由于模块缺失，回退到 misaki.en 基础 G2P")
                return G2P(trf=False)
            except Exception:
                raise ImportError(
                    f"无法初始化 misaki G2P 引擎，请确认已安装 misaki: pip install misaki"
                ) from exc

    def detect_language_from_text(self, text: str) -> str:
        """
        根据文本特征自动检测语言:
            - 包含日文平假名/片假名 → 日语 (ja)
            - 包含中文字符且无假名 → 中文 (zh)
            - 其它（主要为拉丁字母） → 英语 (en)
        """
        # 日文平假名/片假名范围
        if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", text):
            return "ja"

        # 中文字符范围
        if re.search(r"[\u4E00-\u9FFF]", text):
            return "zh"

        # 默认推断为英文或默认语言
        return self.default_language or "en"


def process_text_with_misaki(g2p, text: str) -> np.ndarray:
    """
    使用 misaki 将文本转换为 Kokoro 模型所需的 token ID 序列。

    参数:
        g2p:  misaki G2P 转换器实例
        text: 待合成文本

    返回:
        numpy int64 数组，shape 为 [1, seq_len]
    """
    phoneme_parts: List[str] = []

    for _graphemes, phonemes in g2p(text):
        if phonemes is not None:
            phoneme_parts.append(phonemes)

    phoneme_str = "".join(phoneme_parts)

    if not phoneme_str:
        logger.warning("文本 '%s' 未能提取出有效音素", text)
        return np.array([[0, 0]], dtype=np.int64)

    # 逐字符查表转换为 token ID，未知字符映射为 0 (PAD)
    token_ids = [0] + [VOCAB.get(c, 0) for c in phoneme_str] + [0]
    return np.array([token_ids], dtype=np.int64)


# ===========================================================================
# Wyoming 事件处理器
# ===========================================================================
class TTSEventHandler(AsyncEventHandler):
    """
    Wyoming TTS 事件处理器。

    处理来自 Wyoming 客户端的语音合成请求:
        1. 收到 Describe 事件   → 返回所有注册的声音和语言
        2. 收到 Synthesize 事件 → 动态匹配语言与声音，执行推理并流式返回音频
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
        """
        初始化 TTS 事件处理器。
        """
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

            # 确定本次请求的声音
            voice_name = self.default_voice_name
            if synth.voice and synth.voice.name and synth.voice.name in self.voices:
                voice_name = synth.voice.name

            # 确定本次请求的语言（多语言智能路由）
            # 1. 优先使用请求显式指定的语言
            lang = None
            if synth.voice and synth.voice.language:
                lang = synth.voice.language.split("-")[0].lower()

            # 2. 其次根据声音名称的前缀判断（例如 zf_* 为中文，af_* 为英文，jf_* 为日文）
            if not lang and voice_name:
                prefix = voice_name[0].lower()
                lang = VOICE_PREFIX_TO_LANG.get(prefix)

            # 3. 再次根据文本字符内容自动侦测（平假名/片假名->ja, 汉字->zh, 其它->en）
            if not lang:
                lang = self.g2p_manager.detect_language_from_text(text)

            logger.info("合成请求: 语言=%s, 声音=%s, 文本=%s", lang, voice_name, text)

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

        处理流程:
            1. 取得对应语言的 G2P 引擎，将文本转换为 IPA token IDs
            2. 获取声音风格向量
            3. 在线程池中执行 OpenVINO 推理
            4. 将 float32 音频输出转换为 16-bit PCM

        参数:
            text:       待合成文本
            language:   合成语言（zh / en / ja 等）
            voice_name: 声音名称

        返回:
            16-bit PCM 格式音频字节
        """
        target_lang = language or self.default_language
        g2p = self.g2p_manager.get_g2p(target_lang)

        # 文本 → 音素 token IDs
        tokens = process_text_with_misaki(g2p, text)

        # 获取声音风格向量
        v_name = voice_name or self.default_voice_name
        style = self.voices.get(v_name)
        if style is None:
            # 回退到第一个可用声音
            first_key = next(iter(self.voices))
            logger.warning("未找到声音 '%s'，回退至 '%s'", v_name, first_key)
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
        """
        执行 OpenVINO 模型推理（在线程池中调用）。
        """
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
    voice_name: str = typer.Option(
        "af_heart",
        "--voice-name",
        help="默认声音名称",
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
    language: str = typer.Option(
        "zh",
        "--language",
        "-l",
        help="默认合成语言代码（zh / en / ja / es / fr / hi / it / pt）",
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
    """启动基于 OpenVINO 的 Wyoming TTS（文字转语音）服务，完整支持中/英/日等全球多语言。"""

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
    logger.info("已加载 %d 个声音，列表: %s", len(all_voice_names), all_voice_names[:10])

    # 若指定默认声音不在字典中，则自动选取首个声音
    if voice_name not in voices:
        logger.warning("指定默认声音 '%s' 不存在，自动使用 '%s'", voice_name, all_voice_names[0])
        voice_name = all_voice_names[0]

    # ---- 初始化多语言 G2P 管理器 ----
    g2p_manager = MultiLangG2P(default_language=language)
    # 预加载默认语言的 G2P 引擎
    g2p_manager.get_g2p(language)

    # ---- 自动扫描所有声音并构建 Wyoming 服务信息 ----
    # 为 Home Assistant 生成完整的多语言声音列表
    tts_voices: List[TtsVoice] = []
    for v in all_voice_names:
        prefix = v[0].lower() if v else ""
        v_lang = VOICE_PREFIX_TO_LANG.get(prefix, language)
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
    logger.info("TTS 服务启动: %s:%d (共注册 %d 个声音)", host, port, len(tts_voices))

    asyncio.run(
        server.run(
            partial(
                TTSEventHandler,
                wyoming_info=wyoming_info,
                compiled_model=compiled_model,
                g2p_manager=g2p_manager,
                voices=voices,
                default_voice_name=voice_name,
                default_language=language,
                speed=speed,
                sample_rate=sample_rate,
            )
        )
    )


if __name__ == "__main__":
    app()
