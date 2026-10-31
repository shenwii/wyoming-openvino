#!/usr/bin/env python3
"""
Wyoming TTS 服务
================
基于 OpenVINO 加速的 Kokoro 文字转语音服务，通过 Wyoming 协议对接 Home Assistant。

功能:
    - 使用 misaki 进行多语言文本到音素转换
    - 使用 OpenVINO 推理引擎运行 Kokoro TTS 模型
    - 支持 Intel GPU / CPU / 混合推理 (AUTO / MULTI)
    - 通过 Wyoming 协议与 Home Assistant 无缝集成

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
from functools import partial
from pathlib import Path

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
app = typer.Typer(help="Wyoming TTS 服务 - 基于 OpenVINO 的语音合成")


# ===========================================================================
# Kokoro 音素词汇表
# ===========================================================================
# 来源于 Kokoro / StyleTTS2 的文本处理模块。
# 用于将 misaki 输出的 IPA 音素字符逐字符映射为模型所需的 token ID。
# 索引 0 为填充符（PAD），模型输入的 token 序列首尾需用 0 包裹。
_PAD = "$"
_PUNCTUATION = ';:,.!?¡¿—…"«»"" '
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_LETTERS_IPA = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴ"
    "øɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃ"
    "ˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̈̃ˡ̩"
)
_SYMBOLS = [_PAD] + list(_PUNCTUATION) + list(_LETTERS) + list(_LETTERS_IPA)
VOCAB: dict[str, int] = {s: i for i, s in enumerate(_SYMBOLS)}


# ===========================================================================
# G2P（字素到音素）工具
# ===========================================================================
def create_g2p(language: str):
    """
    根据语言创建 misaki G2P（字素到音素）转换器。

    misaki 是 Kokoro TTS 的官方文本前端，负责将文本转为 IPA 音素。
    不同语言使用不同的 G2P 引擎。

    参数:
        language: 语言代码（zh / en / ja / ko / fr 等）

    返回:
        misaki G2P 实例

    异常:
        ValueError: 不支持的语言代码
        ImportError: 缺少 misaki 或对应语言模块
    """
    lang = language.split("-")[0].lower()

    try:
        if lang == "zh":
            from misaki.zh import G2P

            return G2P()
        elif lang == "en":
            from misaki.en import G2P

            # trf=False 使用基于规则的 G2P，无需额外的 transformer 模型
            return G2P(trf=False)
        elif lang == "ja":
            from misaki.ja import G2P

            return G2P()
        elif lang == "ko":
            from misaki.ko import G2P

            return G2P()
        elif lang == "fr":
            from misaki.fr import G2P

            return G2P()
        else:
            raise ValueError(
                f"不支持的语言: {language}。目前支持: zh, en, ja, ko, fr"
            )
    except ImportError as exc:
        raise ImportError(
            f"无法加载 '{language}' 的 G2P 模块。"
            f"请确认已安装 misaki 及对应语言包: pip install misaki"
        ) from exc


def process_text_with_misaki(g2p, text: str) -> np.ndarray:
    """
    使用 misaki 将文本转换为 Kokoro 模型所需的 token ID 序列。

    处理流程:
        1. 调用 G2P 将文本分段转为 IPA 音素字符串
        2. 将各段音素拼接为完整字符串
        3. 逐字符查表（VOCAB）转换为整数 token ID
        4. 首尾填充 PAD token（ID = 0）

    参数:
        g2p:  misaki G2P 转换器实例
        text: 待合成的文本

    返回:
        numpy int64 数组，shape 为 [1, seq_len]
    """
    phoneme_parts: list[str] = []

    # G2P 以生成器方式逐段输出 (字素, 音素) 元组
    for _graphemes, phonemes in g2p(text):
        if phonemes is not None:
            phoneme_parts.append(phonemes)

    phoneme_str = "".join(phoneme_parts)

    if not phoneme_str:
        logger.warning("文本 '%s' 未产生任何音素", text)
        return np.array([[0, 0]], dtype=np.int64)

    # 逐字符转换为 token ID，未知字符映射为 0（PAD）
    token_ids = [0] + [VOCAB.get(c, 0) for c in phoneme_str] + [0]

    return np.array([token_ids], dtype=np.int64)


# ===========================================================================
# Wyoming 事件处理器
# ===========================================================================
class TTSEventHandler(AsyncEventHandler):
    """
    Wyoming TTS 事件处理器。

    处理来自 Wyoming 客户端的语音合成请求:
        1. 收到 Describe 事件   → 返回服务信息
        2. 收到 Synthesize 事件 → 执行 TTS 推理并流式返回音频
    """

    def __init__(
        self,
        *args,
        wyoming_info: Info,
        compiled_model: ov.CompiledModel,
        g2p,
        voices: dict,
        voice_name: str,
        speed: float,
        sample_rate: int,
        **kwargs,
    ) -> None:
        """
        初始化 TTS 事件处理器。

        参数:
            wyoming_info:   Wyoming 服务描述信息
            compiled_model: 已编译的 OpenVINO 模型
            g2p:            misaki G2P 转换器
            voices:         声音风格向量字典 {name: np.ndarray}
            voice_name:     使用的声音名称
            speed:          语速倍率
            sample_rate:    音频采样率（Hz）
        """
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.compiled_model = compiled_model
        self.g2p = g2p
        self.voices = voices
        self.voice_name = voice_name
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
                logger.warning("收到空的合成文本，发送空音频")
                await self.write_event(
                    AudioStart(
                        rate=self.sample_rate, width=2, channels=1
                    ).event()
                )
                await self.write_event(AudioStop().event())
                return True

            logger.info("合成文本: %s", text)

            # 发送音频流开始事件（告知客户端音频格式）
            await self.write_event(
                AudioStart(
                    rate=self.sample_rate,
                    width=2,       # 16-bit PCM
                    channels=1,    # 单声道
                ).event()
            )

            # 调用 OpenVINO 进行语音合成
            audio_bytes = await self.call_openvino(text)

            # 分块发送音频数据（每块 1024 字节，避免单次传输过大）
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

    async def call_openvino(self, text: str) -> bytes:
        """
        使用 OpenVINO Kokoro 模型进行语音合成。

        处理流程:
            1. 使用 misaki 将文本转换为音素 token IDs
            2. 从 voices 字典获取声音风格向量
            3. 在线程池中执行 OpenVINO 推理
            4. 将 float32 音频输出转换为 16-bit PCM

        参数:
            text: 待合成的文本

        返回:
            16-bit PCM 格式的音频字节数据
        """
        # 文本 → 音素 token IDs
        tokens = process_text_with_misaki(self.g2p, text)

        # 获取声音风格向量
        style = self.voices[self.voice_name]
        if style.ndim == 1:
            style = style[np.newaxis, :]  # 确保 shape 为 [1, style_dim]

        speed_array = np.array([self.speed], dtype=np.float32)

        # 在线程池中执行推理，避免阻塞异步事件循环
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
        执行 OpenVINO 模型推理（同步方法，在线程池中调用）。

        模型输入:
            - tokens: 音素 token ID 数组，shape [1, seq_len]，dtype int64
            - style:  声音风格向量，shape [1, style_dim]，dtype float32
            - speed:  语速倍率，shape [1]，dtype float32

        模型输出:
            - 音频波形，float32，范围 [-1.0, 1.0]

        参数:
            tokens: 音素 token ID 数组
            style:  声音风格向量
            speed:  语速倍率

        返回:
            16-bit PCM 格式的音频字节数据
        """
        # 执行 OpenVINO 推理
        # 输入名称与 Kokoro ONNX/OpenVINO 模型保持一致
        output = self.compiled_model(
            {"tokens": tokens, "style": style, "speed": speed}
        )

        # 获取第一个输出张量（音频波形）
        audio = output[self.compiled_model.output(0)]
        audio = np.squeeze(audio)  # 去除 batch 维度

        # float32 音频 → 16-bit PCM 整数
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
        help="使用的声音名称（需在 voices 文件中存在）",
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
        help="合成语言代码（zh / en / ja / ko / fr）",
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
    """启动基于 OpenVINO 的 Wyoming TTS（文字转语音）服务。"""

    # 初始化日志系统
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("正在加载 TTS 模型: %s", model_path)
    logger.info("推理设备: %s", device)

    # ---- 初始化 OpenVINO 推理引擎 ----
    core = ov.Core()

    # 显示可用设备，方便调试
    available_devices = core.available_devices
    logger.info("可用推理设备: %s", available_devices)

    # 读取并编译模型到指定设备
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device)
    logger.info("模型编译完毕 (设备: %s)", device)

    # 显示模型输入输出信息，方便调试
    for inp in compiled_model.inputs:
        logger.debug("模型输入: %s, shape=%s, dtype=%s", inp.get_any_name(), inp.shape, inp.element_type)
    for out in compiled_model.outputs:
        logger.debug("模型输出: %s, shape=%s, dtype=%s", out.get_any_name(), out.shape, out.element_type)

    # ---- 加载声音风格文件 ----
    logger.info("加载声音风格文件: %s", voices_path)
    voices_raw = np.load(voices_path, allow_pickle=True)

    # 支持两种格式:
    #   1. np.save(path, dict) → 加载后为 0-d array，需 .item() 解包
    #   2. np.savez(path, **dict) → 加载后为 NpzFile
    if isinstance(voices_raw, np.ndarray) and voices_raw.ndim == 0:
        voices = voices_raw.item()
    elif hasattr(voices_raw, "files"):
        # NpzFile 格式
        voices = {name: voices_raw[name] for name in voices_raw.files}
    else:
        voices = voices_raw

    if isinstance(voices, dict):
        logger.info("可用声音: %s", list(voices.keys()))
    else:
        logger.warning("声音文件格式非字典，将使用索引访问")

    if isinstance(voices, dict) and voice_name not in voices:
        available = list(voices.keys())
        raise typer.BadParameter(
            f"声音 '{voice_name}' 不存在。可用声音: {available}"
        )

    logger.info("使用声音: %s", voice_name)

    # ---- 初始化 G2P 转换器 ----
    logger.info("初始化 G2P 转换器（语言: %s）...", language)
    g2p = create_g2p(language)
    logger.info("G2P 初始化完毕")

    # ---- 构建 Wyoming 服务信息 ----
    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="wyoming-openvino-tts",
                description=f"基于 OpenVINO 的 {model_name} 语音合成服务",
                voices=[
                    TtsVoice(
                        name=voice_name,
                        description=voice_name,
                        languages=[language],
                    )
                ],
                attribution=Attribution(
                    name="Kokoro TTS (OpenVINO)",
                    url="https://github.com/hexgrad/kokoro",
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
                TTSEventHandler,
                wyoming_info=wyoming_info,
                compiled_model=compiled_model,
                g2p=g2p,
                voices=voices,
                voice_name=voice_name,
                speed=speed,
                sample_rate=sample_rate,
            )
        )
    )


if __name__ == "__main__":
    app()
