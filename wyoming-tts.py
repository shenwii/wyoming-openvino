#!/usr/bin/env python3
"""
Wyoming TTS 服务
================
基于 OpenVINO 加速的 Kokoro 文字转语音服务，通过 Wyoming 协议对接 Home Assistant。

模型支持:
    - 原生支持官方 OpenVINO IR 模型目录（含 openvino_model.xml, openvino_model.bin 以及 voices/ 目录）
    - 兼容支持 .onnx 模型及 voices.npy 声音打包文件
    - 自动扫描 voices/ 目录下的所有 *.bin 声音文件，并在 Home Assistant 中全量注册

多语言机制:
    - 语言优先采用 Wyoming 协议在请求中指定的语言（如 HA 配置的 voice.language）
    - 若协议未指定语言，自动依据选用的发音人前缀推断（zf_* -> zh, jf_* -> ja, af_* -> en 等）
    - 若仍未指定，根据待合成的文本内容自动推测（日文假名 -> ja, 汉字 -> zh, 字母 -> en）
    - 命令行只需指定 --model-dir 即可自动就绪！

使用示例:
    # 极简启动：直接传入下载的 Kokoro 官方 OpenVINO 目录
    python wyoming-tts.py --model-dir ./Kokoro-82M-OpenVINO --device GPU
"""

import asyncio
import logging
import re
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
app = typer.Typer(help="Wyoming TTS 服务 - 基于 OpenVINO 的多语言语音合成")


# ===========================================================================
# Kokoro 音素词汇表
# ===========================================================================
_PAD = "$"
_PUNCTUATION = ';:,.!?¡¿—…"«»"" '
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_LETTERS_IPA = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴ"
    "øɵɸθœɶʘɹɺɾɻʀʁɽʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃ"
    "ˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̈̃ˡ̩"
)
_SYMBOLS = [_PAD] + list(_PUNCTUATION) + list(_LETTERS) + list(_LETTERS_IPA)
VOCAB: Dict[str, int] = {s: i for i, s in enumerate(_SYMBOLS)}

# Kokoro 声音名称前缀与语言代码的对应
VOICE_PREFIX_TO_LANG: Dict[str, str] = {
    "a": "en",  # American English (af_*, am_*)
    "b": "en",  # British English (bf_*, bm_*)
    "z": "zh",  # Chinese Mandarin (zf_*, zm_*)
    "j": "ja",  # Japanese (jf_*, jm_*)
    "e": "es",  # Spanish (ef_*, em_*)
    "f": "fr",  # French (ff_*)
    "h": "hi",  # Hindi (hf_*, hm_*)
    "i": "it",  # Italian (if_*, im_*)
    "p": "pt",  # Portuguese (pf_*, pm_*)
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
        if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", text):
            return "ja"
        if re.search(r"[\u4E00-\u9FFF]", text):
            return "zh"
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
# 模型与声音资源自动解析工具
# ===========================================================================
def resolve_model_and_voices(
    model_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    voices_path: Optional[str] = None,
) -> Tuple[Path, Dict[str, np.ndarray]]:
    """
    自动查找并解析 Kokoro 模型与发音人声音数据。

    支持两种形式:
      1. 官方 OpenVINO 目录:
         包含 openvino_model.xml, openvino_model.bin 以及 voices/*.bin
      2. 单文件形式:
         --model-path 指定 .xml 或 .onnx，--voices-path 指定 voices 目录或 .npy 文件
    """
    # 1. 确定模型路径
    target_model_file: Optional[Path] = None

    if model_path:
        p = Path(model_path)
        if p.is_dir():
            # 用户传的是目录
            candidates = [p / "openvino_model.xml", p / "model.xml"] + list(p.glob("*.xml")) + list(p.glob("*.onnx"))
            for cand in candidates:
                if cand.exists():
                    target_model_file = cand
                    break
        elif p.exists():
            target_model_file = p

    if not target_model_file and model_dir:
        d = Path(model_dir)
        candidates = [d / "openvino_model.xml", d / "model.xml"] + list(d.glob("*.xml")) + list(d.glob("*.onnx"))
        for cand in candidates:
            if cand.exists():
                target_model_file = cand
                break

    if not target_model_file or not target_model_file.exists():
        raise FileNotFoundError(
            f"未找到 Kokoro 模型文件（openvino_model.xml 或 *.onnx）。"
            f"请检查传入的路径: model_dir={model_dir}, model_path={model_path}"
        )

    logger.info("已选定模型文件: %s", target_model_file)

    # 2. 确定声音资源位置
    voices: Dict[str, np.ndarray] = {}
    base_dir = target_model_file.parent

    # 寻找声音来源（优先使用 voices_path，若未传则在模型同目录下查找）
    voice_sources: List[Path] = []
    if voices_path:
        voice_sources.append(Path(voices_path))
    if model_dir:
        voice_sources.append(Path(model_dir) / "voices")
        voice_sources.append(Path(model_dir) / "voices.npy")
    voice_sources.append(base_dir / "voices")
    voice_sources.append(base_dir / "voices.npy")

    for src in voice_sources:
        if not src.exists():
            continue

        # 场景 A: voices 目录（包含众多 *.bin 单个音色文件，如官方 OpenVINO 仓库）
        if src.is_dir():
            bin_files = list(src.glob("*.bin"))
            if bin_files:
                logger.info("在目录 %s 中找到 %d 个 .bin 声音文件", src, len(bin_files))
                for bf in bin_files:
                    v_name = bf.stem  # 如 'af_heart'
                    try:
                        # 官方 Kokoro .bin 文件为 float32 原始二进制向量
                        # shape 典型为 (-1, 1, 256) 或 (-1, 256)
                        data = np.fromfile(bf, dtype=np.float32)
                        if data.size % 256 == 0:
                            data = data.reshape(-1, 256)
                        voices[v_name] = data
                    except Exception as err:
                        logger.warning("加载声音文件 %s 失败: %s", bf, err)
                if voices:
                    break

        # 场景 B: voices.npy / .npz 单个打包文件
        elif src.is_file() and (src.suffix in [".npy", ".npz"] or src.name == "voices.bin"):
            try:
                voices_raw = np.load(src, allow_pickle=True)
                if isinstance(voices_raw, np.ndarray) and voices_raw.ndim == 0:
                    voices = voices_raw.item()
                elif hasattr(voices_raw, "files"):
                    voices = {name: voices_raw[name] for name in voices_raw.files}
                elif isinstance(voices_raw, dict):
                    voices = voices_raw
                if voices:
                    logger.info("从打包文件 %s 成功加载 %d 个声音", src, len(voices))
                    break
            except Exception as err:
                logger.warning("解析打包文件 %s 失败: %s", src, err)

    if not voices:
        raise FileNotFoundError(
            f"未找到可用的声音数据（未在 voices/ 目录下找到 *.bin 或 voices.npy 文件）。"
            f"请检查声音目录或指定 --voices-path"
        )

    return target_model_file, voices


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
        voices: Dict[str, np.ndarray],
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
            await self.write_event(self.wyoming_info.event())

        elif Synthesize.is_type(event.type):
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
            lang = None
            if synth.voice and synth.voice.language:
                lang = synth.voice.language.split("-")[0].lower()

            if not lang and voice_name:
                prefix = voice_name[0].lower()
                lang = VOICE_PREFIX_TO_LANG.get(prefix)

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
        voice_data = self.voices.get(v_name)
        if voice_data is None:
            first_key = next(iter(self.voices))
            logger.warning("未找到声音 '%s'，自动回退至 '%s'", v_name, first_key)
            voice_data = self.voices[first_key]

        # 提取当前 token 长度适配的 style 向量
        # Kokoro 的 voice embedding 文件通常包含按序列长度索引的向量
        token_len = tokens.shape[1]
        if voice_data.ndim == 2:
            # shape: (N, 256)
            idx = min(token_len, len(voice_data) - 1)
            style = voice_data[idx : idx + 1]  # shape: (1, 256)
        elif voice_data.ndim == 3:
            # shape: (N, 1, 256)
            idx = min(token_len, len(voice_data) - 1)
            style = voice_data[idx]            # shape: (1, 256)
        elif voice_data.ndim == 1:
            style = voice_data[np.newaxis, :]  # shape: (1, 256)
        else:
            style = voice_data

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
    model_dir: Optional[str] = typer.Option(
        None,
        "--model-dir",
        "-m",
        help="Kokoro OpenVINO 模型目录路径（自动定位 openvino_model.xml 及 voices/ 目录）",
    ),
    model_path: Optional[str] = typer.Option(
        None,
        "--model-path",
        help="指定 Kokoro 模型文件（.xml 或 .onnx，若已指定 --model-dir 可不填）",
    ),
    voices_path: Optional[str] = typer.Option(
        None,
        "--voices-path",
        "-v",
        help="指定声音目录或文件（若已指定 --model-dir 且包含 voices/ 目录可不填）",
    ),
    voice_name: Optional[str] = typer.Option(
        None,
        "--voice-name",
        help="默认声音名称（默认自动取声音文件列表中的首个；HA 发起请求时优先遵循 HA 选定的声音）",
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
    """启动基于 OpenVINO 的 Wyoming TTS 服务，原生支持官方 OpenVINO IR 格式与 voices/*.bin 声音。"""

    # 初始化日志系统
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not model_dir and not model_path:
        raise typer.BadParameter("必须通过 --model-dir 指定模型目录，或通过 --model-path 指定模型文件。")

    # ---- 自动解析模型与声音资源 ----
    resolved_model_file, voices = resolve_model_and_voices(
        model_dir=model_dir,
        model_path=model_path,
        voices_path=voices_path,
    )

    all_voice_names = list(voices.keys())
    selected_voice = voice_name
    if not selected_voice or selected_voice not in voices:
        selected_voice = all_voice_names[0]

    logger.info("已加载 %d 个声音，默认声音: %s", len(all_voice_names), selected_voice)

    # ---- 初始化 OpenVINO 推理引擎 ----
    core = ov.Core()
    logger.info("可用推理设备: %s", core.available_devices)
    logger.info("正在加载并编译模型: %s (设备: %s)", resolved_model_file, device)

    model = core.read_model(str(resolved_model_file))
    compiled_model = core.compile_model(model, device)
    logger.info("模型编译完毕")

    # ---- 初始化多语言 G2P 管理器 ----
    default_lang = language or "auto"
    g2p_manager = MultiLangG2P(fallback_language="zh" if default_lang == "auto" else default_lang)

    # ---- 自动扫描所有声音并构建 Wyoming 服务信息 ----
    tts_attribution = Attribution(
        name="Kokoro TTS (OpenVINO)",
        url="https://github.com/hexgrad/kokoro",
    )

    tts_voices: List[TtsVoice] = []
    for v in all_voice_names:
        prefix = v[0].lower() if v else ""
        v_lang = VOICE_PREFIX_TO_LANG.get(prefix, "zh")
        tts_voices.append(
            TtsVoice(
                name=v,
                description=f"{v} ({v_lang})",
                languages=[v_lang],
                attribution=tts_attribution,
                installed=True,
                version="1.0.0",
            )
        )

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="wyoming-openvino-tts",
                description=f"基于 OpenVINO 的 {model_name} 多语言语音合成服务 (中/英/日等)",
                installed=True,
                version="1.0.0",
                voices=tts_voices,
                attribution=tts_attribution,
            )
        ],
    )

    # ---- 启动 Wyoming 异步服务 ----
    server = AsyncServer.from_uri(f"tcp://{host}:{port}")
    logger.info(
        "TTS 服务启动: %s:%d (共注册 %d 个声音，支持自动语言识别与 HA 动态指定)",
        host,
        port,
        len(tts_voices),
    )

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
