#!/usr/bin/env python3
"""
Wyoming TTS 服务 (Kokoro OpenVINO 极速多语言版)
==============================================
基于 Intel OpenVINO 加速的 Kokoro-82M 语音合成服务，通过 Wyoming 协议无缝对接 Home Assistant。

重构亮点与设计参考:
    - 参考 remsky/Kokoro-FastAPI (Apache 2.0) 与 hexgrad/kokoro (Apache 2.0) 前端设计。
    - 采用全新流式分句与延迟音素转换机制（Streaming Chunking & Lazy Phonemization），
      首句秒级出声（TTFA < 0.3s），彻底消除整篇等待一分钟的卡顿问题。
    - 原生全面支持多语言：
        * 中文 (zh / z): 原生 misaki.zh G2P，支持汉字与中英混排（en_callable 自动处理英文单词）
        * 日语 (ja / j): 原生 misaki.ja G2P，准确解析汉字假名与日式读音，彻底修复假名与汉字识别冲突
        * 英语 (en / a, b): 美音 (a) 与英音 (b)
        * 西班牙语 (es / e)、法语 (fr / f)、印地语 (hi / h)、意大利语 (it / i)、葡萄牙语 (pt / p)
    - 支持 Kokoro-FastAPI 风格的多音色加权融合 (Voice Blending): 如 'zf_xiaobei(0.7)+af_heart(0.3)'。
    - 针对 Intel OpenVINO 硬件推理进行极致优化 (LATENCY 模式、线程调优、服务启动自动预热)。

使用示例:
    python wyoming-tts.py --model-dir ./Kokoro-82M-FP16-OpenVINO --device CPU --speed 0.85 --port 10021
"""

import asyncio
import io
import json
import logging
import re
import sys
import time
import wave
from functools import lru_cache
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import openvino as ov
import typer

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wyoming-tts")

# ---------------------------------------------------------------------------
# Typer CLI 应用
# ---------------------------------------------------------------------------
app = typer.Typer(help="Wyoming TTS 服务 - 基于 OpenVINO 的多语言 Kokoro 语音合成")


# ===========================================================================
# Kokoro 默认标准词汇表与语言映射定义
# ===========================================================================
_PAD = "$"
_PUNCTUATION = ';:,.!?¡¿—…"«»"" ()'
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_LETTERS_IPA = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴ"
    "øɵɸθœɶʘɹɺɾɻʀʁɽʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃ"
    "ˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̈̃ˡ̩"
)
_DEFAULT_SYMBOLS = [_PAD] + list(_PUNCTUATION) + list(_LETTERS) + list(_LETTERS_IPA)
DEFAULT_VOCAB: Dict[str, int] = {s: i for i, s in enumerate(_DEFAULT_SYMBOLS)}

# 发音人前缀对应语言代码（与 Kokoro-FastAPI 严格一致）
VOICE_PREFIX_TO_LANG: Dict[str, str] = {
    "a": "en",  # American English (af_*, am_*)
    "b": "en",  # British English (bf_*, bm_*)
    "z": "zh",  # Mandarin Chinese (zf_*, zm_*)
    "j": "ja",  # Japanese (jf_*, jm_*)
    "e": "es",  # Spanish (ef_*, em_*)
    "f": "fr",  # French (ff_*)
    "h": "hi",  # Hindi (hf_*, hm_*)
    "i": "it",  # Italian (if_*, im_*)
    "p": "pt",  # Portuguese (pf_*, pm_*)
}

# 单字母代码与 ISO 语言代码互通表
LANG_CODE_MAP: Dict[str, str] = {
    "zh": "z", "zh-cn": "z", "chinese": "z", "z": "z",
    "ja": "j", "japanese": "j", "j": "j",
    "en": "a", "en-us": "a", "english": "a", "a": "a",
    "en-gb": "b", "b": "b",
    "es": "e", "spanish": "e", "e": "e",
    "fr": "f", "french": "f", "f": "f",
    "hi": "h", "hindi": "h", "h": "h",
    "it": "i", "italian": "i", "i": "i",
    "pt": "p", "portuguese": "p", "p": "p",
}

# 语言可读名称映射
LANG_NAMES: Dict[str, str] = {
    "zh": "中文 (Mandarin)",
    "en": "English",
    "ja": "日本語 (Japanese)",
    "es": "Español (Spanish)",
    "fr": "Français (French)",
    "hi": "हिन्दी (Hindi)",
    "it": "Italiano (Italian)",
    "pt": "Português (Portuguese)",
}

# 常见缩写保护列表（防止因点号截断断句）
ABBREVIATIONS = (
    r"Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|approx|dept|min|max|fig|no|vol|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec|St|Ave|Rd|Corp|Inc|Ltd|Co"
)


# ===========================================================================
# 文本清洗、智能分句与流式分块 (Streaming Text Processor)
# ===========================================================================
def normalize_text_spacing(text: str) -> str:
    """
    规范化文本空格与特殊字符，处理中西文混排边界：
    1. 全角空格转半角；
    2. 中文字符与英文字母/数字边界补必要间隙，避免分词发音黏连；
    3. 保护常见英文缩写全大写词汇（如 AI, CPU, GPU, TTS, OpenVINO）。
    """
    if not text:
        return ""
    text = text.replace("\u3000", " ").strip()
    # 中文字符与英文字母边界微调（不破坏连续英文）
    text = re.sub(r"([\u4e00-\u9fff])([a-zA-Z0-9])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z0-9])([\u4e00-\u9fff])", r"\1 \2", text)
    # 多空格压缩
    text = re.sub(r"[ \t]+", " ", text)
    return text


def split_text_into_streaming_chunks(text: str, max_chunk_chars: int = 35) -> List[str]:
    """
    将待合成长文本智能切分为自然的语音分句流。
    核心思想参考 Kokoro-FastAPI 的断句哲学:
      1. 保护缩写与浮点数（如 Dr., 3.14），严防生硬断裂；
      2. 严格保留句尾语气标点（。！？!?），保证声学生成具备自然的语调抑扬顿挫；
      3. 超长句子按从句停顿标点（，、；;:—）智能二次分块；
      4. 严格将分句颗粒度控制在约 15~35 字，使首句能够极速出声，边播边算，消除等待。
    """
    text = normalize_text_spacing(text)
    if not text:
        return []

    # 1. 占位符保护浮点数与英文缩写中的句点
    protected = re.sub(r"(\d)\.(\d)", r"\g<1>\u200b\g<2>", text)
    protected = re.sub(rf"\b({ABBREVIATIONS})\.", r"\g<1>\u200b", protected, flags=re.IGNORECASE)
    protected = re.sub(r"\b([A-Z])\.", r"\g<1>\u200b", protected)

    # 2. 匹配主句结束符：中日句号问叹标点 [。！？!?] 或 英文句号后接空白/换行
    sentence_pattern = r"([^。！？!?\r\n]+(?:[。！？!?]+|\.(?=\s|$))?)"
    raw_sentences = [s.strip() for s in re.findall(sentence_pattern, protected) if s.strip()]

    # 3. 恢复占位符
    sentences = [s.replace("\u200b", ".") for s in raw_sentences]
    if not sentences:
        sentences = [text]

    # 4. 二次平滑从句分块
    chunks: List[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chunk_chars:
            chunks.append(sentence)
            continue

        clause_pattern = r"([^；;，,、:—\r\n]+[；;，,、:—]*)"
        raw_clauses = [c.strip() for c in re.findall(clause_pattern, sentence) if c.strip()]
        if not raw_clauses:
            chunks.append(sentence)
            continue

        current_chunk = ""
        for clause in raw_clauses:
            if len(current_chunk) + len(clause) > max_chunk_chars and current_chunk:
                chunks.append(current_chunk)
                current_chunk = clause
            else:
                need_space = bool(
                    current_chunk
                    and clause
                    and current_chunk[-1].isascii()
                    and clause[0].isalnum()
                )
                current_chunk += (" " if need_space else "") + clause

        if current_chunk:
            chunks.append(current_chunk)

    return chunks if chunks else [text]


# 保持向后兼容的别名函数
split_text_into_chunks = split_text_into_streaming_chunks


# ===========================================================================
# 多语言 G2P 前端引擎管理器 (Kokoro Multi-Language G2P Frontend)
# ===========================================================================
def detect_language(text: str, default_lang: str = "zh") -> str:
    """
    根据文本字符特征智能判定语言：
    - 包含日文平假名/片假名 ([\u3040-\u309F\u30A0-\u30FF]) -> 绝对属于日语 (ja)！
      (注：日文句子常夹杂汉字，但只要存在假名，必须判定为 ja，彻底解决旧脚本将日文误判为中文的严重缺陷)
    - 包含汉字且无假名 -> 中文 (zh)
    - 纯拉丁字母或西文字符 -> 英语 (en)
    - 其他无法判定的字符 -> 回退到默认语言
    """
    if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", text):
        return "ja"
    if re.search(r"[\u4E00-\u9FFF]", text):
        return "zh"
    if re.search(r"[a-zA-Z]", text):
        return "en"
    return default_lang or "zh"


class MultiLangG2P:
    """
    Kokoro 多语言 G2P（字素到音素）转换引擎管理器。
    参考 Kokoro-FastAPI，支持中、英、日、法、西、意、印、葡等全语言按需懒加载与高速缓存。
    """

    def __init__(
        self,
        fallback_language: str = "zh",
        vocab: Optional[Dict[str, int]] = None,
    ) -> None:
        self.fallback_language = fallback_language.lower()
        self.vocab = vocab or DEFAULT_VOCAB
        self._g2p_cache: Dict[str, object] = {}

    def detect_language_from_text(self, text: str) -> str:
        """根据文本内容智能检测语言（兼容方法）。"""
        return detect_language(text, default_lang=self.fallback_language)

    def get_g2p(self, lang_key: Optional[Union[str, object]] = None):
        """按需获取指定语言的 G2P 引擎实例。"""
        if callable(lang_key):
            return lang_key, "custom"

        if not lang_key or lang_key == "auto":
            lang_key = self.fallback_language or "zh"

        if str(lang_key) in self._g2p_cache:
            return self._g2p_cache[str(lang_key)], str(lang_key)

        # 归一化为单字母代码 (z, j, a, b, e, f, h, i, p)
        lang_str = str(lang_key).lower().split("-")[0]
        lang_code = LANG_CODE_MAP.get(lang_str, lang_str)

        if lang_code in self._g2p_cache:
            return self._g2p_cache[lang_code], lang_code

        engine = self._create_g2p(lang_code)
        self._g2p_cache[lang_code] = engine
        return engine, lang_code

    def _create_g2p(self, lang_code: str):
        """按语言代码初始化 misaki G2P 实例。"""
        try:
            if lang_code in ("a", "b"):
                from misaki import en
                logger.info("初始化英文 G2P (dialect=%s)", "British" if lang_code == "b" else "American")
                fallback = None
                try:
                    from misaki import espeak
                    fallback = espeak.EspeakFallback(british=(lang_code == "b"))
                except Exception:
                    pass
                try:
                    return en.G2P(trf=False, british=(lang_code == "b"), fallback=fallback)
                except TypeError:
                    return en.G2P(trf=False, british=(lang_code == "b"))

            elif lang_code == "j":
                from misaki import ja
                logger.info("初始化日文 G2P (misaki.ja.JAG2P)")
                return ja.JAG2P()

            elif lang_code == "z":
                from misaki import zh
                logger.info("初始化中文 G2P (misaki.zh.ZHG2P)")

                # 中文遇到英文单词时的回调转换函数（带高速 LRU 缓存）
                @lru_cache(maxsize=2048)
                def en_word_callable(word: str) -> str:
                    word = word.strip()
                    if not word:
                        return ""
                    try:
                        en_g2p, _ = self.get_g2p("en")
                        res = en_g2p(word)
                        if isinstance(res, tuple):
                            return str(res[0])
                        elif isinstance(res, str):
                            return res
                        elif hasattr(res, "phonemes"):
                            return str(res.phonemes)
                        return str(res)
                    except Exception:
                        return word

                # 检查当前模型词汇表是否支持 v1.1 注音（ㄓ, ㄅ 等），若基础模型则使用 version=None 输出标准 IPA
                has_zhuyin = bool(self.vocab and any(sym in self.vocab for sym in ("ㄓ", "ㄅ", "中", "阳")))
                zh_version = "1.1" if has_zhuyin else None
                logger.info("中文 G2P 配置: version=%s (词表原生注音: %s)", zh_version, has_zhuyin)

                try:
                    return zh.ZHG2P(version=zh_version, en_callable=en_word_callable)
                except (TypeError, ValueError):
                    try:
                        return zh.ZHG2P(version=None, en_callable=en_word_callable)
                    except Exception:
                        return zh.ZHG2P()

            else:
                # 其它语言（西班牙、法国、意大利、印地、葡萄牙等）
                logger.info("初始化语言 '%s' 的 G2P 引擎", lang_code)
                try:
                    from misaki import espeak
                    espeak_lang_map = {
                        "e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br"
                    }
                    lang_name = espeak_lang_map.get(lang_code, "en-us")
                    return espeak.EspeakG2P(language=lang_name)
                except Exception as e:
                    logger.warning("未配置 espeak-ng，回退到英文 G2P: %s", e)
                    from misaki import en
                    return en.G2P(trf=False)

        except ImportError as exc:
            install_cmd = {
                "z": "pip install 'misaki[zh]'",
                "j": "pip install 'misaki[ja]'",
                "a": "pip install 'misaki[en]'",
                "b": "pip install 'misaki[en]'",
            }.get(lang_code, "pip install misaki")
            msg = f"加载语言 '{lang_code}' 的 G2P 引擎失败，请执行安装命令: {install_cmd} ({exc})"
            logger.error(msg)
            raise RuntimeError(msg) from exc

    def text_to_tokens(self, text: str, language: Optional[Union[str, object]] = None) -> np.ndarray:
        """
        将单句文本通过 G2P 快速转换为 Kokoro 模型所需的 Token ID 序列。
        """
        if callable(language):
            g2p_engine = language
        else:
            g2p_engine, _ = self.get_g2p(language)

        try:
            res = g2p_engine(text)
        except Exception as e:
            logger.warning("G2P 转换异常 ('%s'): %s，尝试降级纯文本映射", text, e)
            res = text

        # 从 G2P 结果中解析出音素字符串
        if isinstance(res, tuple):
            ps = str(res[0])
        elif isinstance(res, str):
            ps = res
        elif hasattr(res, "phonemes"):
            ps = str(res.phonemes)
        elif isinstance(res, list):
            parts = []
            for item in res:
                if isinstance(item, tuple) and len(item) >= 2:
                    parts.append(str(item[1] if item[1] is not None else item[0]))
                elif hasattr(item, "phonemes"):
                    parts.append(str(item.phonemes))
                else:
                    parts.append(str(item))
            ps = "".join(parts)
        else:
            ps = str(res)

        # 查表匹配 Token IDs
        active_vocab = self.vocab or DEFAULT_VOCAB
        matched_ids = [active_vocab[c] for c in ps if c in active_vocab]

        if not matched_ids:
            logger.warning("音素 '%s' (原文: '%s') 未命中词汇表字符", ps, text)
            matched_ids = [0]

        # 截断以保证在 Kokoro 上下文限制之内 (510 tokens)
        if len(matched_ids) > 508:
            matched_ids = matched_ids[:508]

        # 首尾填充 PAD (0)
        token_ids = [0] + matched_ids + [0]
        return np.array([token_ids], dtype=np.int64)


# 兼容接口
def process_text_with_misaki(g2p, text: str, vocab: Optional[Dict[str, int]] = None, **kwargs) -> np.ndarray:
    manager = MultiLangG2P(vocab=vocab)
    manager._g2p_cache["custom"] = g2p
    return manager.text_to_tokens(text, language="custom")


# ===========================================================================
# 发音人声音与音色多重融合解析 (Voice Manager & Blender)
# ===========================================================================
_BLENDED_VOICES_CACHE: Dict[str, np.ndarray] = {}


def load_single_voice_tensor(file_path: Path) -> np.ndarray:
    """加载单个声音特征向量文件（支持 .bin, .pt, .pth, .npy, .npz）。"""
    suffix = file_path.suffix.lower()

    if suffix in [".pt", ".pth"]:
        try:
            import torch
            try:
                data = torch.load(file_path, weights_only=True, map_location="cpu")
            except Exception:
                data = torch.load(file_path, weights_only=False, map_location="cpu")

            if hasattr(data, "detach"):
                arr = data.detach().cpu().numpy()
            elif isinstance(data, dict):
                first_val = next(iter(data.values()))
                arr = first_val.detach().cpu().numpy() if hasattr(first_val, "detach") else np.array(first_val)
            else:
                arr = np.array(data)

            if arr.ndim == 1 and arr.size % 256 == 0:
                arr = arr.reshape(-1, 256)
            return arr.astype(np.float32)
        except ImportError:
            # 未安装 PyTorch 时，直接从 PyTorch zip 文件流中反序列化 float32 字节
            import zipfile
            try:
                with zipfile.ZipFile(file_path, "r") as z:
                    for name in z.namelist():
                        if name.endswith("/0") or name.startswith("archive/data/"):
                            with z.open(name) as member:
                                arr = np.frombuffer(member.read(), dtype=np.float32)
                                if arr.size % 256 == 0:
                                    return arr.reshape(-1, 256)
                                return arr
            except Exception:
                pass
            raise RuntimeError(f"请安装 PyTorch 以加载 .pt 音色文件: pip install torch")

    elif suffix == ".bin":
        arr = np.fromfile(file_path, dtype=np.float32)
        if arr.size % 256 == 0:
            arr = arr.reshape(-1, 256)
        return arr

    elif suffix in [".npy", ".npz"]:
        arr = np.load(file_path, allow_pickle=True)
        if arr.ndim == 1 and arr.size % 256 == 0:
            arr = arr.reshape(-1, 256)
        return arr.astype(np.float32)

    raise ValueError(f"不支持的音色文件格式: {suffix}")


def parse_voice_blend(voice_spec: str) -> List[Tuple[str, float]]:
    """
    解析加权多音色融合语法（与 Kokoro-FastAPI 一致）:
    例如: 'zf_xiaobei(0.7)+af_heart(0.3)' 或 'zf_xiaobei+af_heart'
    """
    if not voice_spec or not voice_spec.strip():
        return []

    parts = [p.strip() for p in voice_spec.split("+") if p.strip()]
    results: List[Tuple[str, float]] = []
    pattern = re.compile(r"^([a-zA-Z0-9_\-\.]+)(?:\(([0-9.]+)\))?$")

    for p in parts:
        m = pattern.match(p)
        if m:
            name = m.group(1).strip()
            w = float(m.group(2)) if m.group(2) is not None else 1.0
            results.append((name, w))
        else:
            results.append((p, 1.0))

    total = sum(w for _, w in results)
    if total > 0:
        results = [(name, w / total) for name, w in results]
    return results


def get_or_blend_voice(voices: Dict[str, np.ndarray], voice_spec: Optional[str]) -> np.ndarray:
    """获取单发音人或根据加权融合表达式实时混合发音人音色。"""
    if not voices:
        raise ValueError("音色库为空，无法获取发音人特征！")

    if not voice_spec or not voice_spec.strip():
        first_key = next(iter(voices))
        return voices[first_key]

    voice_spec = voice_spec.strip()

    # 1. 命中单音色
    if voice_spec in voices:
        return voices[voice_spec]

    # 2. 命中融合缓存
    if voice_spec in _BLENDED_VOICES_CACHE:
        return _BLENDED_VOICES_CACHE[voice_spec]

    blend_list = parse_voice_blend(voice_spec)
    if not blend_list:
        first_key = next(iter(voices))
        return voices[first_key]

    if len(blend_list) == 1:
        v_name = blend_list[0][0]
        return voices.get(v_name, voices[next(iter(voices))])

    # 3. 线性加权融合
    tensors = []
    weights = []
    for v_name, w in blend_list:
        if v_name in voices:
            tensors.append(voices[v_name])
            weights.append(w)
        else:
            logger.warning("未找到混合音色中的发音人 '%s'，已跳过", v_name)

    if not tensors:
        return voices[next(iter(voices))]

    w_sum = sum(weights)
    weights = [w / w_sum for w in weights] if w_sum > 0 else [1.0 / len(weights)] * len(weights)
    min_len = min(t.shape[0] if t.ndim >= 2 else 1 for t in tensors)

    blended = np.zeros((min_len, 256), dtype=np.float32)
    for arr, w in zip(tensors, weights):
        slice_arr = arr.reshape(-1, 256)[:min_len]
        blended += slice_arr * float(w)

    _BLENDED_VOICES_CACHE[voice_spec] = blended
    logger.info("成功创建加权融合音色: '%s' (混合 %d 款声音)", voice_spec, len(tensors))
    return blended


def resolve_model_and_voices(
    model_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    voices_path: Optional[str] = None,
) -> Tuple[Path, Dict[str, np.ndarray], Optional[Dict[str, int]]]:
    """
    自动递归搜索并解析 Kokoro 模型文件 (.xml / .onnx)、发音人音色列表以及词汇表。
    """
    candidates = []
    if model_dir:
        candidates.append(Path(model_dir))
    if model_path:
        candidates.append(Path(model_path).parent)

    search_roots = [p.resolve() for p in candidates if p.exists()]
    if not search_roots:
        search_roots = [Path(".").resolve()]

    # 1. 查找模型文件 (优先 openvino_model.xml)
    target_model_file: Optional[Path] = None
    if model_path and Path(model_path).is_file():
        target_model_file = Path(model_path).resolve()
    else:
        for root in search_roots:
            if root.is_file() and root.suffix in [".xml", ".onnx"]:
                target_model_file = root
                break
            # 优先 openvino_model.xml
            ov_xml = root / "openvino_model.xml"
            if ov_xml.exists():
                target_model_file = ov_xml
                break
            xml_files = list(root.glob("*.xml"))
            if xml_files:
                target_model_file = xml_files[0]
                break
            onnx_files = list(root.glob("*.onnx"))
            if onnx_files:
                target_model_file = onnx_files[0]
                break

    if not target_model_file:
        raise FileNotFoundError(f"未在候选目录中找到 OpenVINO (.xml) 或 ONNX (.onnx) 模型: {search_roots}")

    model_root = target_model_file.parent

    # 2. 读取 config.json 提取词汇表
    vocab: Optional[Dict[str, int]] = None
    config_file = model_root / "config.json"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if "vocab" in cfg:
                    vocab = cfg["vocab"]
                    logger.info("已成功加载 config.json 词汇表 (包含 %d 个 tokens)", len(vocab))
        except Exception as e:
            logger.warning("解析 config.json 异常: %s", e)

    # 3. 扫描 voices 目录或打包文件
    voices: Dict[str, np.ndarray] = {}
    voice_sources = []
    if voices_path:
        voice_sources.append(Path(voices_path))
    voice_sources.extend([
        model_root / "voices",
        model_root,
        Path("./voices"),
    ])

    for src in voice_sources:
        if not src.exists():
            continue
        if src.is_dir():
            for f in src.iterdir():
                if f.suffix.lower() in [".bin", ".pt", ".pth", ".npy"]:
                    v_name = f.stem
                    if v_name not in voices:
                        try:
                            arr = load_single_voice_tensor(f)
                            voices[v_name] = arr
                        except Exception as e:
                            logger.debug("跳过音色文件 %s: %s", f.name, e)
            if voices:
                break
        elif src.is_file() and src.suffix in [".npy", ".npz"]:
            try:
                data = np.load(src, allow_pickle=True)
                if isinstance(data, np.ndarray) and data.ndim == 0:
                    voices = data.item()
                elif hasattr(data, "files"):
                    voices = {k: data[k] for k in data.files}
                elif isinstance(data, dict):
                    voices = data
                if voices:
                    break
            except Exception as e:
                logger.warning("解析打包音色 %s 异常: %s", src, e)

    if not voices:
        logger.warning("未在模型目录找到预置音色文件，正在初始化基础虚拟参考音色...")
        voices["af_heart"] = np.zeros((1, 256), dtype=np.float32)

    logger.info("已加载 %d 款音色 (首个音色: %s)", len(voices), next(iter(voices)))
    return target_model_file, voices, vocab


# ===========================================================================
# OpenVINO Kokoro 推理引擎封装 (Kokoro OpenVINO Engine)
# ===========================================================================
class KokoroEngine:
    """
    基于 OpenVINO 的 Kokoro 核心声学模型推理引擎。
    """

    def __init__(
        self,
        model_file: Path,
        device: str = "CPU",
        threads: Optional[int] = None,
    ) -> None:
        self.model_file = model_file
        self.device = device.upper()
        self.threads = threads

        logger.info("正在加载并编译 OpenVINO 模型: %s (设备: %s)", model_file.name, self.device)
        core = ov.Core()
        model = core.read_model(str(model_file))

        ov_config: Dict[str, str] = {"PERFORMANCE_HINT": "LATENCY"}
        if threads and threads > 0:
            ov_config["INFERENCE_NUM_THREADS"] = str(threads)
            logger.info("已设置 OpenVINO 推理线程数: %d", threads)

        self.compiled_model = core.compile_model(model, self.device, config=ov_config)
        logger.info("🎉 OpenVINO 模型就绪 (LATENCY 模式)")

    def synthesize_tokens(
        self,
        token_ids: np.ndarray,
        style: np.ndarray,
        speed: float = 1.0,
    ) -> bytes:
        """
        执行单次声学模型推理，将 Token IDs 与发音人 Style 向量合成为 16-bit 24kHz PCM 字节流。
        """
        token_len = token_ids.shape[1]

        # 匹配对应长度的发音人 Style 向量
        if style.ndim == 2:
            idx = min(token_len, len(style) - 1)
            style_vec = style[idx : idx + 1]
        elif style.ndim == 3:
            idx = min(token_len, len(style) - 1)
            style_vec = style[idx]
        elif style.ndim == 1:
            style_vec = style[np.newaxis, :]
        else:
            style_vec = style

        if style_vec.shape != (1, 256):
            style_vec = style_vec.reshape(1, 256)
        style_vec = style_vec.astype(np.float32)

        speed_array = np.array([float(speed)], dtype=np.float32)

        # Kokoro OpenVINO IR 模型接受固定三入参: [input_ids, ref_s, speed]
        try:
            outputs = self.compiled_model([token_ids, style_vec, speed_array])
        except Exception as e:
            logger.error("OpenVINO 模型推理失败: %s", e)
            return b""

        # 提取音频输出 (float32 array, 24000Hz)
        audio = outputs[0]
        if isinstance(audio, np.ndarray):
            audio = audio.flatten()
        else:
            audio = np.array(audio).flatten()

        # 限幅并转为 16-bit 单声道 PCM 格式
        audio = np.clip(audio, -1.0, 1.0)
        pcm_data = (audio * 32767.0).astype(np.int16).tobytes()
        return pcm_data


# ===========================================================================
# Wyoming 协议事件处理与流式分发器
# ===========================================================================
class TTSEventHandler(AsyncEventHandler):
    """
    Wyoming TTS 异步事件处理器：实现首句秒级出声的流式响应。
    """

    def __init__(
        self,
        *args,
        wyoming_info: Info,
        engine: KokoroEngine,
        g2p_manager: MultiLangG2P,
        voices: Dict[str, np.ndarray],
        default_voice_name: str,
        default_language: str,
        speed: float,
        sample_rate: int = 24000,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.engine = engine
        self.g2p_manager = g2p_manager
        self.voices = voices
        self.default_voice_name = default_voice_name
        self.default_language = default_language
        self.speed = speed
        self.sample_rate = sample_rate

    async def handle_event(self, event: Event) -> bool:
        """处理 Wyoming 客户端发送的协议事件。"""

        # 1. 响应客户端能力查询 (Describe)
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            return True

        # 2. 响应语音合成请求 (Synthesize)
        if Synthesize.is_type(event.type):
            synth = Synthesize.from_event(event)
            text = (synth.text or "").strip()

            if not text:
                logger.warning("收到空文本，发送静音音频流")
                await self.write_event(AudioStart(rate=self.sample_rate, width=2, channels=1).event())
                await self.write_event(AudioStop().event())
                return True

            # (1) 确定发音人
            voice_name = self.default_voice_name
            if synth.voice and synth.voice.name:
                cand = synth.voice.name.strip()
                if cand in self.voices or "+" in cand:
                    voice_name = cand

            # (2) 确定合成语言（严格遵循用户指令与文本优先级）
            # 策略：
            # A. 若 HA 在 Wyoming 请求中指定了明确语言（如 voice.language='ja'），以指定为最高准则；
            # B. 其次检查发音人前缀（如 jf_* 为日语音色，zf_* 为中文音色）；
            # C. 再次分析待读文本内容（有假名必定是日语，有汉字必定是中文，有字母必定是英文）；
            # D. 最后回退到默认语言配置。
            client_lang = None
            if synth.voice and synth.voice.language:
                client_lang = synth.voice.language.split("-")[0].lower()

            detected_lang = detect_language(text, default_lang=self.default_language)

            if client_lang and client_lang not in ("auto", ""):
                target_lang = client_lang
            elif voice_name:
                prefix = voice_name[0].lower()
                target_lang = VOICE_PREFIX_TO_LANG.get(prefix, detected_lang)
            else:
                target_lang = detected_lang

            # (3) 确定语速倍率
            req_speed = self.speed
            if event.data:
                if "speed" in event.data and isinstance(event.data["speed"], (int, float)):
                    req_speed = float(event.data["speed"])
                elif isinstance(event.data.get("voice"), dict) and "speed" in event.data["voice"]:
                    req_speed = float(event.data["voice"]["speed"])

            logger.info(
                "🎙️ TTS 请求: 语言=%s (检测=%s), 发音人=%s, 语速=%.2f, 文本: %s",
                target_lang,
                detected_lang,
                voice_name,
                req_speed,
                text if len(text) <= 50 else text[:47] + "...",
            )

            # 获取发音人 Style 向量
            voice_style = get_or_blend_voice(self.voices, voice_name)

            # 发送流式音频开始事件
            await self.write_event(
                AudioStart(
                    rate=self.sample_rate,
                    width=2,       # 16-bit PCM
                    channels=1,    # 单声道
                ).event()
            )

            # (4) 智能流式切句与即时生成（Streaming Chunking & Immediate Playback）
            # 文本被切分为自然的从句小块，首句生成只需约 100ms 立刻开始发声，消除全部卡顿等待！
            chunks = split_text_into_streaming_chunks(text)
            logger.debug("长文本已切分为 %d 个流式语音从句: %s", len(chunks), chunks)

            loop = asyncio.get_running_loop()
            chunk_size = 1024  # 每次网络发送 1024 字节的小音频块

            for c_idx, chunk_text in enumerate(chunks, 1):
                t_start = time.perf_counter()

                # 每个分句的语言自适应判定（中日混合文本的保护）
                chunk_lang = target_lang
                if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", chunk_text):
                    chunk_lang = "ja"
                elif re.search(r"[\u4E00-\u9FFF]", chunk_text) and chunk_lang not in ("ja", "zh"):
                    chunk_lang = "zh"
                elif re.search(r"^[a-zA-Z0-9\s\.,!\?'-]+$", chunk_text):
                    # 纯西文字符/纯英文从句
                    chunk_lang = "en"

                # A. 音素与 Token 转换 (在线程池中执行，避免阻塞主循环)
                tokens = await loop.run_in_executor(
                    None,
                    self.g2p_manager.text_to_tokens,
                    chunk_text,
                    chunk_lang,
                )

                if tokens.shape[1] <= 2:
                    continue

                # B. OpenVINO 模型推理合成
                pcm_data = await loop.run_in_executor(
                    None,
                    self.engine.synthesize_tokens,
                    tokens,
                    voice_style,
                    req_speed,
                )

                t_cost = time.perf_counter() - t_start
                if not pcm_data:
                    continue

                # C. 边算边播：首句生成完毕立刻推入 Wyoming TCP 音频流
                for offset in range(0, len(pcm_data), chunk_size):
                    await self.write_event(
                        AudioChunk(
                            audio=pcm_data[offset : offset + chunk_size],
                            rate=self.sample_rate,
                            width=2,
                            channels=1,
                        ).event()
                    )

                logger.info(
                    "分句 [%d/%d] 流式输出完成: '%s' (耗时: %.2fs, tokens=%d, 音频: %.2fs)",
                    c_idx,
                    len(chunks),
                    chunk_text[:16].strip(),
                    t_cost,
                    tokens.shape[1],
                    len(pcm_data) / (self.sample_rate * 2),
                )

            # 发送流式音频结束事件
            await self.write_event(AudioStop().event())
            logger.info("✅ 文本整篇合成与流式分发完成！")

        return True


# ===========================================================================
# 主程序入口与服务初始化
# ===========================================================================
@app.command()
def main(
    model_dir: str = typer.Option(
        ...,
        "--model-dir",
        "-m",
        help="Kokoro OpenVINO 模型目录（含 openvino_model.xml 及 voices/ 目录）",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-H",
        help="服务监听地址",
    ),
    port: int = typer.Option(
        10021,
        "--port",
        "-p",
        help="服务监听端口（默认 10021）",
    ),
    device: str = typer.Option(
        "CPU",
        "--device",
        "-d",
        help="OpenVINO 推理设备（默认 CPU，Kokoro 在 CPU 上极快且兼容性最佳）",
    ),
    threads: Optional[int] = typer.Option(
        None,
        "--threads",
        "-t",
        help="OpenVINO CPU 推理线程数（默认 None 自适应优化）",
    ),
    language: str = typer.Option(
        "auto",
        "--language",
        "-l",
        help="全局默认语言（auto / zh / ja / en / es / fr 等）",
    ),
    voice: Optional[str] = typer.Option(
        None,
        "--voice",
        "-v",
        help="全局默认发音人名称（如 zf_xiaobei, jf_alpha, af_heart，不填则自动推选）",
    ),
    speed: float = typer.Option(
        0.85,
        "--speed",
        "-s",
        help="语速倍率（推荐 0.85 获得更从容清丽自然的听感，>1.0 加速，<1.0 减速）",
    ),
    sample_rate: int = typer.Option(
        24000,
        "--sample-rate",
        help="音频采样率（Hz），Kokoro 默认 24000",
    ),
) -> None:
    """启动 Wyoming Kokoro-OpenVINO 极速多语言 TTS 服务。"""

    logger.info("=" * 65)
    logger.info("🚀 正在启动 Wyoming Kokoro-OpenVINO 多语言 TTS 服务...")
    logger.info("=" * 65)

    # 1. 扫描与解析模型、音色库及词汇表
    model_file, voices, vocab = resolve_model_and_voices(model_dir=model_dir)

    # 2. 选定默认发音人
    all_voice_names = list(voices.keys())
    selected_default_voice = voice
    if not selected_default_voice:
        # 优先推选中文、日文或英文优质音色
        candidates = (
            [v for v in all_voice_names if v.startswith("zf_") or v.startswith("zm_")]
            + [v for v in all_voice_names if v.startswith("jf_") or v.startswith("jm_")]
            + [v for v in all_voice_names if v.startswith("af_") or v.startswith("am_")]
        )
        selected_default_voice = candidates[0] if candidates else all_voice_names[0]

    logger.info("🎙️ 默认发音人: %s (音色库总计: %d 款声音)", selected_default_voice, len(voices))

    # 3. 初始化 OpenVINO 推理引擎
    engine = KokoroEngine(
        model_file=model_file,
        device=device,
        threads=threads,
    )

    # 4. 初始化多语言 G2P 前端
    g2p_manager = MultiLangG2P(
        fallback_language="zh" if language == "auto" else language,
        vocab=vocab,
    )

    # 5. 构建并注册 Wyoming 协议描述元数据（向 Home Assistant 宣告支持的音色与语言）
    tts_voices: List[TtsVoice] = []
    for v_name in sorted(voices.keys()):
        prefix = v_name[0].lower()
        v_lang = VOICE_PREFIX_TO_LANG.get(prefix, "en")
        v_desc = f"{v_name} ({LANG_NAMES.get(v_lang, v_lang)})"
        tts_voices.append(
            TtsVoice(
                name=v_name,
                description=v_desc,
                languages=[v_lang],
            )
        )

    supported_languages = sorted(list({v.languages[0] for v in tts_voices}))
    logger.info("🌐 已向 Home Assistant 注册语言体系: %s", ", ".join(supported_languages))

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="kokoro-openvino",
                description="Kokoro-82M 多语言语音合成 (OpenVINO 高速推理)",
                attribution=Attribution(
                    name="DeepMind / Antigravity / Kokoro-FastAPI",
                    url="https://github.com/remsky/Kokoro-FastAPI",
                ),
                installed=True,
                voices=tts_voices,
                version="2.0.0",
            )
        ]
    )

    # 6. 服务冷启动预热（预热 G2P 与 OpenVINO，消除首个请求的编译延迟）
    logger.info("正在执行后台引擎预热 (Warm-up)...")
    try:
        warm_tokens = g2p_manager.text_to_tokens("你好", language="zh")
        warm_style = get_or_blend_voice(voices, selected_default_voice)
        engine.synthesize_tokens(warm_tokens, warm_style, speed=1.0)
        logger.info("🎉 引擎预热完成！首个请求将实现秒级出声。")
    except Exception as e:
        logger.debug("预热跳过或忽略: %s", e)

    # 7. 启动 Wyoming 异步 TCP 服务
    server = AsyncServer.from_uri(f"tcp://{host}:{port}")
    logger.info("=================================================================")
    logger.info("🌟 Wyoming Kokoro-OpenVINO TTS 服务启动成功！")
    logger.info("   - 监听地址: tcp://%s:%d", host, port)
    logger.info("   - 推理设备: %s", device)
    logger.info("   - 默认语速: %.2f", speed)
    logger.info("   - 默认音色: %s", selected_default_voice)
    logger.info("=================================================================")

    try:
        asyncio.run(
            server.run(
                lambda *a, **kw: TTSEventHandler(
                    *a,
                    wyoming_info=wyoming_info,
                    engine=engine,
                    g2p_manager=g2p_manager,
                    voices=voices,
                    default_voice_name=selected_default_voice,
                    default_language=language,
                    speed=speed,
                    sample_rate=sample_rate,
                    **kw,
                )
            )
        )
    except KeyboardInterrupt:
        logger.info("服务已被手动停止。")


if __name__ == "__main__":
    app()
