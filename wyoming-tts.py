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
import io
import logging
import re
import time
import wave
from functools import lru_cache, partial
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
_PUNCTUATION = ';:,.!?¡¿—…"«»"" ()'
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
# 多语言 G2P（字素到音素）转换提取与辅助工具
# ===========================================================================
def extract_phonemes(res) -> str:
    """从 misaki 返回的各种复杂数据结构（元组、字符串、对象、分段列表）中稳健提取音素字符串。"""
    if not res:
        return ""
    if isinstance(res, tuple):
        return res[0] if res else ""
    elif isinstance(res, str):
        return res
    elif hasattr(res, "phonemes"):
        return res.phonemes
    elif isinstance(res, list):
        parts = []
        for item in res:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, tuple) and len(item) >= 2:
                parts.append(str(item[1] if item[1] is not None else item[0]))
            elif isinstance(item, tuple) and len(item) == 1:
                parts.append(str(item[0]))
            elif hasattr(item, "phonemes"):
                parts.append(str(item.phonemes))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(res)


class MultiLangG2P:
    """
    多语言 G2P 转换管理器。
    按需延迟加载各语言的 G2P 引擎，支持同一服务内同时处理中、英、日等语言文本。
    """

    def __init__(self, fallback_language: str = "zh", vocab: Optional[Dict[str, int]] = None) -> None:
        self.fallback_language = fallback_language.split("-")[0].lower()
        self.vocab = vocab
        self._g2p_instances: Dict[str, object] = {}

    def get_g2p(self, language: Optional[str] = None):
        """获取指定语言的 G2P 实例（按需加载并缓存）。支持单字母与 ISO 语言代码。"""
        lang = (language or self.fallback_language or "zh").split("-")[0].lower()
        if lang == "auto" or not lang:
            lang = self.fallback_language or "zh"

        # 标准化映射：支持 Kokoro-FastAPI 的单字母代码与标准 ISO 语言代码互通
        lang_norm_map = {
            "z": "zh", "j": "ja", "a": "en", "b": "en",
            "e": "es", "f": "fr", "h": "hi", "i": "it", "p": "pt", "k": "ko"
        }
        lang = lang_norm_map.get(lang, lang)

        if lang in self._g2p_instances:
            return self._g2p_instances[lang]

        g2p = self._create_g2p(lang)
        self._g2p_instances[lang] = g2p
        return g2p

    def _create_g2p(self, lang: str):
        """创建对应语言的 misaki G2P 引擎。"""
        try:
            if lang in ("zh", "z"):
                from misaki import zh
                cls = getattr(zh, "ZHG2P", None) or getattr(zh, "G2P", None)
                if cls is None:
                    raise ImportError("无法在 misaki.zh 中找到 ZHG2P 类")
                logger.info("已加载中文 G2P 引擎 (misaki.zh.%s)", cls.__name__)

                # 优化英文单词回调：无多余空格污染，稳健提取音素，带 LRU 高速缓存
                @lru_cache(maxsize=1024)
                def fast_en_callable(w: str) -> str:
                    w = w.strip()
                    if not w:
                        return ""
                    try:
                        en_engine = self.get_g2p("en")
                        res = en_engine(w)
                        return extract_phonemes(res).strip()
                    except Exception:
                        return w

                # 智能版本自适应：
                # 检查当前模型词汇表中是否包含 Kokoro v1.1-zh 专属的注音符号（如 'ㄓ', 'ㄅ' 等）
                # - 若包含，说明为 Kokoro-82M-v1.1-zh 原生注音模型，使用 version="1.1"；
                # - 若不包含（如 Kokoro-82M / Echo9Zulu 等基础 OpenVINO 模型），官方推荐使用 version=None，
                #   生成与基础模型词汇表 100% 匹配的国际音标与声调箭头（↓↑→↗↘），保证发音自然优美，绝不掉词！
                is_v11 = bool(self.vocab and any(m in self.vocab for m in ("ㄓ", "ㄅ", "中", "阳", "要")))
                zh_version = "1.1" if is_v11 else None
                logger.info("初始化中文 G2P: version=%s (检测到 v1.1 注音词表: %s)", zh_version, is_v11)

                try:
                    return cls(version=zh_version, en_callable=fast_en_callable)
                except (TypeError, ValueError):
                    try:
                        return cls(version=None, en_callable=fast_en_callable)
                    except (TypeError, ValueError):
                        return cls(en_callable=fast_en_callable)

            elif lang in ("en", "a", "b"):
                from misaki import en
                cls = getattr(en, "G2P", None) or getattr(en, "ENG2P", None)
                if cls is None:
                    raise ImportError("无法在 misaki.en 中找到 G2P 类")
                logger.info("已加载英文 G2P 引擎 (misaki.en.%s)", cls.__name__)

                # 尝试启用 espeak fallback 兜底处理生僻/超出词表的英文单词（参考 Kokoro-FastAPI）
                fallback = None
                try:
                    from misaki import espeak
                    fallback = espeak.EspeakFallback(british=(lang in ("b", "gb")))
                    logger.debug("已为英文 G2P 启用 EspeakFallback 兜底")
                except Exception as e:
                    logger.debug("未启用 EspeakFallback (可选): %s", e)

                try:
                    return cls(trf=False, british=(lang in ("b", "gb")), fallback=fallback)
                except TypeError:
                    return cls(trf=False)

            elif lang in ("ja", "j"):
                from misaki import ja
                cls = getattr(ja, "JAG2P", None) or getattr(ja, "G2P", None)
                if cls is None:
                    raise ImportError("无法在 misaki.ja 中找到 JAG2P 类")
                logger.info("已加载日文 G2P 引擎 (misaki.ja.%s)", cls.__name__)
                return cls()

            elif lang in ("es", "e"):
                from misaki import es
                cls = getattr(es, "ESG2P", None) or getattr(es, "G2P", None)
                return cls()

            elif lang in ("fr", "f"):
                from misaki import fr
                cls = getattr(fr, "FRG2P", None) or getattr(fr, "G2P", None)
                return cls()

            elif lang in ("hi", "h"):
                from misaki import hi
                cls = getattr(hi, "HIG2P", None) or getattr(hi, "G2P", None)
                return cls()

            elif lang in ("it", "i"):
                from misaki import it
                cls = getattr(it, "ITG2P", None) or getattr(it, "G2P", None)
                return cls()

            elif lang in ("pt", "p"):
                from misaki import pt
                cls = getattr(pt, "PTG2P", None) or getattr(pt, "G2P", None)
                return cls()

            elif lang in ("ko", "k"):
                from misaki import ko
                cls = getattr(ko, "KOG2P", None) or getattr(ko, "G2P", None)
                return cls()

            else:
                logger.warning("未找到专门的 '%s' G2P 引擎，回退到基础英文/通用 G2P", lang)
                from misaki.en import G2P
                return G2P(trf=False)

        except ImportError as exc:
            install_hint = {
                "zh": "pip install ordered-set pypinyin pypinyin-dict jieba cn2an",
                "en": "pip install spacy num2words",
                "ja": "pip install pyopenjtalk fugashi unidic-lite jaconv mojimoji",
            }.get(lang, f"pip install 'misaki[{lang}]'")

            err_msg = (
                f"加载语言 '{lang}' 的 G2P 引擎失败，缺少必要依赖 ({exc})。\n"
                f"请在当前虚拟环境中执行命令安装: {install_hint}"
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg) from exc

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


# 常见英语缩写词（句中出现时不应作为句子断句分割，参考 Kokoro-FastAPI Issue #308 / PR #520）
COMMON_ABBREVIATIONS = (
    r"Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|approx|app|dept|min|max|fig|no|vol|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec|St|Ave|Rd|Corp|Inc|Ltd|Co"
)


def normalize_mixed_text(text: str) -> str:
    """
    对中英日混合文本进行预清洗与规范化:
    1. 中文字符与英文字母边界补空格，避免发音黏连；
    2. 常见全大写缩写词（如 AI, CPU, GPU, TTS, USB, IoT）展开为空格分隔读音（如 A I, C P U）；
    3. 全角空格转半角，规范化标点空白。
    """
    text = text.replace("\u3000", " ")
    # 中文字符与英文字母边界补空格
    text = re.sub(r"([\u4e00-\u9fff])([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])([\u4e00-\u9fff])", r"\1 \2", text)
    # 大写缩写展开（如 AI -> A I）
    text = re.sub(r"\b([A-Z]{2,4})\b", lambda m: " ".join(list(m.group(1))), text)
    # 多空格压缩
    text = re.sub(r"[ \t]+", " ", text)
    return text


def smart_sentence_split(text: str) -> List[str]:
    """
    智能句子切分算法（借鉴 Kokoro-FastAPI 的 UAX #29 文本分割思想）：
    1. 保护缩写词中的点（如 Dr. Wang, etc., e.g.）以及浮点数（如 3.14, 0.85），避免被误切断；
    2. 兼容中日英文句子结束符（。！？!? 以及英文句号加空格）；
    3. 保留句末标点符号，确保送入 Kokoro 模型时具备自然的语调起伏与停顿；
    4. 消除括号、多重标点引发的错误分割。
    """
    if not text:
        return []

    # 1. 临时占位符保护英文缩写和浮点数中的 '.'
    protected_text = re.sub(r"(\d)\.(\d)", r"\g<1>" + "\u200b" + r"\g<2>", text)
    protected_text = re.sub(
        rf"\b({COMMON_ABBREVIATIONS})\.",
        r"\g<1>" + "\u200b",
        protected_text,
        flags=re.IGNORECASE,
    )
    protected_text = re.sub(r"\b([A-Z])\.", r"\g<1>" + "\u200b", protected_text)

    # 2. 匹配主句结束符：中日句号问叹标点 [。！？!?] 或 英文句号后接空白/换行/行尾
    # 同时保留标点附在句尾
    sentence_pattern = r"([^。！？!?\n]+(?:[。！？!?]+|\.(?=\s|$))?)"
    raw_sentences = [s.strip() for s in re.findall(sentence_pattern, protected_text) if s.strip()]

    # 3. 恢复占位符
    sentences = [s.replace("\u200b", ".") for s in raw_sentences]
    return sentences if sentences else [text]


def split_text_into_chunks(text: str, max_chunk_chars: int = 35) -> List[str]:
    """
    将长文本智能切分为自然的语音分句（推荐默认约 25-35 字/句）。
    整合 Kokoro-FastAPI 的 UAX #29 智能断句、缩写保护与自然从句切分:
      1. 保护英文缩写（Dr., etc.）与浮点数（3.14），绝不生硬截断；
      2. 完整保留句尾标点（。！？!?），提供自然声学顿挫与语调起伏；
      3. 超长句子按从句次级标点（逗号、分号、顿号、破折号）平滑切分；
      4. 首句极速出声（Time To First Audio 压缩至 1~2 秒内），后续音频平滑边播边算，彻底消除卡顿。
    """
    text = normalize_mixed_text(text.strip())
    if not text:
        return []

    sentences = smart_sentence_split(text)
    chunks: List[str] = []

    for sentence in sentences:
        if len(sentence) <= max_chunk_chars:
            chunks.append(sentence)
            continue

        # 如果单个句子超出 max_chunk_chars，按次级从句标点切分（保留逗号、分号、顿号等）
        clause_pattern = r"([^；;，,、:—\r\n]+[；;，,、:—]*)"
        raw_clauses = [c.strip() for c in re.findall(clause_pattern, sentence) if c.strip()]
        if not raw_clauses:
            raw_clauses = [sentence]

        current_chunk = ""
        for clause in raw_clauses:
            if len(current_chunk) + len(clause) > max_chunk_chars and current_chunk:
                chunks.append(current_chunk)
                current_chunk = clause
            else:
                need_space = bool(current_chunk and clause and clause[0].isalnum() and clause[0].isascii())
                current_chunk += (" " if need_space else "") + clause
        if current_chunk:
            chunks.append(current_chunk)

    return chunks


def process_text_with_misaki(
    g2p,
    text: str,
    vocab: Optional[Dict[str, int]] = None,
    **kwargs,
) -> np.ndarray:
    """
    使用 misaki 将文本转换为 Kokoro 模型所需的 token ID 序列。
    支持从模型 config.json 动态传入精确 vocab 映射字典。
    """
    res = g2p(text)
    phoneme_str = extract_phonemes(res)
    logger.info("音素转换: '%s' -> '%s'", text, phoneme_str)

    if not phoneme_str:
        logger.warning("文本 '%s' 未能提取出有效音素", text)
        return np.array([[0, 0]], dtype=np.int64)

    active_vocab = vocab if vocab is not None else VOCAB

    # 逐字符查表转换为 token ID
    # 过滤掉不存在的字符，首尾各填充 0 (PAD)
    matched_ids = [active_vocab[c] for c in phoneme_str if c in active_vocab]
    if not matched_ids:
        logger.warning("音素 '%s' 中的字符均未命中词汇表！", phoneme_str)
        matched_ids = [0]

    has_zh_chars = bool(re.search(r"[\u4E00-\u9FFF]", text))
    if has_zh_chars and len(matched_ids) < 10 and len(text) > 5:
        logger.warning(
            "⚠️ 中文句子 token 匹配数异常偏低 (%d 字 -> 仅 %d tokens)，请检查当前模型目录下的 config.json 词汇表配置！",
            len(text),
            len(matched_ids),
        )

    # Kokoro 模型上下文最大限制为 510 tokens (PLBERT max_position_embeddings=512)
    if len(matched_ids) > 508:
        matched_ids = matched_ids[:508]

    token_ids = [0] + matched_ids + [0]
    return np.array([token_ids], dtype=np.int64)


# ===========================================================================
# 模型与声音资源自动解析工具
# ===========================================================================
def load_voice_file(file_path: Path) -> np.ndarray:
    """加载单个声音文件（支持 .pt, .pth, .bin, .npy），返回 numpy 数组。"""
    suffix = file_path.suffix.lower()

    if suffix in [".pt", ".pth"]:
        # 1. 优先使用 torch 加载
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
            return arr
        except ImportError:
            # 2. 若未安装 torch，尝试直接解析 PyTorch 的 zip 归档格式提取二进制 float32 权重
            import zipfile
            try:
                with zipfile.ZipFile(file_path, "r") as z:
                    for member in z.namelist():
                        if member.endswith("/0") or member.startswith("archive/data/"):
                            with z.open(member) as f:
                                raw = f.read()
                                arr = np.frombuffer(raw, dtype=np.float32)
                                if arr.size % 256 == 0:
                                    return arr.reshape(-1, 256)
                                return arr
            except Exception:
                pass
            raise RuntimeError(
                f"加载 {file_path.name} 失败：请安装 PyTorch (pip install torch) 以解析 .pt 声音文件"
            )

    elif suffix == ".bin":
        arr = np.fromfile(file_path, dtype=np.float32)
        if arr.size % 256 == 0:
            arr = arr.reshape(-1, 256)
        return arr

    elif suffix in [".npy", ".npz"]:
        arr = np.load(file_path, allow_pickle=True)
        if arr.ndim == 1 and arr.size % 256 == 0:
            arr = arr.reshape(-1, 256)
        return arr

    raise ValueError(f"不支持的声音文件格式: {suffix}")


# ===========================================================================
# 发音人多音色线性融合 (Kokoro-FastAPI 声音混合机制)
# ===========================================================================
_BLENDED_VOICES_CACHE: Dict[str, np.ndarray] = {}


def parse_voice_blend(voice_spec: str) -> List[Tuple[str, float]]:
    """
    解析发音人混合表达式（借鉴 Kokoro-FastAPI 声音融合语法）。
    支持格式:
      - 单音色: 'zf_xiaobei' 或 'af_heart'
      - 均等权重混合: 'zf_xiaobei+af_heart'
      - 自定义权重混合: 'zf_xiaobei(0.7)+af_heart(0.3)' 或 'af_bella(2)+af_sky(1)'
    """
    if not voice_spec or not voice_spec.strip():
        return []

    parts = [p.strip() for p in voice_spec.split("+") if p.strip()]
    blend_list: List[Tuple[str, float]] = []
    pattern = re.compile(r"^([a-zA-Z0-9_\-\.]+)(?:\(([0-9.]+)\))?$")

    for part in parts:
        m = pattern.match(part)
        if m:
            v_name = m.group(1).strip()
            weight = float(m.group(2)) if m.group(2) is not None else 1.0
            blend_list.append((v_name, weight))
        else:
            blend_list.append((part, 1.0))

    total_weight = sum(w for _, w in blend_list)
    if total_weight > 0:
        blend_list = [(name, w / total_weight) for name, w in blend_list]
    return blend_list


def get_or_blend_voice(voices: Dict[str, np.ndarray], voice_spec: Optional[str]) -> np.ndarray:
    """
    获取单发音人音色或融合多发音人音色（支持 Kokoro-FastAPI 风格的多音色加权线性融合）。
    """
    if not voices:
        raise ValueError("发音人列表为空，无法获取声音！")

    if not voice_spec:
        first_key = next(iter(voices))
        return voices[first_key]

    voice_spec = voice_spec.strip()

    # 1. 直接命中已加载单发音人
    if voice_spec in voices:
        return voices[voice_spec]

    # 2. 检查融合缓存
    if voice_spec in _BLENDED_VOICES_CACHE:
        return _BLENDED_VOICES_CACHE[voice_spec]

    # 3. 混合发音人解析
    blend_specs = parse_voice_blend(voice_spec)
    if not blend_specs:
        first_key = next(iter(voices))
        return voices[first_key]

    if len(blend_specs) == 1:
        v_name = blend_specs[0][0]
        if v_name in voices:
            return voices[v_name]
        logger.warning("未找到发音人 '%s'，使用默认发音人", v_name)
        first_key = next(iter(voices))
        return voices[first_key]

    # 多音色加权线性融合
    arrays = []
    weights = []
    for v_name, weight in blend_specs:
        if v_name in voices:
            arr = voices[v_name]
            arrays.append(arr)
            weights.append(weight)
        else:
            logger.warning("发音人混合中未找到音色 '%s'，已跳过", v_name)

    if not arrays:
        logger.warning("发音人混合表达式 '%s' 中的所有音色均未命中，使用默认发音人", voice_spec)
        first_key = next(iter(voices))
        return voices[first_key]

    # 归一化可用音色的权重
    w_sum = sum(weights)
    if w_sum > 0:
        weights = [w / w_sum for w in weights]
    else:
        weights = [1.0 / len(weights)] * len(weights)

    # 确保统一对齐维度并裁切到最小长度
    min_len = min(a.shape[0] if a.ndim >= 2 else 1 for a in arrays)

    standardized_arrays = []
    for a in arrays:
        if a.ndim == 1:
            standardized_arrays.append(a.reshape(1, -1))
        elif a.ndim == 3:
            standardized_arrays.append(a.squeeze(0)[:min_len])
        else:
            standardized_arrays.append(a[:min_len])

    blended = np.zeros_like(standardized_arrays[0][:min_len], dtype=np.float32)
    for arr, w in zip(standardized_arrays, weights):
        blended += arr[:min_len].astype(np.float32) * float(w)

    _BLENDED_VOICES_CACHE[voice_spec] = blended
    logger.info("成功创建融合发音人 '%s' (融合 %d 种音色)", voice_spec, len(arrays))
    return blended


def resolve_model_and_voices(
    model_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    voices_path: Optional[str] = None,
) -> Tuple[Path, Dict[str, np.ndarray], Optional[Dict[str, int]]]:
    """
    自动查找并解析 Kokoro 模型、发音人声音数据以及官方 config.json 词汇表。

    支持多种形式:
      1. 官方 OpenVINO 目录:
         包含 openvino_model.xml, openvino_model.bin 以及 voices/*.pt 或 voices/*.bin
      2. 单文件/目录形式:
         --model-path 指定 .xml 或 .onnx，--voices-path 指定 voices 目录或 .npy 文件
    """
    # 1. 确定模型路径
    target_model_file: Optional[Path] = None

    if model_path:
        p = Path(model_path)
        if p.is_dir():
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
    base_dir = target_model_file.parent

    # 尝试加载官方 config.json 中的 vocab 字典（保证中/英/日音标精准对齐）
    vocab: Optional[Dict[str, int]] = None
    config_candidates = [
        base_dir / "config.json",
        base_dir.parent / "config.json",
    ]
    if model_dir:
        config_candidates.insert(0, Path(model_dir) / "config.json")
    if model_path:
        config_candidates.insert(0, Path(model_path).parent / "config.json")

    for cfg_cand in config_candidates:
        if cfg_cand.exists():
            try:
                import json
                with cfg_cand.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "vocab" in cfg and isinstance(cfg["vocab"], dict):
                    vocab = cfg["vocab"]
                    is_v11 = any(m in vocab for m in ("ㄓ", "ㄅ", "中", "阳", "要"))
                    logger.info("从 %s 成功载入官方模型词汇表 (包含 %d 个音标 token 符号, v1.1-zh=%s)", cfg_cand.name, len(vocab), is_v11)
                    break
            except Exception as e:
                logger.warning("解析 %s 的 vocab 失败: %s", cfg_cand, e)

    # 2. 确定声音资源位置
    voices: Dict[str, np.ndarray] = {}

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

        logger.info("检查声音候选路径: %s", src)

        # 场景 A: voices 目录（包含众多 *.pt, *.bin, *.npy 单个音色文件）
        if src.is_dir():
            voice_files = []
            for ext in ["*.pt", "*.pth", "*.bin", "*.npy"]:
                voice_files.extend(src.glob(ext))

            if voice_files:
                logger.info("在目录 %s 中找到 %d 个声音文件 (*.pt / *.bin 等)", src, len(voice_files))
                for vf in voice_files:
                    v_name = vf.stem  # 如 'af_heart'
                    try:
                        data = load_voice_file(vf)
                        voices[v_name] = data
                    except Exception as err:
                        logger.warning("加载声音文件 %s 失败: %s", vf, err)
                if voices:
                    logger.info("成功加载 %d 个发音人音色", len(voices))
                    break

        # 场景 B: voices.npy / .npz / voices.bin 单个打包文件
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
            f"未在候选路径中找到可用的声音数据（支持 voices/ 目录下的 *.pt, *.bin 或 voices.npy 打包文件）。"
            f"请检查路径: model_dir={model_dir}, voices_path={voices_path}"
        )

    return target_model_file, voices, vocab


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
        vocab: Optional[Dict[str, int]] = None,
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
        self.vocab = vocab
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

            # 1. 确定声音：优先使用 Wyoming 协议指定的 voice (支持单音色或多音色融合表达式)
            voice_name = self.default_voice_name
            if synth.voice and synth.voice.name:
                v_cand = synth.voice.name.strip()
                if v_cand in self.voices or "+" in v_cand:
                    voice_name = v_cand

            # 2. 智能确定目标合成语言：
            # 核心策略（内容优先）：
            # - 若文本包含中文汉字，必须使用 'zh'（中文 G2P 原生支持汉字与中英混排，而英文 G2P 会完全过滤丢失汉字）
            # - 若文本包含日文假名，必须使用 'ja'
            # - 若文本为纯西文字符/英文：优先使用 'en'（即使用户使用 zf_* 发音人，也能以地道英文音标发音）
            # - 若无法判定（纯数字或标点）：优先遵循客户端请求语言、声音前缀或兜底配置
            detected_lang = self.g2p_manager.detect_language_from_text(text)

            client_lang = None
            if synth.voice and synth.voice.language:
                client_lang = synth.voice.language.split("-")[0].lower()

            if detected_lang in ("zh", "ja", "en"):
                lang = detected_lang
            elif client_lang and client_lang != "auto":
                lang = client_lang
            elif voice_name:
                prefix = voice_name[0].lower()
                lang = VOICE_PREFIX_TO_LANG.get(prefix, self.default_language or "zh")
            else:
                lang = self.default_language or "zh"

            # 3. 确定语速：优先使用 Wyoming 事件中传递的 speed，否则使用全局默认配置
            req_speed = self.speed
            if event.data:
                if "speed" in event.data and isinstance(event.data["speed"], (int, float)):
                    req_speed = float(event.data["speed"])
                elif isinstance(event.data.get("voice"), dict) and "speed" in event.data["voice"]:
                    req_speed = float(event.data["voice"]["speed"])

            logger.info("TTS 合成: 语言=%s (检测=%s), 声音=%s, 语速=%.2f, 文本=%s", lang, detected_lang, voice_name, req_speed, text)

            # 发送音频流开始事件
            await self.write_event(
                AudioStart(
                    rate=self.sample_rate,
                    width=2,       # 16-bit PCM
                    channels=1,    # 单声道
                ).event()
            )

            # 4. 智能分句流式合成与分块发送:
            # 保证长文本不会超出模型 512 上下文，并且首句生成完毕立刻出声，极大缩短等待时间！
            chunks = split_text_into_chunks(text)
            logger.debug("文本分句完成 (%d 句): %s", len(chunks), chunks)

            chunk_size = 1024
            for c_idx, chunk_text in enumerate(chunks, 1):
                chunk_audio = await self.call_openvino(
                    text=chunk_text,
                    language=lang,
                    voice_name=voice_name,
                    speed=req_speed,
                )
                if not chunk_audio:
                    continue

                for i in range(0, len(chunk_audio), chunk_size):
                    await self.write_event(
                        AudioChunk(
                            audio=chunk_audio[i : i + chunk_size],
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
        speed: Optional[float] = None,
    ) -> bytes:
        """
        使用 OpenVINO Kokoro 模型进行多语言语音合成。
        """
        t0 = time.perf_counter()
        target_lang = language or self.default_language or "zh"

        # 每句切片自适应保护：
        # - 若切片含有汉字，必须使用 zh（支持中英混排）
        # - 若切片含有假名，必须使用 ja
        # - 若切片纯为英文字母且无汉字假名，切换至 en 获得纯正英式/美式语调与音标
        if re.search(r"[\u4E00-\u9FFF]", text):
            target_lang = "zh"
        elif re.search(r"[\u3040-\u309F\u30A0-\u30FF]", text):
            target_lang = "ja"
        elif re.search(r"[a-zA-Z]", text) and not re.search(r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]", text):
            if target_lang in ("zh", "ja"):
                target_lang = "en"

        g2p = self.g2p_manager.get_g2p(target_lang)

        # 文本 → 音素 token IDs（传入官方精准 vocab 字典）
        tokens = process_text_with_misaki(
            g2p,
            text,
            vocab=self.vocab,
        )
        t_g2p = time.perf_counter() - t0

        # 获取声音风格向量（支持单音色与多音色加权融合）
        v_name = voice_name or self.default_voice_name
        voice_data = get_or_blend_voice(self.voices, v_name)

        # 提取当前 token 长度适配的 style 向量
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

        # 确保 style 为 float32 且 shape 规范为 (1, 256)
        if style.shape != (1, 256):
            style = style.reshape(1, 256)
        style = style.astype(np.float32)

        target_speed = speed if speed is not None else self.speed
        speed_array = np.array([target_speed], dtype=np.float32)

        # 在线程池中执行推理
        t_infer_start = time.perf_counter()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._run_inference,
            tokens,
            style,
            speed_array,
        )
        t_infer = time.perf_counter() - t_infer_start

        logger.info(
            "合成片段完成: %s... (G2P: %.2fs, 模型推理: %.2fs, 总计: %.2fs, tokens=%d)",
            text[:16].strip(),
            t_g2p,
            t_infer,
            t_g2p + t_infer,
            token_len,
        )

        return result

    def _run_inference(
        self,
        tokens: np.ndarray,
        style: np.ndarray,
        speed: np.ndarray,
    ) -> bytes:
        """执行 OpenVINO 模型推理。"""
        # Kokoro OpenVINO 官方 IR 模型接受固定顺序的列表输入: [input_ids, ref_s, speed]
        try:
            output = self.compiled_model([tokens, style, speed])
        except Exception as e:
            logger.debug("顺序列表传参未直接命中，尝试命名端口传参: %s", e)
            inputs = {}
            for inp in self.compiled_model.inputs:
                name = inp.get_any_name().lower()
                if "token" in name or "input" in name or "id" in name:
                    inputs[inp] = tokens
                elif "style" in name or "ref" in name:
                    inputs[inp] = style
                elif "speed" in name:
                    inputs[inp] = speed
                else:
                    inputs[inp] = tokens
            output = self.compiled_model(inputs)

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
        "CPU",
        "--device",
        "-d",
        help="OpenVINO 推理设备（默认 CPU，Kokoro 模型在 CPU 上极快且兼容性最佳）",
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
        help="语速倍率（默认 1.0；推荐设为 0.8~0.85 获得更从容自然的中文/日文听感，>1.0 加速，<1.0 减速）",
    ),
    sample_rate: int = typer.Option(
        24000,
        "--sample-rate",
        help="音频采样率（Hz），Kokoro 默认 24000",
    ),
    threads: Optional[int] = typer.Option(
        None,
        "--threads",
        "-t",
        help="OpenVINO CPU 推理线程数（默认 None 由 OpenVINO 根据系统 CPU 拓扑自适应优化）",
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
    resolved_model_file, voices, vocab = resolve_model_and_voices(
        model_dir=model_dir,
        model_path=model_path,
        voices_path=voices_path,
    )

    all_voice_names = list(voices.keys())
    selected_voice = voice_name
    if not selected_voice or (selected_voice not in voices and "+" not in selected_voice):
        # 优先默认推选中文优质音色（zf_* / zm_*），其次英文音色（af_* / am_*），若无则取首个
        preferred_voices = (
            [v for v in all_voice_names if v.startswith("zf_") or v.startswith("zm_")] +
            [v for v in all_voice_names if v.startswith("af_") or v.startswith("am_")]
        )
        selected_voice = preferred_voices[0] if preferred_voices else all_voice_names[0]

    logger.info("已加载 %d 个声音，默认声音: %s", len(all_voice_names), selected_voice)

    # ---- 初始化 OpenVINO 推理引擎 ----
    core = ov.Core()
    logger.info("可用推理设备: %s", core.available_devices)
    logger.info("正在加载并编译模型: %s (设备: %s)", resolved_model_file, device)

    model = core.read_model(str(resolved_model_file))
    ov_config: Dict[str, str] = {
        "PERFORMANCE_HINT": "LATENCY",
    }
    if threads is not None and threads > 0:
        ov_config["INFERENCE_NUM_THREADS"] = str(threads)
        logger.info("已指定 OpenVINO 推理线程数: %d", threads)
    else:
        logger.info("OpenVINO 线程模式: 自动根据硬件拓扑自适应调度 (LATENCY 模式)")

    try:
        compiled_model = core.compile_model(model, device, config=ov_config)
        logger.info("模型编译成功 (设备: %s, 模式: LATENCY)", device)
    except Exception as exc:
        if device.upper() != "CPU":
            logger.warning("在 '%s' 上编译模型失败: %s。自动回退到 'CPU' 设备...", device, exc)
            device = "CPU"
            compiled_model = core.compile_model(model, "CPU", config=ov_config)
            logger.info("模型在 'CPU' 设备上编译成功")
        else:
            raise

    # ---- 初始化多语言 G2P 管理器 ----
    default_lang = language or "auto"
    g2p_manager = MultiLangG2P(
        fallback_language="zh" if default_lang == "auto" else default_lang,
        vocab=vocab,
    )

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

    # ---- 预热 G2P 与 OpenVINO 推理引擎（消除首句冷启动数十秒延迟）----
    logger.info("正在执行首句冷启动预热 (G2P & OpenVINO)...")
    try:
        warmup_g2p = g2p_manager.get_g2p("zh")
        g2p_manager.get_g2p("en")
        warmup_tokens = process_text_with_misaki(warmup_g2p, "你好", vocab=vocab)
        dummy_style = np.zeros((1, 256), dtype=np.float32)
        dummy_speed = np.array([1.0], dtype=np.float32)
        compiled_model([warmup_tokens, dummy_style, dummy_speed])
        logger.info("🎉 服务预热完成，已进入就绪状态（首句将实现秒级发声）！")
    except Exception as exc:
        logger.debug("服务预热跳过或出现异常 (不影响正常运行): %s", exc)

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
                vocab=vocab,
                default_voice_name=selected_voice,
                default_language=default_lang,
                speed=speed,
                sample_rate=sample_rate,
            )
        )
    )


if __name__ == "__main__":
    app()
