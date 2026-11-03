#!/usr/bin/env python3
"""
Wyoming TTS Service (Kokoro OpenVINO High-Speed Multilingual)
============================================================
Kokoro-82M text-to-speech service accelerated by Intel OpenVINO, seamlessly
integrated with Home Assistant via the standard Wyoming protocol.

Features & Design:
    - Designed with inspiration from remsky/Kokoro-FastAPI and hexgrad/kokoro frontends.
    - Streaming chunking and lazy phonemization for sub-second first-chunk audio (TTFA < 0.3s).
    - Native multilingual support:
        * Mandarin Chinese (zh / z): Native misaki.zh G2P with Chinese/English mixed text handling
        * Japanese (ja / j): Native misaki.ja G2P with accurate kanji & kana pronunciation
        * English (en / a, b): American English (a) and British English (b)
        * Spanish (es / e), French (fr / f), Hindi (hi / h), Italian (it / i), Portuguese (pt / p)
    - Weighted voice blending: e.g. 'zf_xiaobei(0.7)+af_heart(0.3)'.
    - Intel OpenVINO optimization (LATENCY mode, thread configuration, cold-start warm-up).

Usage Example:
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

from i18n import t

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wyoming-tts")

# ---------------------------------------------------------------------------
# Typer CLI application
# ---------------------------------------------------------------------------
app = typer.Typer(help=t("cli.help"))


# ===========================================================================
# Kokoro Default Vocabulary & Language Mappings (178 Tokens)
# ===========================================================================
_DEFAULT_VOCAB_SYMBOLS = [
    "$", ";", ":", ",", ".", "!", "?", "—", "…", "“", "”", "«", "»", "   ", "  ", " ",
    "¡", "¿", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "ɑ", "ɒ", "æ", "ɓ", "ɔ", "ɕ", "ç", "ɗ", "ɘ", "ə", "ɚ", "ɛ", "ɜ", "ɟ", "ɠ", "ɢ",
    "ɣ", "ɦ", "ɧ", "ɨ", "ɪ", "ɫ", "ɬ", "ɭ", "ɮ", "ɯ", "ɰ", "ɱ", "ɲ", "ɳ", "ɵ", "ɶ",
    "ɸ", "ɹ", "ɺ", "ɻ", "ɼ", "ɽ", "ɾ", "ɿ", "ʀ", "ʁ", "ʂ", "ʃ", "ʄ", "ʅ", "ʆ", "ʇ",
    "ʈ", "ʉ", "ʊ", "ʋ", "ʌ", "ʍ", "ʎ", "ʏ", "ʐ", "ʑ", "ʒ", "ʓ", "ʔ", "ʕ", "ʖ", "ʗ",
    "ʘ", "ʙ", "ʚ", "ʛ", "ʜ", "ʝ", "ʞ", "ʟ", "ʠ", "ʡ", "ʢ", "ʣ", "ʤ", "ʥ", "ʦ", "ʧ",
    "ʩ", "ʪ", "ʫ", "ʬ", "ʭ", "ʮ", "ʯ", "ʰ", "ʱ", "ʲ", "ʳ", "ʴ", "ʵ", "ʶ", "ʷ", "ʸ",
    "ʹ", "ʺ", "ʻ", "ʼ", "ʽ", "ʾ", "ʿ", "ˀ", "ˁ", "˂", "˃", "˄"
]
DEFAULT_VOCAB: Dict[str, int] = {s: i for i, s in enumerate(_DEFAULT_VOCAB_SYMBOLS)}

# Voice prefix to primary language code mapping
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

# Single-letter and ISO language code bidirectional mapping
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


def get_lang_display_name(code: str) -> str:
    """Return localized display name for a language code from YAML."""
    translated = t(f"languages.{code}")
    return translated if translated != f"languages.{code}" else code


# Common abbreviation protection list (avoids splitting sentences on dots)
ABBREVIATIONS = (
    r"Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|approx|dept|min|max|fig|no|vol|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec|St|Ave|Rd|Corp|Inc|Ltd|Co"
)


# ===========================================================================
# Streaming Text Processor
# ===========================================================================
def normalize_text_spacing(text: str) -> str:
    """
    Normalize text whitespace and boundaries between CJK and Latin characters:
    1. Converts full-width spaces to half-width.
    2. Inserts space between Chinese characters and alphanumeric words.
    3. Preserves acronyms (AI, CPU, GPU, OpenVINO).
    """
    if not text:
        return ""
    text = text.replace("\u3000", " ").strip()
    text = re.sub(r"([\u4e00-\u9fff])([a-zA-Z0-9])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z0-9])([\u4e00-\u9fff])", r"\1 \2", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def split_text_into_streaming_chunks(text: str, max_chunk_chars: int = 65) -> List[str]:
    """
    Intelligently split long text into natural sentence streaming clauses.
    Principles:
      1. Protect abbreviations and decimals (e.g. Dr., 3.14).
      2. Preserve terminal punctuation (。！？!?) for expressive acoustic intonation.
      3. Sub-split long clauses along intermediate punctuation (，、；;:—).
      4. Maintain clause size around 15-35 characters for sub-second first-chunk audio.
    """
    text = normalize_text_spacing(text)
    if not text:
        return []

    ZWS = "\u200b"
    protected = re.sub(r"(\d)\.(\d)", rf"\g<1>{ZWS}\g<2>", text)
    protected = re.sub(rf"\b({ABBREVIATIONS})\.", rf"\g<1>{ZWS}", protected, flags=re.IGNORECASE)
    protected = re.sub(r"\b([A-Z])\.", rf"\g<1>{ZWS}", protected)

    sentence_pattern = r"([^。！？!?\r\n]+(?:[。！？!?]+|\.(?=\s|$))?)"
    raw_sentences = [s.strip() for s in re.findall(sentence_pattern, protected) if s.strip()]

    sentences = [s.replace(ZWS, ".") for s in raw_sentences]
    if not sentences:
        sentences = [text]

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

        safe_clauses: List[str] = []
        for c in raw_clauses:
            if len(c) > max_chunk_chars:
                if " " in c:
                    words = c.split(" ")
                    buf = ""
                    for w in words:
                        if len(buf) + len(w) + 1 > max_chunk_chars and buf:
                            safe_clauses.append(buf)
                            buf = w
                        else:
                            buf = (buf + " " + w).strip()
                    if buf:
                        safe_clauses.append(buf)
                else:
                    for k in range(0, len(c), max_chunk_chars):
                        safe_clauses.append(c[k : k + max_chunk_chars])
            else:
                safe_clauses.append(c)

        current_chunk = ""
        for clause in safe_clauses:
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


split_text_into_chunks = split_text_into_streaming_chunks


# ===========================================================================
# Kokoro Multi-Language G2P Frontend
# ===========================================================================
def detect_language(text: str, default_lang: str = "zh") -> str:
    """
    Detect language from character features:
    - Contains Hiragana or Katakana -> Japanese (ja)
    - Contains Hanzi without kana -> Chinese (zh)
    - Pure Latin characters -> English (en)
    - Otherwise fallback to default language
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
    Kokoro Multilingual G2P (Grapheme-to-Phoneme) Frontend Manager.
    Supports on-demand lazy loading and caching for all Kokoro language families.
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
        """Detect language from text content."""
        return detect_language(text, default_lang=self.fallback_language)

    def get_g2p(self, lang_key: Optional[Union[str, object]] = None):
        """Get or initialize G2P engine for the requested language."""
        if callable(lang_key):
            return lang_key, "custom"

        if not lang_key or lang_key == "auto":
            lang_key = self.fallback_language or "zh"

        if str(lang_key) in self._g2p_cache:
            return self._g2p_cache[str(lang_key)], str(lang_key)

        lang_str = str(lang_key).lower().split("-")[0]
        lang_code = LANG_CODE_MAP.get(lang_str, lang_str)

        if lang_code in self._g2p_cache:
            return self._g2p_cache[lang_code], lang_code

        engine = self._create_g2p(lang_code)
        self._g2p_cache[lang_code] = engine
        return engine, lang_code

    def _create_g2p(self, lang_code: str):
        """Initialize G2P engine by language code: prefers KPipeline, falls back to misaki."""
        try:
            from kokoro import KPipeline
            logger.info(t("log.pipeline_frontend", lang_code=lang_code))
            return KPipeline(lang_code=lang_code, model=False)
        except Exception:
            pass

        try:
            if lang_code in ("a", "b"):
                from misaki import en
                logger.info(t("log.init_en_g2p", dialect="British" if lang_code == "b" else "American"))
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
                logger.info(t("log.init_ja_g2p"))
                return ja.JAG2P()

            elif lang_code == "z":
                from misaki import zh
                logger.info(t("log.init_zh_g2p"))

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

                has_zhuyin = bool(self.vocab and any(sym in self.vocab for sym in ("ㄓ", "ㄅ", "中", "阳")))
                zh_version = "1.1" if has_zhuyin else None
                logger.info(t("log.zh_config", version=zh_version, zhuyin=has_zhuyin))

                try:
                    return zh.ZHG2P(version=zh_version, en_callable=en_word_callable)
                except (TypeError, ValueError):
                    try:
                        return zh.ZHG2P(version=None, en_callable=en_word_callable)
                    except Exception:
                        return zh.ZHG2P()

            else:
                logger.info(t("log.init_other_g2p", lang_code=lang_code))
                try:
                    from misaki import espeak
                    espeak_lang_map = {
                        "e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br"
                    }
                    lang_name = espeak_lang_map.get(lang_code, "en-us")
                    return espeak.EspeakG2P(language=lang_name)
                except Exception as e:
                    logger.warning(t("log.espeak_fallback", error=str(e)))
                    from misaki import en
                    return en.G2P(trf=False)

        except ImportError as exc:
            install_cmd = {
                "z": "pip install 'misaki[zh]'",
                "j": "pip install 'misaki[ja]'",
                "a": "pip install 'misaki[en]'",
                "b": "pip install 'misaki[en]'",
            }.get(lang_code, "pip install misaki")
            msg = f"Failed to load G2P engine for '{lang_code}', please install: {install_cmd} ({exc})"
            logger.error(msg)
            raise RuntimeError(msg) from exc

    def text_to_tokens(self, text: str, language: Optional[Union[str, object]] = None) -> np.ndarray:
        """Convert single sentence text to Kokoro token ID sequence."""
        if callable(language):
            g2p_engine = language
        else:
            g2p_engine, _ = self.get_g2p(language)

        try:
            res = g2p_engine(text)
        except Exception as e:
            logger.warning(t("log.g2p_error", text=text, error=str(e)))
            res = text

        from inspect import isgenerator
        if isgenerator(res) or hasattr(res, "__next__"):
            phoneme_parts = []
            for item in res:
                if hasattr(item, "phonemes") and item.phonemes:
                    phoneme_parts.append(str(item.phonemes))
                elif isinstance(item, tuple) and len(item) >= 2 and item[1]:
                    phoneme_parts.append(str(item[1]))
                elif isinstance(item, str):
                    phoneme_parts.append(item)
            ps = "".join(phoneme_parts)
        elif isinstance(res, tuple):
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

        active_vocab = self.vocab or DEFAULT_VOCAB
        matched_ids = [active_vocab[c] for c in ps if c in active_vocab]

        if not matched_ids:
            logger.warning(t("log.phoneme_miss", phoneme=ps, text=text))
            matched_ids = [0]

        if len(matched_ids) > 508:
            matched_ids = matched_ids[:508]

        token_ids = [0] + matched_ids + [0]
        return np.array([token_ids], dtype=np.int64)


def process_text_with_misaki(g2p, text: str, vocab: Optional[Dict[str, int]] = None, **kwargs) -> np.ndarray:
    manager = MultiLangG2P(vocab=vocab)
    manager._g2p_cache["custom"] = g2p
    return manager.text_to_tokens(text, language="custom")


# ===========================================================================
# Voice Manager & Blender
# ===========================================================================
_BLENDED_VOICES_CACHE: Dict[str, np.ndarray] = {}


def load_single_voice_tensor(file_path: Path) -> np.ndarray:
    """Load a voice feature tensor file (.bin, .pt, .pth, .npy, .npz)."""
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
            raise RuntimeError("PyTorch is required to load .pt voice files: pip install torch")

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

    raise ValueError(f"Unsupported voice format: {suffix}")


def parse_voice_blend(voice_spec: str) -> List[Tuple[str, float]]:
    """
    Parse weighted voice blend syntax (compatible with Kokoro-FastAPI):
    e.g. 'zf_xiaobei(0.7)+af_heart(0.3)' or 'zf_xiaobei+af_heart'
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
    """Retrieve or blend voice style vectors."""
    if not voices:
        raise ValueError("Voice library is empty!")

    if not voice_spec or not voice_spec.strip():
        first_key = next(iter(voices))
        return voices[first_key]

    voice_spec = voice_spec.strip()

    if voice_spec in voices:
        return voices[voice_spec]

    if voice_spec in _BLENDED_VOICES_CACHE:
        return _BLENDED_VOICES_CACHE[voice_spec]

    blend_list = parse_voice_blend(voice_spec)
    if not blend_list:
        first_key = next(iter(voices))
        return voices[first_key]

    if len(blend_list) == 1:
        v_name = blend_list[0][0]
        return voices.get(v_name, voices[next(iter(voices))])

    tensors = []
    weights = []
    for v_name, w in blend_list:
        if v_name in voices:
            tensors.append(voices[v_name])
            weights.append(w)
        else:
            logger.warning(t("log.voice_skip", voice=v_name))

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
    logger.info(t("log.blend_success", spec=voice_spec, count=len(tensors)))
    return blended


def resolve_model_and_voices(
    model_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    voices_path: Optional[str] = None,
) -> Tuple[Path, Dict[str, np.ndarray], Optional[Dict[str, int]]]:
    """Search and load Kokoro model (.xml/.onnx), voices, and vocabulary config."""
    candidates = []
    if model_dir:
        candidates.append(Path(model_dir))
    if model_path:
        candidates.append(Path(model_path).parent)

    search_roots = [p.resolve() for p in candidates if p.exists()]
    if not search_roots:
        search_roots = [Path(".").resolve()]

    target_model_file: Optional[Path] = None
    if model_path and Path(model_path).is_file():
        target_model_file = Path(model_path).resolve()
    else:
        for root in search_roots:
            if root.is_file() and root.suffix in [".xml", ".onnx"]:
                target_model_file = root
                break
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
        raise FileNotFoundError(f"OpenVINO (.xml) or ONNX model not found in {search_roots}")

    model_root = target_model_file.parent

    vocab: Optional[Dict[str, int]] = None
    config_candidates = [
        model_root / "config.json",
        model_root.parent / "config.json",
        Path("./config.json"),
    ]
    for c_path in config_candidates:
        if c_path.is_file():
            try:
                with open(c_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if "vocab" in cfg:
                        vocab = cfg["vocab"]
                        logger.info(t("log.vocab_loaded", path=str(c_path), tokens=len(vocab)))
                        break
            except Exception as e:
                logger.warning(t("log.config_error", path=str(c_path), error=str(e)))

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
                            logger.debug(t("log.voice_file_skip", name=f.name, error=str(e)))
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
                logger.warning(t("log.voice_archive_error", src=str(src), error=str(e)))

    if not voices:
        logger.warning(t("log.no_preset_voices"))
        voices["af_heart"] = np.zeros((1, 256), dtype=np.float32)

    logger.info(t("log.voices_loaded", count=len(voices), first_voice=next(iter(voices))))
    return target_model_file, voices, vocab


# ===========================================================================
# OpenVINO Kokoro Engine Wrapper
# ===========================================================================
class KokoroEngine:
    """Kokoro acoustic model inference engine powered by OpenVINO."""

    def __init__(
        self,
        model_file: Path,
        device: str = "CPU",
        threads: Optional[int] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model_file = model_file
        self.device = device.upper()
        self.threads = threads

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = model_file.parent / ".ov_cache"

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(t("log.cache_dir_info", path=str(self.cache_dir)))
        except Exception as e:
            logger.warning(f"Could not create cache directory {self.cache_dir}: {e}")
            self.cache_dir = None

        logger.info(t("log.loading_model", model=model_file.name, device=self.device))
        core = ov.Core()

        if self.cache_dir:
            try:
                core.set_property({"CACHE_DIR": str(self.cache_dir)})
            except Exception as e:
                logger.debug(f"Failed to set CACHE_DIR on Core: {e}")

        model = core.read_model(str(model_file))

        # GPU 动态维度优化：设置有界动态形状 (Bounded Dynamic Shape)
        # 声明 token 长度范围在 1..512 之间，利于 GPU 驱动预分配显存池并抑制重分配卡顿
        try:
            inputs = model.inputs
            if len(inputs) >= 1:
                token_input = inputs[0]
                pshape = token_input.partial_shape
                if pshape.rank.is_static and pshape.rank.get_length() == 2:
                    bounded_shape = ov.PartialShape([1, ov.Dimension(1, 512)])
                    model.reshape({token_input.any_name: bounded_shape})
                    logger.debug(f"Configured bounded dynamic shape for tokens: {bounded_shape}")
        except Exception as e:
            logger.debug(f"Bounded dynamic shape reshape skipped: {e}")

        ov_config: Dict[str, str] = {"PERFORMANCE_HINT": "LATENCY"}
        if self.cache_dir:
            ov_config["CACHE_DIR"] = str(self.cache_dir)

        if "GPU" in self.device:
            ov_config["INFERENCE_PRECISION_HINT"] = "f16"
            try:
                import os
                ov_config["COMPILATION_NUM_THREADS"] = str(max(1, os.cpu_count() or 4))
            except Exception:
                pass

        if threads and threads > 0:
            ov_config["INFERENCE_NUM_THREADS"] = str(threads)
            logger.info(t("log.threads_set", threads=threads))

        self.compiled_model = core.compile_model(model, self.device, config=ov_config)
        logger.info(t("log.model_ready"))

    def synthesize_tokens(
        self,
        token_ids: np.ndarray,
        style: np.ndarray,
        speed: float = 1.0,
    ) -> bytes:
        """Run acoustic model inference to produce 16-bit 24kHz PCM bytes."""
        token_len = token_ids.shape[1]

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

        try:
            outputs = self.compiled_model([token_ids, style_vec, speed_array])
        except Exception as e:
            logger.error(t("log.inference_failed", error=str(e)))
            return b""

        audio = outputs[0]
        if isinstance(audio, np.ndarray):
            audio = audio.flatten()
        else:
            audio = np.array(audio).flatten()

        audio = np.clip(audio, -1.0, 1.0)
        pcm_data = (audio * 32767.0).astype(np.int16).tobytes()
        return pcm_data


# ===========================================================================
# Wyoming Event Handler
# ===========================================================================
class TTSEventHandler(AsyncEventHandler):
    """Wyoming TTS asynchronous event handler with streaming chunk response."""

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
        max_chunk_chars: int = 65,
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
        self.max_chunk_chars = max_chunk_chars

    async def handle_event(self, event: Event) -> bool:
        """Handle incoming Wyoming protocol event."""

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            return True

        if Synthesize.is_type(event.type):
            synth = Synthesize.from_event(event)
            text = (synth.text or "").strip()

            if not text:
                logger.warning(t("log.empty_text"))
                await self.write_event(AudioStart(rate=self.sample_rate, width=2, channels=1).event())
                await self.write_event(AudioStop().event())
                return True

            explicit_voice = None
            if synth.voice and synth.voice.name:
                cand = synth.voice.name.strip()
                if cand in self.voices or "+" in cand:
                    explicit_voice = cand

            client_lang = None
            if synth.voice and synth.voice.language:
                cand_lang = synth.voice.language.split("-")[0].lower()
                if cand_lang not in ("auto", "none", ""):
                    client_lang = cand_lang

            detected_lang = detect_language(text, default_lang=self.default_language)

            if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", text):
                target_lang = "ja"
            elif client_lang:
                target_lang = client_lang
            elif explicit_voice:
                prefix = explicit_voice[0].lower()
                if prefix == "z" and re.search(r"^[a-zA-Z0-9\s\.,!\?'-]+$", text.strip()):
                    target_lang = "en"
                else:
                    target_lang = VOICE_PREFIX_TO_LANG.get(prefix, detected_lang)
            else:
                target_lang = detected_lang

            if explicit_voice:
                voice_name = explicit_voice
            else:
                target_code = LANG_CODE_MAP.get(target_lang, target_lang)
                matching_voices = [
                    v for v in self.voices.keys()
                    if v.startswith(f"{target_code}f_") or v.startswith(f"{target_code}m_")
                ]
                if matching_voices:
                    voice_name = matching_voices[0]
                else:
                    voice_name = self.default_voice_name

            req_speed = self.speed
            if event.data:
                if "speed" in event.data and isinstance(event.data["speed"], (int, float)):
                    req_speed = float(event.data["speed"])
                elif isinstance(event.data.get("voice"), dict) and "speed" in event.data["voice"]:
                    req_speed = float(event.data["voice"]["speed"])

            logger.info(
                t(
                    "log.request",
                    lang=target_lang,
                    detected=detected_lang,
                    voice=voice_name,
                    speed=req_speed,
                    text=text if len(text) <= 50 else text[:47] + "...",
                )
            )

            voice_style = get_or_blend_voice(self.voices, voice_name)

            await self.write_event(
                AudioStart(
                    rate=self.sample_rate,
                    width=2,
                    channels=1,
                ).event()
            )

            chunks = split_text_into_streaming_chunks(text, max_chunk_chars=self.max_chunk_chars)
            logger.debug(t("log.chunks_split", count=len(chunks), chunks=chunks))

            loop = asyncio.get_running_loop()
            chunk_size = 1024

            for c_idx, chunk_text in enumerate(chunks, 1):
                t_start = time.perf_counter()

                chunk_lang = target_lang
                if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", chunk_text):
                    chunk_lang = "ja"
                elif re.search(r"[\u4E00-\u9FFF]", chunk_text) and chunk_lang not in ("ja", "zh"):
                    chunk_lang = "zh"
                elif re.search(r"^[a-zA-Z0-9\s\.,!\?'-]+$", chunk_text):
                    chunk_lang = "en"

                tokens = await loop.run_in_executor(
                    None,
                    self.g2p_manager.text_to_tokens,
                    chunk_text,
                    chunk_lang,
                )

                if tokens.shape[1] <= 2:
                    continue

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
                    t(
                        "log.chunk_output",
                        index=c_idx,
                        total=len(chunks),
                        text=chunk_text[:16].strip(),
                        time=t_cost,
                        tokens=tokens.shape[1],
                        duration=len(pcm_data) / (self.sample_rate * 2),
                    )
                )

            await self.write_event(AudioStop().event())
            logger.info(t("log.synthesis_complete"))

        return True


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
        "-H",
        help=t("cli.options.host"),
    ),
    port: int = typer.Option(
        10021,
        "--port",
        "-p",
        help=t("cli.options.port"),
    ),
    device: str = typer.Option(
        "CPU",
        "--device",
        "-d",
        help=t("cli.options.device"),
    ),
    threads: Optional[int] = typer.Option(
        None,
        "--threads",
        "-t",
        help=t("cli.options.threads"),
    ),
    cache_dir: Optional[str] = typer.Option(
        None,
        "--cache-dir",
        help=t("cli.options.cache_dir"),
    ),
    max_chunk_chars: int = typer.Option(
        65,
        "--max-chunk-chars",
        "-c",
        help=t("cli.options.max_chunk_chars"),
    ),
    language: str = typer.Option(
        "auto",
        "--language",
        "-l",
        help=t("cli.options.language"),
    ),
    voice: Optional[str] = typer.Option(
        None,
        "--voice",
        "-v",
        help=t("cli.options.voice"),
    ),
    speed: float = typer.Option(
        0.85,
        "--speed",
        "-s",
        help=t("cli.options.speed"),
    ),
    sample_rate: int = typer.Option(
        24000,
        "--sample-rate",
        help=t("cli.options.sample_rate"),
    ),
) -> None:
    """Launch Wyoming Kokoro-OpenVINO ultra-fast multilingual TTS service."""

    logger.info("=" * 65)
    logger.info(t("log.starting_service"))
    logger.info("=" * 65)

    model_file, voices, vocab = resolve_model_and_voices(model_dir=model_dir)

    all_voice_names = list(voices.keys())
    selected_default_voice = voice
    if not selected_default_voice:
        candidates = (
            [v for v in all_voice_names if v.startswith("zf_") or v.startswith("zm_")]
            + [v for v in all_voice_names if v.startswith("jf_") or v.startswith("jm_")]
            + [v for v in all_voice_names if v.startswith("af_") or v.startswith("am_")]
        )
        selected_default_voice = candidates[0] if candidates else all_voice_names[0]

    logger.info(t("log.default_voice", voice=selected_default_voice, count=len(voices)))

    engine = KokoroEngine(
        model_file=model_file,
        device=device,
        threads=threads,
        cache_dir=cache_dir,
    )

    g2p_manager = MultiLangG2P(
        fallback_language="zh" if language == "auto" else language,
        vocab=vocab,
    )

    tts_attribution = Attribution(
        name="Kokoro TTS (OpenVINO)",
        url="https://github.com/remsky/Kokoro-FastAPI",
    )

    tts_voices: List[TtsVoice] = []
    for v_name in sorted(voices.keys()):
        prefix = v_name[0].lower()
        v_lang = VOICE_PREFIX_TO_LANG.get(prefix, "en")
        v_desc = f"{v_name} ({get_lang_display_name(v_lang)})"
        tts_voices.append(
            TtsVoice(
                name=v_name,
                description=v_desc,
                languages=[v_lang],
                attribution=tts_attribution,
                installed=True,
                version="2.0.0",
            )
        )

    supported_languages = sorted(list({v.languages[0] for v in tts_voices}))
    logger.info(t("log.ha_registered_langs", langs=", ".join(supported_languages)))

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="kokoro-openvino",
                description=t("wyoming.program_description"),
                attribution=tts_attribution,
                installed=True,
                voices=tts_voices,
                version="2.0.0",
            )
        ]
    )

    logger.info(t("log.warmup_start"))
    try:
        warm_tokens = g2p_manager.text_to_tokens("Hello", language="en")
        warm_style = get_or_blend_voice(voices, selected_default_voice)
        engine.synthesize_tokens(warm_tokens, warm_style, speed=1.0)

        # 针对 GPU/HETERO 执行多尺度动态序列预热，将多尺寸内核预先编译写入 CACHE_DIR 磁盘缓存中
        if "GPU" in engine.device:
            logger.info(t("log.gpu_warmup"))
            warmup_lengths = [16, 32, 48, 64, 96, 128, 160, 200]
            for wlen in warmup_lengths:
                dummy_tok = np.ones((1, wlen), dtype=np.int64)
                dummy_tok[0, 0] = 0
                dummy_tok[0, -1] = 0
                engine.synthesize_tokens(dummy_tok, warm_style, speed=1.0)
            logger.info(t("log.gpu_warmup_done"))

        logger.info(t("log.warmup_done"))
    except Exception as e:
        logger.debug(t("log.warmup_skip", error=str(e)))

    server = AsyncServer.from_uri(f"tcp://{host}:{port}")
    logger.info("=================================================================")
    logger.info(t("log.service_ready"))
    logger.info(t("log.listen_addr", host=host, port=port))
    logger.info(t("log.device_info", device=device))
    if engine.cache_dir:
        logger.info(t("log.cache_dir_info", path=str(engine.cache_dir)))
    if "HETERO" in device.upper() and "GPU" in device.upper():
        logger.info(t("log.hetero_gpu_tip"))
    logger.info(t("log.speed_info", speed=speed))
    logger.info(t("log.voice_info", voice=selected_default_voice))
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
                    max_chunk_chars=max_chunk_chars,
                    **kw,
                )
            )
        )
    except KeyboardInterrupt:
        logger.info(t("log.service_stopped"))


if __name__ == "__main__":
    app()
