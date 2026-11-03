#!/usr/bin/env python3
"""
Wyoming Wake Word Detection Service
===================================
Low-latency, high-concurrency wake word detection service based on OpenVINO
acceleration and openWakeWord models.
Designed for Home Assistant voice assistant pipelines via the standard Wyoming protocol.

Core Features:
    1. Full Hardware Acceleration:
       - Deep optimization for Intel NPU (Neural Processing Unit in Core Ultra, etc.)
       - Automatic static shape reshaping required by NPU
       - Smart fallback: NPU -> GPU -> CPU
    2. Multi-Model Concurrency:
       - Concurrent multi-wake-word listening (e.g. 'Hey Jarvis', 'Alexa', etc.)
       - Shared audio feature extraction pipeline (MelSpectrogram + Speech Embedding)
    3. Flexible Configuration:
       - YAML / JSON config file support
       - Per-model cutoff, refractory period, and device overrides
       - Pure CLI mode without config file
    4. Standard Wyoming Protocol:
       - Full support for Describe, Detect, AudioChunk, AudioStop events
       - Home Assistant auto-discovery and model list exposition

Usage Examples:
    # 1. Launch with YAML config (recommended)
    python wyoming-wakeword.py --config wakeword_config.yaml

    # 2. Launch directly with single model on NPU
    python wyoming-wakeword.py --model models/hey_jarvis_v0.1.onnx --device NPU --cutoff 0.6

    # 3. Automatically load all wake word models in directory
    python wyoming-wakeword.py --model-dir ./models --device NPU
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import urllib.request

import numpy as np
import openvino as ov
import typer

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, WakeModel, WakeProgram
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.wake import Detect, Detection, NotDetected

from i18n import t

# Optional YAML support
try:
    import yaml
except ImportError:
    yaml = None

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger("wyoming-wakeword")


def configure_logging(level_name: str) -> None:
    """Configure global logging level."""
    numeric_level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger().setLevel(numeric_level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(numeric_level)
    logger.setLevel(numeric_level)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000          # 16kHz
CHUNK_SAMPLES = 1280         # openWakeWord standard frame size (80ms = 1280 samples)
CHUNK_BYTES = CHUNK_SAMPLES * 2  # 16-bit linear PCM single channel (2 bytes/sample)
MEL_HISTORY_FRAMES = 76      # MelSpectrogram history frames required by embedding model
EMBEDDING_DIM = 96           # Speech embedding vector dimension
DEFAULT_CONTEXT_LEN = 16     # Classifier context embedding frames (16 * 80ms = 1.28s)

# Default openWakeWord official feature extraction model URLs
DEFAULT_MELSPEC_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx"
DEFAULT_EMBEDDING_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"


# ===========================================================================
# Configuration Data Structures
# ===========================================================================
@dataclass
class SingleModelConfig:
    """Configuration for a single wake word model."""
    name: str
    model_path: str
    phrase: Optional[str] = None
    cutoff: Optional[float] = None
    refractory: Optional[float] = None
    trigger_level: Optional[int] = None
    sliding_window_size: Optional[int] = None
    device: Optional[str] = None
    languages: List[str] = field(default_factory=lambda: ["en"])


@dataclass
class WakewordServerConfig:
    """Global server configuration for wake word detection."""
    host: str = "0.0.0.0"
    port: int = 10400
    log_level: str = "INFO"
    default_device: str = "NPU"
    default_cutoff: float = 0.5
    default_refractory: float = 2.0
    default_trigger_level: int = 1
    default_sliding_window_size: int = 1
    melspec_model_path: Optional[str] = None
    melspec_device: Optional[str] = None
    embedding_model_path: Optional[str] = None
    embedding_device: Optional[str] = None
    models: List[SingleModelConfig] = field(default_factory=list)


# ===========================================================================
# Model Path Resolution and Automatic Downloading
# ===========================================================================
def find_or_download_feature_model(
    model_name: str,
    custom_path: Optional[str],
    download_url: str,
    target_dir: Path,
) -> Path:
    """
    Locate or download openWakeWord base feature models (melspectrogram / embedding_model).

    Search order:
        1. User-specified path
        2. target_dir / model_name
        3. openwakeword python package bundled models
        4. Automatic download from official GitHub Releases
    """
    if custom_path:
        p = Path(custom_path).expanduser().resolve()
        if p.exists():
            return p
        logger.warning(t("log.custom_path_missing", path=custom_path))

    local_path = target_dir / model_name
    if local_path.exists():
        return local_path

    try:
        import openwakeword
        pkg_model_path = Path(openwakeword.__file__).parent / "resources" / "models" / model_name
        if pkg_model_path.exists():
            logger.info(t("log.found_in_pkg", path=pkg_model_path))
            return pkg_model_path
    except ImportError:
        pass

    logger.info(t("log.download_start", model=model_name))
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(t("log.download_url", url=download_url, path=local_path))
    urllib.request.urlretrieve(download_url, str(local_path))
    logger.info(t("log.download_success", model=model_name))
    return local_path


# ===========================================================================
# OpenVINO Model Runner with Hardware Fallback & NPU Reshape
# ===========================================================================
class OpenVINOModelRunner:
    """Compiles and executes OpenVINO models with hardware fallback chain (NPU -> GPU -> CPU)."""

    def __init__(self, core: ov.Core) -> None:
        self.core = core

    def compile_model_safe(
        self,
        model_path: Union[str, Path],
        requested_device: str,
        static_shape: Optional[List[int]] = None,
        input_name: Optional[str] = None,
    ) -> Tuple[ov.CompiledModel, str]:
        """Read and safely compile OpenVINO model with NPU static reshaping and device fallback."""
        path_str = str(model_path)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Model file not found: {path_str}")

        logger.debug(t("log.read_model", path=path_str))
        model = self.core.read_model(path_str)

        # Ensure static shape for NPU execution
        is_npu = "NPU" in requested_device.upper()
        if is_npu or static_shape is not None:
            self._ensure_static_shape(model, static_shape, input_name)

        device_chain = [requested_device]
        if "NPU" in requested_device.upper():
            device_chain.extend(["GPU", "CPU"])
        elif "GPU" in requested_device.upper():
            device_chain.append("CPU")

        candidate_devices = []
        for d in device_chain:
            if d not in candidate_devices:
                candidate_devices.append(d)

        last_error = None
        for dev in candidate_devices:
            try:
                try:
                    self.core.set_property(dev, {"PERFORMANCE_HINT": "LATENCY"})
                except Exception:
                    pass

                logger.info(t("log.compiling_model", device=dev, name=os.path.basename(path_str)))
                compiled = self.core.compile_model(model, dev)
                logger.info(t("log.compile_success", device=dev, name=os.path.basename(path_str)))
                return compiled, dev
            except Exception as e:
                logger.warning(t("log.compile_failed", device=dev, error=str(e)))
                last_error = e

        raise RuntimeError(
            f"Model {path_str} failed compilation on all candidate devices: {candidate_devices}"
        ) from last_error

    @staticmethod
    def _ensure_static_shape(
        model: ov.Model,
        forced_shape: Optional[List[int]] = None,
        input_name: Optional[str] = None,
    ) -> None:
        """Ensure input nodes have static shape for NPU compatibility."""
        reshape_dict = {}

        for input_node in model.inputs:
            name = input_name if input_name else input_node.get_any_name()
            pshape = input_node.get_partial_shape()

            if forced_shape:
                reshape_dict[name] = forced_shape
            elif pshape.is_dynamic:
                static_dims = []
                for dim in pshape:
                    if dim.is_dynamic:
                        static_dims.append(1)
                    else:
                        static_dims.append(dim.get_length())
                reshape_dict[name] = static_dims

        if reshape_dict:
            logger.info(t("log.npu_reshape", shapes=reshape_dict))
            try:
                model.reshape(reshape_dict)
            except Exception as e:
                logger.warning(t("log.npu_reshape_failed", error=str(e)))


# ===========================================================================
# Active Wake Word Classifier
# ===========================================================================
class LoadedClassifier:
    """Loaded and compiled wake word classifier instance."""

    def __init__(
        self,
        config: SingleModelConfig,
        compiled_model: ov.CompiledModel,
        actual_device: str,
        context_len: int = DEFAULT_CONTEXT_LEN,
        is_end_to_end: bool = False,
        effective_cutoff: float = 0.5,
        effective_refractory: float = 2.0,
        trigger_level: int = 1,
        sliding_window_size: int = 1,
    ) -> None:
        self.config = config
        self.compiled_model = compiled_model
        self.actual_device = actual_device
        self.context_len = context_len
        self.is_end_to_end = is_end_to_end
        self._effective_cutoff = effective_cutoff
        self._effective_refractory = effective_refractory
        self._trigger_level = max(1, int(trigger_level))
        self._sliding_window_size = max(1, int(sliding_window_size))
        self.last_triggered_time: float = 0.0

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def phrase(self) -> str:
        return self.config.phrase or self.config.name

    @property
    def cutoff(self) -> float:
        return self._effective_cutoff

    @cutoff.setter
    def cutoff(self, value: float) -> None:
        self._effective_cutoff = value

    @property
    def trigger_level(self) -> int:
        return self._trigger_level

    @trigger_level.setter
    def trigger_level(self, value: int) -> None:
        self._trigger_level = max(1, int(value))

    @property
    def sliding_window_size(self) -> int:
        return self._sliding_window_size

    @sliding_window_size.setter
    def sliding_window_size(self, value: int) -> None:
        self._sliding_window_size = max(1, int(value))

    @property
    def refractory(self) -> float:
        return self._effective_refractory

    @refractory.setter
    def refractory(self, value: float) -> None:
        self._effective_refractory = value

    def predict(self, feature_input: np.ndarray) -> float:
        """Execute inference and return wake probability score (0.0 to 1.0)."""
        out = self.compiled_model([feature_input])[0]
        if out.ndim >= 2 and out.shape[-1] == 2:
            return float(out[0, 1])
        return float(out.flat[0])

    def check_trigger(self, score: float, current_time: float, current_hits: int = 1) -> bool:
        """Check if score meets threshold and adheres to trigger level & refractory period."""
        if score >= self.cutoff and current_hits >= self.trigger_level:
            if current_time - self.last_triggered_time >= self.refractory:
                self.last_triggered_time = current_time
                return True
            else:
                logger.debug(
                    t(
                        "log.in_refractory",
                        name=self.name,
                        score=score,
                        remaining=self.refractory - (current_time - self.last_triggered_time),
                    )
                )
        return False


# ===========================================================================
# Shared Audio Feature Pipeline (MelSpectrogram + Speech Embedding)
# ===========================================================================
class AudioFeaturePipeline:
    """Pipeline transforming 16kHz PCM audio slices to 96-dim Speech Embeddings."""

    def __init__(
        self,
        runner: OpenVINOModelRunner,
        melspec_path: Union[str, Path],
        embedding_path: Union[str, Path],
        default_device: str = "NPU",
        melspec_device: Optional[str] = None,
        embedding_device: Optional[str] = None,
    ) -> None:
        logger.info(t("log.init_feature_pipeline"))

        target_melspec_dev = melspec_device or default_device
        self.melspec_model, self.melspec_device = runner.compile_model_safe(
            model_path=melspec_path,
            requested_device=target_melspec_dev,
            static_shape=[1, CHUNK_SAMPLES],
        )

        raw_emb_model = runner.core.read_model(str(embedding_path))
        emb_shape = [1, MEL_HISTORY_FRAMES, 32, 1]
        for inp in raw_emb_model.inputs:
            ps = inp.get_partial_shape()
            if len(ps) == 4 and ps[1].is_static and ps[1].get_length() == 1:
                emb_shape = [1, 1, MEL_HISTORY_FRAMES, 32]
                break

        self.emb_shape = emb_shape
        target_embedding_dev = embedding_device or default_device
        self.embedding_model, self.embedding_device = runner.compile_model_safe(
            model_path=embedding_path,
            requested_device=target_embedding_dev,
            static_shape=emb_shape,
        )

        logger.info(
            t(
                "log.feature_pipeline_ready",
                melspec_dev=self.melspec_device,
                embedding_dev=self.embedding_device,
            )
        )

    def compute_melspectrogram(self, audio_1280_samples: np.ndarray) -> np.ndarray:
        """Compute MelSpectrogram for 1280 samples (~80ms). Formula: (melspec / 10.0) + 2.0."""
        inp = audio_1280_samples.astype(np.float32).reshape(1, CHUNK_SAMPLES)
        raw_mel = self.melspec_model([inp])[0]
        scaled_mel = (raw_mel / 10.0) + 2.0
        return scaled_mel.reshape(-1, 32)

    def compute_embedding(self, mel_frames_76: np.ndarray) -> np.ndarray:
        """Compute 96-dim speech embedding vector from 76 mel frames."""
        inp = mel_frames_76.astype(np.float32).reshape(self.emb_shape)
        emb = self.embedding_model([inp])[0]
        return emb.reshape(EMBEDDING_DIM)


# ===========================================================================
# Wake Word Engine Manager
# ===========================================================================
class WakeWordEngine:
    """Coordinates multi-model wake word detection and feature extraction."""

    def __init__(self, config: WakewordServerConfig) -> None:
        self.config = config
        self.core = ov.Core()
        self.runner = OpenVINOModelRunner(self.core)

        models_dir = Path("models").resolve()
        models_dir.mkdir(parents=True, exist_ok=True)

        has_embedding_based_models = any(
            not m.model_path.endswith("_e2e.onnx") for m in config.models
        )

        self.feature_pipeline: Optional[AudioFeaturePipeline] = None
        if has_embedding_based_models:
            melspec_p = find_or_download_feature_model(
                "melspectrogram.onnx",
                config.melspec_model_path,
                DEFAULT_MELSPEC_URL,
                models_dir,
            )
            embedding_p = find_or_download_feature_model(
                "embedding_model.onnx",
                config.embedding_model_path,
                DEFAULT_EMBEDDING_URL,
                models_dir,
            )
            self.feature_pipeline = AudioFeaturePipeline(
                runner=self.runner,
                melspec_path=melspec_p,
                embedding_path=embedding_p,
                default_device=config.default_device,
                melspec_device=config.melspec_device,
                embedding_device=config.embedding_device,
            )

        self.classifiers: Dict[str, LoadedClassifier] = {}
        for m_cfg in config.models:
            self._load_classifier(m_cfg)

        if not self.classifiers:
            logger.warning(t("log.no_models_loaded"))

    def _load_classifier(self, m_cfg: SingleModelConfig) -> None:
        """Load, compile, and register a single wake word classifier."""
        device = m_cfg.device or self.config.default_device
        cutoff = m_cfg.cutoff if m_cfg.cutoff is not None else self.config.default_cutoff
        refractory = m_cfg.refractory if m_cfg.refractory is not None else self.config.default_refractory
        trigger_level = m_cfg.trigger_level if m_cfg.trigger_level is not None else self.config.default_trigger_level
        sliding_window_size = m_cfg.sliding_window_size if m_cfg.sliding_window_size is not None else self.config.default_sliding_window_size

        path = Path(m_cfg.model_path).expanduser().resolve()
        if not path.exists():
            candidate = Path("models") / m_cfg.model_path
            if candidate.exists():
                path = candidate.resolve()
            else:
                raise FileNotFoundError(f"Wake word model not found: {m_cfg.model_path}")

        try:
            temp_model = self.core.read_model(str(path))
        except Exception as exc:
            err_msg = str(exc)
            if "externally stored data" in err_msg or ".data" in err_msg:
                raise RuntimeError(
                    f"模型 '{m_cfg.name}' ({path}) 加载失败: 该 ONNX 模型的权重保存在外部数据文件 (.data) 中且文件缺失或脱节。"
                    f"Wyoming 服务需要自包含的单文件 ONNX 模型。请使用 wakeword_trainer 的嵌入导出功能或工具重新打包为单文件 ONNX。"
                ) from exc
            raise
        inp = temp_model.inputs[0]
        inp_shape = inp.get_partial_shape()

        is_e2e = False
        context_len = DEFAULT_CONTEXT_LEN
        static_shape = None

        if len(inp_shape) == 3 and inp_shape[-1] == EMBEDDING_DIM:
            if inp_shape[1].is_static:
                context_len = inp_shape[1].get_length()
            static_shape = [1, context_len, EMBEDDING_DIM]
            logger.info(t("log.model_embedding_based", name=m_cfg.name, window=context_len))
        elif len(inp_shape) == 3 and inp_shape[-1] in (32, 40):
            raise ValueError(
                f"Model '{m_cfg.name}' input shape {inp_shape} is mel-spectrogram based (e.g. microWakeWord). "
                f"This service requires openWakeWord classifiers with input shape [1, 16, 96]."
            )
        elif len(inp_shape) == 2:
            is_e2e = True
            e2e_samples = inp_shape[1].get_length() if inp_shape[1].is_static else CHUNK_SAMPLES
            static_shape = [1, e2e_samples]
            logger.info(t("log.model_e2e_based", name=m_cfg.name, samples=e2e_samples))
        else:
            raise ValueError(f"Unsupported wake word input shape: {inp_shape} ({path})")

        compiled, actual_dev = self.runner.compile_model_safe(
            model_path=path,
            requested_device=device,
            static_shape=static_shape,
        )

        classifier = LoadedClassifier(
            config=m_cfg,
            compiled_model=compiled,
            actual_device=actual_dev,
            context_len=context_len,
            is_end_to_end=is_e2e,
            effective_cutoff=cutoff,
            effective_refractory=refractory,
            trigger_level=trigger_level,
            sliding_window_size=sliding_window_size,
        )
        self.classifiers[m_cfg.name] = classifier
        logger.info(
            t(
                "log.model_registered",
                name=classifier.name,
                phrase=classifier.phrase,
                cutoff=classifier.cutoff,
                trigger_level=classifier.trigger_level,
                sliding_window_size=classifier.sliding_window_size,
                refractory=classifier.refractory,
                device=classifier.actual_device,
            )
        )


# ===========================================================================
# Wyoming Event Handler
# ===========================================================================
class OpenVINOWakewordHandler(AsyncEventHandler):
    """Processes Wyoming client audio streams and dispatches wake events."""

    def __init__(
        self,
        *args,
        engine: WakeWordEngine,
        wyoming_info: Info,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.wyoming_info = wyoming_info

        self.active_models: List[LoadedClassifier] = list(engine.classifiers.values())
        self.raw_audio_buffer = bytearray()
        self.melspec_buffer = np.zeros((MEL_HISTORY_FRAMES, 32), dtype=np.float32)
        self.embedding_buffer = np.zeros((DEFAULT_CONTEXT_LEN, EMBEDDING_DIM), dtype=np.float32)
        self.model_hits: Dict[str, int] = defaultdict(int)
        self.score_windows: Dict[str, deque] = {}
        self.has_detected = False

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming protocol events."""

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            return True

        if Detect.is_type(event.type):
            detect = Detect.from_event(event)
            self._reset_session()

            if detect.names:
                filtered = [
                    self.engine.classifiers[n]
                    for n in detect.names
                    if n in self.engine.classifiers
                ]
                if filtered:
                    self.active_models = filtered
                    logger.debug(t("log.client_filter", models=[m.name for m in filtered]))
                else:
                    logger.warning(t("log.client_filter_none", names=detect.names))
                    self.active_models = list(self.engine.classifiers.values())
            else:
                self.active_models = list(self.engine.classifiers.values())

            return True

        if AudioStart.is_type(event.type):
            self._reset_session()
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.raw_audio_buffer.extend(chunk.audio)

            while len(self.raw_audio_buffer) >= CHUNK_BYTES:
                chunk_slice = self.raw_audio_buffer[:CHUNK_BYTES]
                del self.raw_audio_buffer[:CHUNK_BYTES]

                audio_samples = np.frombuffer(chunk_slice, dtype=np.int16)

                detected_model = None
                try:
                    detected_model = await self._process_frame(audio_samples)
                except Exception as exc:
                    logger.error(t("log.inference_error", error=str(exc)), exc_info=True)
                if detected_model:
                    self.has_detected = True
                    logger.info(
                        t(
                            "log.wake_detected",
                            name=detected_model.name,
                            phrase=detected_model.phrase,
                            timestamp=chunk.timestamp,
                        )
                    )
                    await self.write_event(
                        Detection(
                            name=detected_model.name,
                            timestamp=chunk.timestamp,
                        ).event()
                    )
                    self._on_detected_cooldown()

            return True

        if AudioStop.is_type(event.type):
            if not self.has_detected:
                logger.debug(t("log.not_detected"))
                await self.write_event(NotDetected().event())
            self._reset_session()
            return True

        return True

    def _reset_session(self) -> None:
        """Reset session streaming buffers."""
        self.raw_audio_buffer.clear()
        self.melspec_buffer.fill(0)
        self.embedding_buffer.fill(0)
        self.model_hits.clear()
        for dq in self.score_windows.values():
            dq.clear()
        self.has_detected = False

    def _on_detected_cooldown(self) -> None:
        """Clear embedding buffer after positive detection to prevent immediate duplicate triggers."""
        self.embedding_buffer.fill(0)

    async def _process_frame(self, audio_samples: np.ndarray) -> Optional[LoadedClassifier]:
        """Process a single frame and evaluate across active classifiers."""
        pipeline = self.engine.feature_pipeline
        now = time.time()

        if pipeline:
            new_mels = pipeline.compute_melspectrogram(audio_samples)
            n_new = len(new_mels)
            if n_new >= MEL_HISTORY_FRAMES:
                self.melspec_buffer[:] = new_mels[-MEL_HISTORY_FRAMES:]
            else:
                self.melspec_buffer = np.roll(self.melspec_buffer, -n_new, axis=0)
                self.melspec_buffer[-n_new:] = new_mels

            embedding = pipeline.compute_embedding(self.melspec_buffer)
            self.embedding_buffer = np.roll(self.embedding_buffer, -1, axis=0)
            self.embedding_buffer[-1] = embedding

        for model in self.active_models:
            score = 0.0
            if model.is_end_to_end:
                float_audio = audio_samples.astype(np.float32).reshape(1, -1)
                score = model.predict(float_audio)
            else:
                if len(self.embedding_buffer) >= model.context_len:
                    emb_window = self.embedding_buffer[-model.context_len:]
                    inp_emb = emb_window.reshape(1, model.context_len, EMBEDDING_DIM)
                    score = model.predict(inp_emb)

            if model.name not in self.score_windows or self.score_windows[model.name].maxlen != model.sliding_window_size:
                self.score_windows[model.name] = deque(maxlen=model.sliding_window_size)

            dq = self.score_windows[model.name]
            dq.append(score)

            # Compute sliding window mean average if window size > 1
            if model.sliding_window_size > 1 and len(dq) > 0:
                eval_score = float(sum(dq) / len(dq))
            else:
                eval_score = score

            if eval_score >= model.cutoff:
                self.model_hits[model.name] += 1
            else:
                self.model_hits[model.name] = 0

            hits = self.model_hits[model.name]
            if score > 0.05 or eval_score > 0.05 or hits > 0:
                logger.debug(
                    "Model [%s] score: %.4f (win_avg: %.4f, cutoff: %.2f, hits: %d/%d, win: %d/%d)",
                    model.name, score, eval_score, model.cutoff, hits, model.trigger_level, len(dq), model.sliding_window_size
                )

            if model.check_trigger(eval_score, now, current_hits=hits):
                self.model_hits[model.name] = 0
                dq.clear()
                return model

        return None


# ===========================================================================
# Configuration File Parser
# ===========================================================================
def _safe_float(val, default: float) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val, default: int) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def load_config_file(config_path: Path) -> WakewordServerConfig:
    """Load server and model settings from YAML or JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    ext = config_path.suffix.lower()
    raw_dict = {}

    with open(config_path, "r", encoding="utf-8") as f:
        if ext in [".yaml", ".yml"]:
            if yaml is None:
                raise RuntimeError("pyyaml is required to parse YAML files: pip install pyyaml")
            raw_dict = yaml.safe_load(f) or {}
        elif ext == ".json":
            raw_dict = json.load(f) or {}
        else:
            raise ValueError(f"Unsupported config file extension: {ext}")

    server_sec = raw_dict.get("server") or {}
    feat_sec = raw_dict.get("feature_models") or {}

    host = server_sec.get("host") or raw_dict.get("host") or "0.0.0.0"
    port_val = server_sec.get("port") or raw_dict.get("port") or 10400
    try:
        port = int(port_val)
    except (ValueError, TypeError):
        port = 10400

    log_level = (
        server_sec.get("log_level")
        or server_sec.get("log-level")
        or raw_dict.get("log_level")
        or raw_dict.get("log-level")
        or "INFO"
    )

    default_device = (
        raw_dict.get("default_device")
        or raw_dict.get("device")
        or "NPU"
    )

    default_cutoff = _safe_float(
        raw_dict.get("default_cutoff") or raw_dict.get("cutoff"), 0.5
    )
    default_refractory = _safe_float(
        raw_dict.get("default_refractory") or raw_dict.get("refractory"), 2.0
    )
    default_trigger_level = _safe_int(
        raw_dict.get("default_trigger_level") or raw_dict.get("trigger_level"), 1
    )
    default_sliding_window_size = _safe_int(
        raw_dict.get("default_sliding_window_size")
        or raw_dict.get("sliding_window_size")
        or raw_dict.get("window_size"),
        1,
    )

    melspec_model = feat_sec.get("melspec_model") or raw_dict.get("melspec_model")
    melspec_device = feat_sec.get("melspec_device") or raw_dict.get("melspec_device")
    embedding_model = feat_sec.get("embedding_model") or raw_dict.get("embedding_model")
    embedding_device = feat_sec.get("embedding_device") or raw_dict.get("embedding_device")

    cfg = WakewordServerConfig(
        host=str(host),
        port=port,
        log_level=str(log_level).upper(),
        default_device=str(default_device),
        default_cutoff=default_cutoff,
        default_refractory=default_refractory,
        default_trigger_level=default_trigger_level,
        default_sliding_window_size=default_sliding_window_size,
        melspec_model_path=str(melspec_model) if melspec_model else None,
        melspec_device=str(melspec_device) if melspec_device else None,
        embedding_model_path=str(embedding_model) if embedding_model else None,
        embedding_device=str(embedding_device) if embedding_device else None,
    )

    raw_models = raw_dict.get("models") or []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        m_name = item.get("name")
        m_path = item.get("model_path")
        if not m_name or not m_path:
            logger.warning(t("log.invalid_model_config", item=item))
            continue

        cutoff_raw = item.get("cutoff")
        refractory_raw = item.get("refractory")
        trigger_level_raw = item.get("trigger_level")
        sliding_window_size_raw = item.get("sliding_window_size") or item.get("window_size")

        model_entry = SingleModelConfig(
            name=str(m_name),
            model_path=str(m_path),
            phrase=item.get("phrase", str(m_name).replace("_", " ").title()),
            cutoff=_safe_float(cutoff_raw, cfg.default_cutoff) if cutoff_raw is not None else None,
            refractory=_safe_float(refractory_raw, cfg.default_refractory) if refractory_raw is not None else None,
            trigger_level=_safe_int(trigger_level_raw, cfg.default_trigger_level) if trigger_level_raw is not None else None,
            sliding_window_size=_safe_int(sliding_window_size_raw, cfg.default_sliding_window_size) if sliding_window_size_raw is not None else None,
            device=item.get("device"),
            languages=item.get("languages", ["en"]),
        )
        cfg.models.append(model_entry)

    return cfg


# ===========================================================================
# CLI Application
# ===========================================================================
app = typer.Typer(help=t("cli.help"))


@app.command()
def main(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help=t("cli.options.config"),
    ),
    model: Optional[List[str]] = typer.Option(
        None,
        "--model",
        "-m",
        help=t("cli.options.model"),
    ),
    model_dir: Optional[Path] = typer.Option(
        None,
        "--model-dir",
        help=t("cli.options.model_dir"),
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help=t("cli.options.device"),
    ),
    cutoff: Optional[float] = typer.Option(
        None,
        "--cutoff",
        help=t("cli.options.cutoff"),
    ),
    trigger_level: Optional[int] = typer.Option(
        None,
        "--trigger-level",
        "--trigger_level",
        help=t("cli.options.trigger_level"),
    ),
    sliding_window_size: Optional[int] = typer.Option(
        None,
        "--sliding-window-size",
        "--sliding_window_size",
        "--window-size",
        help=t("cli.options.sliding_window_size"),
    ),
    refractory: Optional[float] = typer.Option(
        None,
        "--refractory",
        help=t("cli.options.refractory"),
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help=t("cli.options.host"),
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        "-p",
        help=t("cli.options.port"),
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help=t("cli.options.log_level"),
    ),
    melspec_model: Optional[str] = typer.Option(
        None,
        "--melspec-model",
        help=t("cli.options.melspec_model"),
    ),
    embedding_model: Optional[str] = typer.Option(
        None,
        "--embedding-model",
        help=t("cli.options.embedding_model"),
    ),
) -> None:
    """Launch OpenVINO-based multi-model wake word detection service."""

    init_log_level = log_level or "INFO"
    configure_logging(init_log_level)

    if config:
        if not config.exists():
            logger.error(t("log.config_not_found", path=config))
            sys.exit(1)
        logger.info(t("log.config_loaded", path=config))
        server_config = load_config_file(config)
    else:
        server_config = WakewordServerConfig()

    if host is not None:
        server_config.host = host
    if port is not None:
        server_config.port = port
    if log_level is not None:
        server_config.log_level = log_level.upper()
    if device is not None:
        server_config.default_device = device
    if cutoff is not None:
        server_config.default_cutoff = cutoff
    if trigger_level is not None:
        server_config.default_trigger_level = max(1, trigger_level)
    if sliding_window_size is not None:
        server_config.default_sliding_window_size = max(1, sliding_window_size)
    if refractory is not None:
        server_config.default_refractory = refractory
    if melspec_model is not None:
        server_config.melspec_model_path = melspec_model
    if embedding_model is not None:
        server_config.embedding_model_path = embedding_model

    configure_logging(server_config.log_level)
    logger.debug(t("log.log_level_active", level=server_config.log_level))

    if model:
        for m_path in model:
            base_name = Path(m_path).stem
            clean_name = base_name.split("_v")[0] if "_v" in base_name else base_name
            entry = SingleModelConfig(
                name=clean_name,
                model_path=m_path,
                phrase=clean_name.replace("_", " ").title(),
                cutoff=cutoff,
                refractory=refractory,
                trigger_level=trigger_level,
                sliding_window_size=sliding_window_size,
                device=device,
            )
            existing_idx = next(
                (i for i, m in enumerate(server_config.models) if m.name == clean_name),
                None,
            )
            if existing_idx is not None:
                logger.info(t("log.cmd_override_config", name=clean_name))
                server_config.models[existing_idx] = entry
            else:
                server_config.models.append(entry)

    if model_dir and model_dir.is_dir():
        logger.info(t("log.scan_dir", path=model_dir))
        for ext in ("*.onnx", "*.tflite", "*.xml"):
            for file_p in sorted(model_dir.glob(ext)):
                if file_p.stem.startswith("melspectrogram") or file_p.stem.startswith("embedding_model"):
                    continue
                clean_name = file_p.stem.split("_v")[0] if "_v" in file_p.stem else file_p.stem
                if not any(m.name == clean_name for m in server_config.models):
                    server_config.models.append(
                        SingleModelConfig(
                            name=clean_name,
                            model_path=str(file_p),
                            phrase=clean_name.replace("_", " ").title(),
                            cutoff=cutoff,
                            refractory=refractory,
                            trigger_level=trigger_level,
                            sliding_window_size=sliding_window_size,
                            device=device,
                        )
                    )

    if not server_config.models:
        logger.error(t("log.no_models_specified"))
        sys.exit(1)

    engine = WakeWordEngine(server_config)

    wake_models = [
        WakeModel(
            name=m.name,
            description=t("wyoming.model_description", phrase=m.phrase, device=m.actual_device),
            phrase=m.phrase,
            languages=m.config.languages,
            attribution=Attribution(
                name="openWakeWord / OpenVINO",
                url="https://github.com/dscripka/openWakeWord",
            ),
            installed=True,
            version="1.0.0",
        )
        for m in engine.classifiers.values()
    ]

    wyoming_info = Info(
        wake=[
            WakeProgram(
                name="wyoming-openvino-wakeword",
                description=t("wyoming.program_description"),
                attribution=Attribution(
                    name="Wyoming OpenVINO",
                    url="https://github.com/wyoming-openvino",
                ),
                installed=True,
                version="1.0.0",
                models=wake_models,
            )
        ]
    )

    server_uri = f"tcp://{server_config.host}:{server_config.port}"
    server = AsyncServer.from_uri(server_uri)
    logger.info("=================================================================")
    logger.info(t("log.service_started", uri=server_uri))
    logger.info(t("log.device_strategy", device=server_config.default_device))
    logger.info(t("log.active_list_header", count=len(engine.classifiers)))
    for m in engine.classifiers.values():
        logger.info(
            t(
                "log.active_list_item",
                name=m.name,
                phrase=m.phrase,
                cutoff=m.cutoff,
                refractory=m.refractory,
                device=m.actual_device,
            )
        )
    logger.info("=================================================================")

    async def run_server():
        await server.run(
            lambda *args, **kwargs: OpenVINOWakewordHandler(
                *args,
                engine=engine,
                wyoming_info=wyoming_info,
                **kwargs,
            )
        )

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info(t("log.shutdown"))


if __name__ == "__main__":
    app()
