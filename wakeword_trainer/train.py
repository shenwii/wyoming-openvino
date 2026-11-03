#!/usr/bin/env python3
"""
openWakeWord model training, evaluation, and NPU-optimized ONNX exporter (train.py).
=============================================================================
1. Feature extraction pipeline:
   Uses openWakeWord official melspectrogram.onnx and embedding_model.onnx to extract
   96-dim speech embeddings from augmented positive and negative audio.
   Supports hardware acceleration via OpenVINO (CPU or Intel GPU).
2. Negative sample blending:
   Fuses synthetic hard negatives with ACAV100M 2000-hour negative audio features.
3. Neural network training:
   Trains a standard openWakeWord DNN classifier on CPU, CUDA, or Intel GPU (PyTorch XPU).
4. Intel NPU dedicated optimization export:
   - Pure static shape [1, 16, 96] -> [1, 1]
   - Pure Gemm + ReLU + Sigmoid computation graph for 100% NPU hardware acceleration
   - Eliminates dynamic shape compilation crashes and CPU fallback
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml

from i18n import t

# Optional onnx and inference runtimes
try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import openvino as ov
except ImportError:
    ov = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_wakeword")

# Constants aligned with openWakeWord and wyoming-wakeword
CHUNK_SAMPLES = 1280       # 80ms frame (16000 * 0.08)
MEL_HISTORY_FRAMES = 76    # History frames required by embedding model
EMBEDDING_DIM = 96         # Speech embedding vector dimension
DEFAULT_CONTEXT_LEN = 16   # Context window frames (16 * 80ms = 1.28s)


# ==============================================================================
# Hardware Device Resolvers
# ==============================================================================
def resolve_torch_device(requested: str = "auto") -> torch.device:
    """
    Resolve PyTorch training device.
    Supports 'auto', 'cpu', 'cuda', and 'xpu' (Intel Arc / Core Ultra GPU via PyTorch 2.4+ / IPEX).
    """
    req = str(requested).lower().strip()
    if req == "auto":
        # 1. Check Intel GPU (PyTorch 2.4+ native XPU backend or IPEX)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        # 2. Check NVIDIA GPU (CUDA)
        if torch.cuda.is_available():
            return torch.device("cuda")
        # 3. Fallback to CPU
        return torch.device("cpu")
    elif req in ("xpu", "intel", "intel_gpu"):
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        logger.warning("Intel XPU requested but torch.xpu is not available. Falling back to CPU.")
        return torch.device("cpu")
    elif req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    else:
        return torch.device(req)


# ==============================================================================
# Feature Extraction Pipeline
# ==============================================================================
class FeatureExtractor:
    """Audio feature extraction pipeline using openWakeWord base ONNX models."""

    def __init__(self, models_dir: Path, requested_device: str = "auto") -> None:
        self.models_dir = models_dir
        melspec_p = models_dir / "melspectrogram.onnx"
        embedding_p = models_dir / "embedding_model.onnx"

        # Check parent models directory
        if not melspec_p.exists():
            parent_melspec = Path(__file__).resolve().parent.parent / "models" / "melspectrogram.onnx"
            if parent_melspec.exists():
                melspec_p = parent_melspec

        if not embedding_p.exists():
            parent_emb = Path(__file__).resolve().parent.parent / "models" / "embedding_model.onnx"
            if parent_emb.exists():
                embedding_p = parent_emb

        if not melspec_p.exists() or not embedding_p.exists():
            raise FileNotFoundError(
                t("train.extractor_missing", mel=melspec_p, emb=embedding_p)
            )

        req_dev = str(requested_device).upper().strip()
        self.actual_device = "CPU"

        # Prefer OpenVINO if GPU requested or available; otherwise onnxruntime or openvino
        if ov is not None and (req_dev in ("GPU", "AUTO") or ort is None):
            core = ov.Core()
            if req_dev in ("GPU", "AUTO") and "GPU" in core.available_devices:
                self.actual_device = "GPU"
            else:
                self.actual_device = "CPU"

            self.melspec_session = core.compile_model(core.read_model(str(melspec_p)), self.actual_device)
            self.emb_session = core.compile_model(core.read_model(str(embedding_p)), self.actual_device)
            self.backend = "openvino"
            emb_shape = self.emb_session.inputs[0].get_partial_shape()
        elif ort is not None:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 2
            self.melspec_session = ort.InferenceSession(str(melspec_p), opts, providers=["CPUExecutionProvider"])
            self.emb_session = ort.InferenceSession(str(embedding_p), opts, providers=["CPUExecutionProvider"])
            self.backend = "onnxruntime"
            self.emb_input_name = self.emb_session.get_inputs()[0].name
            emb_shape = self.emb_session.get_inputs()[0].shape
        elif ov is not None:
            core = ov.Core()
            self.melspec_session = core.compile_model(core.read_model(str(melspec_p)), "CPU")
            self.emb_session = core.compile_model(core.read_model(str(embedding_p)), "CPU")
            self.backend = "openvino"
            emb_shape = self.emb_session.inputs[0].get_partial_shape()
        else:
            raise RuntimeError("Please install onnxruntime or openvino (pip install onnxruntime openvino).")

        self.emb_shape = [1, MEL_HISTORY_FRAMES, 32, 1]
        try:
            if len(emb_shape) == 4 and (emb_shape[1] == 1 or str(emb_shape[1]) == "1"):
                self.emb_shape = [1, 1, MEL_HISTORY_FRAMES, 32]
        except Exception:
            pass

    def extract_features_from_audio(self, audio_16k: np.ndarray) -> np.ndarray:
        """Extract 96-dimensional embedding sequence from a 16kHz audio array."""
        num_chunks = len(audio_16k) // CHUNK_SAMPLES
        if num_chunks == 0:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        mel_history = []
        embeddings = []

        for i in range(num_chunks):
            chunk = audio_16k[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES].astype(np.float32).reshape(1, CHUNK_SAMPLES)
            if self.backend == "onnxruntime":
                inp_name = self.melspec_session.get_inputs()[0].name
                raw_mel = self.melspec_session.run(None, {inp_name: chunk})[0]
            else:
                raw_mel = self.melspec_session([chunk])[0]

            scaled_mel = (raw_mel / 10.0) + 2.0
            mel_frame = scaled_mel.reshape(-1, 32)
            for f in mel_frame:
                mel_history.append(f)

            if len(mel_history) >= MEL_HISTORY_FRAMES:
                history_window = np.array(mel_history[-MEL_HISTORY_FRAMES:], dtype=np.float32).reshape(self.emb_shape)
                if self.backend == "onnxruntime":
                    emb = self.emb_session.run(None, {self.emb_input_name: history_window})[0]
                else:
                    emb = self.emb_session([history_window])[0]
                embeddings.append(emb.reshape(EMBEDDING_DIM))

        if not embeddings:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        return np.array(embeddings, dtype=np.float32)


# ==============================================================================
# NPU-Optimized Neural Network Model (PyTorch)
# ==============================================================================
class NPUOptimizedWakeWordDNN(nn.Module):
    """
    Compact feedforward wake word classifier optimized for Intel NPU (OpenVINO VPUX).
    Topological features:
      - Pure Gemm + ReLU + Sigmoid computation graph
      - Static shape [1, 16, 96] -> [1, 1]
      - Dropout used during training, folded out during ONNX export
    """

    def __init__(self, context_length: int = 16, embedding_dim: int = 96, layer_size: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        input_dim = context_length * embedding_dim  # 16 * 96 = 1536
        self.context_length = context_length
        self.embedding_dim = embedding_dim

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, layer_size)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(layer_size, layer_size)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.out_fc = nn.Linear(layer_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.flatten(x)
        h = self.drop1(self.relu1(self.fc1(h)))
        h = self.drop2(self.relu2(self.fc2(h)))
        out = self.sigmoid(self.out_fc(h))
        return out


# ==============================================================================
# Dataset Loader
# ==============================================================================
class WakeWordDataset(Dataset):
    """Wake word training dataset combining positive, synthetic negative, and ACAV100M features."""

    def __init__(
        self,
        positive_features: List[np.ndarray],
        negative_features: List[np.ndarray],
        acav_mmap: Optional[np.ndarray] = None,
        context_len: int = 16,
        embedding_dim: int = 96,
        samples_per_epoch: int = 50000,
    ) -> None:
        self.context_len = context_len
        self.embedding_dim = embedding_dim
        self.samples_per_epoch = samples_per_epoch

        self.pos_windows = []
        for feat in positive_features:
            if len(feat) >= context_len:
                self.pos_windows.append(feat[-context_len:])
                if len(feat) >= context_len + 1:
                    self.pos_windows.append(feat[-(context_len + 1) : -1])
            elif len(feat) > 0:
                pad_len = context_len - len(feat)
                padded = np.pad(feat, ((pad_len, 0), (0, 0)), mode="constant", constant_values=0.0)
                self.pos_windows.append(padded)

        self.neg_windows = []
        for feat in negative_features:
            if len(feat) >= context_len:
                for i in range(len(feat) - context_len + 1):
                    self.neg_windows.append(feat[i : i + context_len])
            elif len(feat) > 0:
                pad_len = context_len - len(feat)
                padded = np.pad(feat, ((pad_len, 0), (0, 0)), mode="constant", constant_values=0.0)
                self.neg_windows.append(padded)

        self.acav_mmap = acav_mmap
        self.has_acav = acav_mmap is not None and len(acav_mmap) > 0

        logger.info(
            t(
                "train.dataset_stats",
                pos=len(self.pos_windows),
                neg=len(self.neg_windows),
                acav=self.has_acav,
            )
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_acav_window(self) -> np.ndarray:
        """Sample a window guaranteed to be shape (context_len, embedding_dim) from ACAV100M mmap."""
        n_samples = len(self.acav_mmap)
        idx = random.randint(0, n_samples - 1)
        raw = self.acav_mmap[idx]

        # Case 1: 3D array (N, context_len, embedding_dim), e.g. (5625000, 16, 96)
        if self.acav_mmap.ndim == 3:
            t_len = raw.shape[0]
            if t_len == self.context_len:
                window = raw
            elif t_len > self.context_len:
                offset = random.randint(0, t_len - self.context_len)
                window = raw[offset : offset + self.context_len]
            else:
                pad_len = self.context_len - t_len
                window = np.pad(raw, ((pad_len, 0), (0, 0)), mode="constant", constant_values=0.0)

        # Case 2: 2D array (N, D)
        elif self.acav_mmap.ndim == 2:
            if self.acav_mmap.shape[1] == self.embedding_dim:
                # Continuous frames along axis 0
                max_start = max(0, n_samples - self.context_len - 1)
                start = random.randint(0, max_start)
                window = self.acav_mmap[start : start + self.context_len]
                if len(window) < self.context_len:
                    window = np.pad(window, ((self.context_len - len(window), 0), (0, 0)), mode="constant")
            elif self.acav_mmap.shape[1] == self.context_len * self.embedding_dim:
                window = raw.reshape(self.context_len, self.embedding_dim)
            else:
                window = raw.reshape(self.context_len, -1)[:, : self.embedding_dim]
        else:
            window = raw.reshape(self.context_len, self.embedding_dim)

        window = np.ascontiguousarray(window)
        if window.dtype == np.int16:
            window = window.astype(np.float32) / 32768.0
        elif window.dtype != np.float32:
            window = window.astype(np.float32)

        if window.shape != (self.context_len, self.embedding_dim):
            window = window.reshape(self.context_len, self.embedding_dim)

        return window

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5 and len(self.pos_windows) > 0:
            window = random.choice(self.pos_windows)
            label = 1.0
        else:
            if self.has_acav and (len(self.neg_windows) == 0 or random.random() < 0.6):
                window = self._sample_acav_window()
            elif len(self.neg_windows) > 0:
                window = random.choice(self.neg_windows)
            elif self.has_acav:
                window = self._sample_acav_window()
            else:
                window = np.random.randn(self.context_len, self.embedding_dim).astype(np.float32) * 0.05
            label = 0.0

        window = np.ascontiguousarray(window)
        if window.dtype == np.int16:
            window = window.astype(np.float32) / 32768.0
        elif window.dtype != np.float32:
            window = window.astype(np.float32)

        if window.shape != (self.context_len, self.embedding_dim):
            window = window.reshape(self.context_len, self.embedding_dim)

        return torch.from_numpy(window).float(), torch.tensor([label], dtype=torch.float32)


# ==============================================================================
# Feature Extraction and Caching
# ==============================================================================
def extract_and_cache_features(
    audio_dir: Path,
    extractor: FeatureExtractor,
    cache_path: Path,
    desc: str,
) -> List[np.ndarray]:
    """Batch extract features from all WAV files in directory and cache to disk."""
    if cache_path.exists():
        logger.info(t("train.cache_found", name=cache_path.name))
        data = np.load(str(cache_path), allow_pickle=True)
        return list(data)

    wav_files = sorted(list(audio_dir.glob("*.wav")))
    logger.info(t("train.extracting_features", desc=desc, count=len(wav_files)))

    features_list = []
    count = 0
    for w in wav_files:
        try:
            audio, _ = sf.read(str(w), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            feats = extractor.extract_features_from_audio(audio)
            if len(feats) > 0:
                features_list.append(feats)
                count += 1
            if count % 100 == 0:
                print(f"\r  {t('train.feature_progress', count=count, total=len(wav_files))}", end="", flush=True)
        except Exception as e:
            logger.debug(f"Error extracting {w.name}: {e}")

    print()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), np.array(features_list, dtype=object), allow_pickle=True)
    logger.info(t("train.feature_saved", desc=desc, path=cache_path, count=len(features_list)))
    return features_list


# ==============================================================================
# Model Evaluation
# ==============================================================================
def evaluate_model(
    model: nn.Module,
    val_pos_features: List[np.ndarray],
    val_neg_mmap: Optional[np.ndarray],
    context_len: int = 16,
    embedding_dim: int = 96,
    cutoff: float = 0.5,
) -> Tuple[float, float]:
    """Evaluate recall and estimated False Alarms Per Hour (FAPH)."""
    model.eval()
    device = next(model.parameters()).device

    pos_correct = 0
    pos_total = 0
    with torch.no_grad():
        for feat in val_pos_features:
            if len(feat) >= context_len:
                win_np = feat[-context_len:]
            elif len(feat) > 0:
                win_np = np.pad(feat, ((context_len - len(feat), 0), (0, 0)), mode="constant")
            else:
                continue

            if win_np.shape != (context_len, embedding_dim):
                win_np = win_np.reshape(context_len, embedding_dim)

            win = torch.from_numpy(np.array(win_np, dtype=np.float32, copy=True)).unsqueeze(0).to(device)
            pred = model(win).item()
            if pred >= cutoff:
                pos_correct += 1
            pos_total += 1

    recall = (pos_correct / pos_total) if pos_total > 0 else 0.0

    false_positives = 0
    total_hours = 0.0

    if val_neg_mmap is not None and len(val_neg_mmap) > 0:
        if val_neg_mmap.ndim == 3:
            total_windows = len(val_neg_mmap)
            total_hours = (total_windows * 0.08) / 3600.0
            stride = 4
            with torch.no_grad():
                for i in range(0, total_windows, stride):
                    win_np = val_neg_mmap[i]
                    if win_np.dtype == np.int16:
                        win_np = win_np.astype(np.float32) / 32768.0
                    if win_np.shape != (context_len, embedding_dim):
                        win_np = win_np.reshape(context_len, embedding_dim)
                    win = torch.from_numpy(np.array(win_np, dtype=np.float32, copy=True)).unsqueeze(0).to(device)
                    pred = model(win).item()
                    if pred >= cutoff:
                        false_positives += 1
            faph = (false_positives * stride) / total_hours if total_hours > 0 else 0.0

        elif val_neg_mmap.ndim == 2 and len(val_neg_mmap) > context_len:
            total_frames = len(val_neg_mmap)
            total_hours = (total_frames * 0.08) / 3600.0
            stride = 4
            with torch.no_grad():
                for i in range(0, total_frames - context_len, stride):
                    win_np = val_neg_mmap[i : i + context_len]
                    if win_np.dtype == np.int16:
                        win_np = win_np.astype(np.float32) / 32768.0
                    if win_np.shape != (context_len, embedding_dim):
                        win_np = win_np.reshape(context_len, embedding_dim)
                    win = torch.from_numpy(np.array(win_np, dtype=np.float32, copy=True)).unsqueeze(0).to(device)
                    pred = model(win).item()
                    if pred >= cutoff:
                        false_positives += 1
            faph = (false_positives * stride) / total_hours if total_hours > 0 else 0.0
        else:
            faph = 0.0
    else:
        faph = 0.0

    return recall, faph


# ==============================================================================
# NPU ONNX Export & External Data Bundler
# ==============================================================================
def bundle_external_data_to_single_onnx(
    onnx_path: Path,
    output_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
) -> bool:
    """Bundle an ONNX model that references external data (.data) into a self-contained single ONNX file."""
    if onnx is None:
        logger.error("The 'onnx' package is required to bundle external data. Please install it with: pip install onnx")
        return False

    from onnx.external_data_helper import convert_model_from_external_data, load_external_data_for_model

    onnx_path = Path(onnx_path).resolve()
    if not onnx_path.exists():
        logger.error("Input ONNX file not found: %s", onnx_path)
        return False

    out_file = Path(output_path).resolve() if output_path else onnx_path
    temp_out = out_file.with_suffix(".bundled.tmp.onnx")

    try:
        logger.info("Loading ONNX model: %s", onnx_path)
        model = onnx.load(str(onnx_path), load_external_data=False)

        has_external = False
        for init in model.graph.initializer:
            if init.has_field("data_location") and init.data_location == onnx.TensorProto.EXTERNAL:
                has_external = True
                for entry in init.external_data:
                    if entry.key == "location":
                        expected_file = onnx_path.parent / entry.value
                        if not expected_file.exists():
                            if data_path and Path(data_path).exists():
                                entry.value = Path(data_path).name
                            else:
                                candidates = (
                                    list(onnx_path.parent.glob("*.data"))
                                    + list(onnx_path.parent.glob("*.tmp.onnx.data"))
                                )
                                if candidates:
                                    logger.info("Auto-mapped external data file '%s' -> '%s'", entry.value, candidates[0].name)
                                    entry.value = candidates[0].name

        if not has_external:
            logger.info("Model %s is already self-contained (no external data found).", onnx_path)
            return True

        load_external_data_for_model(model, base_dir=str(onnx_path.parent))
        convert_model_from_external_data(model)
        onnx.save_model(model, str(temp_out), save_as_external_data=False)

        verified_model = onnx.load(str(temp_out), load_external_data=False)
        onnx.checker.check_model(verified_model)
        for init in verified_model.graph.initializer:
            if init.has_field("data_location") and init.data_location == onnx.TensorProto.EXTERNAL:
                raise RuntimeError(f"Initializer '{init.name}' still references external data.")

        if out_file.exists():
            out_file.unlink()
        temp_out.rename(out_file)

        final_size_kb = out_file.stat().st_size / 1024
        logger.info("✅ Successfully bundled into single ONNX file: %s (%.1f KB)", out_file, final_size_kb)
        return True
    except Exception as exc:
        logger.error("Failed to bundle external data into single ONNX: %s", exc)
        if temp_out.exists():
            temp_out.unlink()
        return False


def export_npu_optimized_onnx(
    model: nn.Module,
    output_onnx_path: Path,
    context_length: int = 16,
    embedding_dim: int = 96,
    opset: int = 17,
) -> bool:
    """Export static ONNX model optimized for Intel NPU with strictly embedded weights."""
    import inspect

    model.eval()
    dummy_input = torch.zeros(1, context_length, embedding_dim, dtype=torch.float32)

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    temp_onnx = output_onnx_path.with_suffix(".tmp.onnx")

    logger.info(t("train.exporting_onnx", path=output_onnx_path.name))

    try:
        export_kwargs = {
            "input_names": ["input"],
            "output_names": ["output"],
            "opset_version": opset,
            "do_constant_folding": True,
            "dynamic_axes": None,  # Strictly static for NPU driver stability
        }

        # Inspect torch.onnx.export signature across PyTorch versions
        sig = inspect.signature(torch.onnx.export)
        if "export_params" in sig.parameters:
            export_kwargs["export_params"] = True
        if "dynamo" in sig.parameters:
            # Disable dynamo in PyTorch 2.5+ to ensure TorchScript embeds all weights directly
            export_kwargs["dynamo"] = False
        if "use_external_data_format" in sig.parameters:
            export_kwargs["use_external_data_format"] = False

        try:
            torch.onnx.export(model, dummy_input, str(temp_onnx), **export_kwargs)
        except TypeError:
            torch.onnx.export(
                model,
                dummy_input,
                str(temp_onnx),
                input_names=["input"],
                output_names=["output"],
                opset_version=opset,
                do_constant_folding=True,
                dynamic_axes=None,
            )

        # Guarantee all weights are embedded directly if onnx is available
        if onnx is not None:
            try:
                from onnx.external_data_helper import convert_model_from_external_data
                onnx_model = onnx.load(str(temp_onnx), load_external_data=True)
                convert_model_from_external_data(onnx_model)
                onnx.save_model(onnx_model, str(temp_onnx), save_as_external_data=False)
            except Exception as e:
                logger.debug("convert_model_from_external_data step skipped: %s", e)

            # Verification
            onnx_model = onnx.load(str(temp_onnx), load_external_data=False)
            onnx.checker.check_model(onnx_model)
            for init in onnx_model.graph.initializer:
                if init.has_field("data_location") and init.data_location == onnx.TensorProto.EXTERNAL:
                    raise RuntimeError(
                        f"ONNX model still references external data for '{init.name}'. "
                        "All weights must be embedded into the single .onnx file."
                    )
            logger.info(t("train.onnx_verified"))

        # Clean up any external data files created during export
        for ext_f in temp_onnx.parent.glob(f"{temp_onnx.name}*.data*"):
            try:
                ext_f.unlink()
            except OSError:
                pass

        test_inp = np.random.randn(1, context_length, embedding_dim).astype(np.float32)
        with torch.no_grad():
            torch_out = model(torch.from_numpy(test_inp)).cpu().numpy()

        if ort is not None:
            sess = ort.InferenceSession(str(temp_onnx), providers=["CPUExecutionProvider"])
            ort_out = sess.run(None, {"input": test_inp})[0]
            np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-4)
            logger.info(t("train.onnx_numerical_match"))

        if output_onnx_path.exists():
            output_onnx_path.unlink()
        temp_onnx.rename(output_onnx_path)

        final_size_kb = output_onnx_path.stat().st_size / 1024
        if final_size_kb < 50:
            logger.warning(
                "Exported ONNX file size is unusually small (%.1f KB). "
                "Expected ~400 KB. Model weights may not be embedded correctly!",
                final_size_kb,
            )

        logger.info(t("train.export_success", path=output_onnx_path, size=final_size_kb))
        return True

    except Exception as exc:
        logger.error(t("train.export_failed", error=str(exc)))
        if temp_onnx.exists():
            temp_onnx.unlink()
        for ext_f in temp_onnx.parent.glob(f"{temp_onnx.name}*.data*"):
            try:
                ext_f.unlink()
            except OSError:
                pass
        return False


def save_model_artifacts(
    model: nn.Module,
    output_dir: Path,
    model_name: str,
    cfg: dict,
    context_length: int,
    embedding_dim: int,
    opset: int,
) -> Tuple[Path, Path, Path]:
    """Save PyTorch weights (.pt), export static ONNX model, and write metadata JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final_onnx_path = output_dir / f"{model_name}.onnx"
    pt_path = output_dir / f"{model_name}.pt"
    info_path = output_dir / f"{model_name}.json"

    # 1. Save PyTorch weights (.pt) for subsequent fast re-export or fine-tuning
    torch.save(model.state_dict(), str(pt_path))
    logger.info("Saved PyTorch weights: %s", pt_path)

    # 2. Export NPU static ONNX model
    export_npu_optimized_onnx(
        model=model,
        output_onnx_path=final_onnx_path,
        context_length=context_length,
        embedding_dim=embedding_dim,
        opset=opset,
    )

    # 3. Write model metadata JSON
    phrase = cfg.get("wakeword", {}).get("phrase")
    lang = cfg.get("wakeword", {}).get("language", "zh")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "model_type": "openWakeWord_DNN_NPU",
                "onnx_file": f"{model_name}.onnx",
                "weights_file": f"{model_name}.pt",
                "phrase": phrase,
                "language": lang,
                "context_length": context_length,
                "embedding_dim": embedding_dim,
                "recommended_cutoff": 0.55,
                "recommended_refractory": 2.0,
                "npu_compatible": True,
                "input_shape": [1, context_length, embedding_dim],
                "output_shape": [1, 1],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n======================================================================")
    print(t("train.congrats_title"))
    print(f"   - ONNX: {final_onnx_path}")
    print(f"   - Weights: {pt_path}")
    print(f"   - Metadata: {info_path}")
    print(f"{t('train.wyoming_usage')}")
    print(f"   python wyoming-wakeword/wyoming-wakeword.py --model {final_onnx_path} --device NPU --cutoff 0.55")
    print("======================================================================\n")

    return final_onnx_path, pt_path, info_path


# ==============================================================================
# Main Training Function
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description=t("cli.train_help"))
    parser.add_argument("--config", default="config.yaml", help="Path to config file [default: config.yaml]")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", default=None, help="PyTorch training device (auto, cpu, cuda, xpu)")
    parser.add_argument("--feature-device", default=None, help="Feature extraction device (auto, CPU, GPU)")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip data loading and training; directly export ONNX and metadata from existing weights (.pt)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained PyTorch weights (.pt) for export [default: output/{model_name}.pt]",
    )
    parser.add_argument(
        "--bundle-external-data",
        type=str,
        default=None,
        help="Path to an ONNX model that references external .data files to bundle into a single self-contained .onnx file",
    )
    parser.add_argument(
        "--external-data-file",
        type=str,
        default=None,
        help="Optional path to external .data file if separated from the ONNX file",
    )
    parser.add_argument(
        "--output-onnx",
        type=str,
        default=None,
        help="Optional destination path for bundled single ONNX file",
    )
    args = parser.parse_args()

    # Fast path: Bundle external data into a single self-contained ONNX file
    if args.bundle_external_data:
        success = bundle_external_data_to_single_onnx(
            onnx_path=Path(args.bundle_external_data),
            output_path=Path(args.output_onnx) if args.output_onnx else None,
            data_path=Path(args.external_data_file) if args.external_data_file else None,
        )
        sys.exit(0 if success else 1)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        cfg_path = Path(__file__).resolve().parent / "config.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / cfg["data"].get("data_dir", "./data")).resolve()
    train_cfg = cfg["training"]

    # Model name prioritization: wakeword.friend_name -> wakeword.friendly_name -> training.model_name
    model_name = (
        cfg.get("wakeword", {}).get("friend_name")
        or cfg.get("wakeword", {}).get("friendly_name")
        or train_cfg.get("model_name")
        or "my_wakeword"
    )

    context_length = int(train_cfg.get("context_length", DEFAULT_CONTEXT_LEN))
    embedding_dim = int(train_cfg.get("embedding_dim", EMBEDDING_DIM))
    layer_size = int(train_cfg.get("layer_size", 64))
    steps = args.steps or int(train_cfg.get("steps", 20000))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 128))
    learning_rate = args.lr or float(train_cfg.get("learning_rate", 0.001))
    dropout = float(train_cfg.get("dropout", 0.1))
    npu_cfg = train_cfg.get("npu_optimization", {})
    opset = int(npu_cfg.get("onnx_opset", 17))

    # Resolve training and feature extraction hardware devices
    train_dev_req = args.device or train_cfg.get("device", "auto")
    train_device = resolve_torch_device(train_dev_req)

    feat_dev_req = args.feature_device or train_cfg.get("feature_device", "auto")

    output_dir = (base_dir / train_cfg.get("output_dir", "./output")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_onnx_path = output_dir / f"{model_name}.onnx"

    print("======================================================================")
    print(t("train.header_title"))
    print(t("train.target_model", name=model_name))
    print(t("train.context_window", window=context_length, sec=context_length * 0.08))
    print(t("train.hyperparams", units=layer_size, steps=steps, batch_size=batch_size))
    print(t("train.training_device", device=str(train_device).upper()))
    print(t("train.output_path", path=final_onnx_path))
    print("======================================================================")

    # --------------------------------------------------------------------------
    # Fast path: Export-only mode (skip data loading and training)
    # --------------------------------------------------------------------------
    if args.export_only:
        ckpt_path = Path(args.checkpoint).resolve() if args.checkpoint else output_dir / f"{model_name}.pt"
        if not ckpt_path.exists():
            logger.error("Checkpoint file not found: %s. Cannot perform --export-only without saved weights (.pt).", ckpt_path)
            sys.exit(1)

        logger.info("Loading weights from %s ...", ckpt_path)
        model = NPUOptimizedWakeWordDNN(
            context_length=context_length,
            embedding_dim=embedding_dim,
            layer_size=layer_size,
            dropout=0.0,
        )
        state_dict = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(state_dict)

        save_model_artifacts(
            model=model,
            output_dir=output_dir,
            model_name=model_name,
            cfg=cfg,
            context_length=context_length,
            embedding_dim=embedding_dim,
            opset=opset,
        )
        return

    # 1. Initialize feature extractor
    models_dir = (base_dir / "lib" / "models").resolve()
    extractor = FeatureExtractor(models_dir=models_dir, requested_device=feat_dev_req)
    print(t("train.feature_device", device=f"{extractor.backend.upper()} [{extractor.actual_device}]"))

    # 2. Extract and load positive and hard negative features
    aug_pos_dir = data_dir / "augmented" / "positive"
    aug_neg_dir = data_dir / "augmented" / "negative"

    if not aug_pos_dir.exists() or len(list(aug_pos_dir.glob("*.wav"))) == 0:
        aug_pos_dir = data_dir / "synthetic_speech" / "positive"

    if not aug_neg_dir.exists() or len(list(aug_neg_dir.glob("*.wav"))) == 0:
        aug_neg_dir = data_dir / "synthetic_speech" / "negative"

    cache_dir = data_dir / "features_cache"
    pos_features = extract_and_cache_features(aug_pos_dir, extractor, cache_dir / "pos_feats.npy", "positive")
    neg_features = extract_and_cache_features(aug_neg_dir, extractor, cache_dir / "neg_feats.npy", "negative") if aug_neg_dir.exists() else []

    # 3. Memory map ACAV100M negative features
    acav_path = data_dir / "negative_features" / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    acav_mmap = None
    if acav_path.exists():
        try:
            logger.info(t("train.acav_loading", name=acav_path.name))
            acav_mmap = np.load(str(acav_path), mmap_mode="r")
            logger.info(t("train.acav_mapped", count=len(acav_mmap)))
            logger.info("  -> ACAV100M array shape: %s, dtype: %s", acav_mmap.shape, acav_mmap.dtype)
        except Exception as e:
            logger.warning(t("train.acav_failed", error=str(e)))

    # 4. Load validation set features
    val_path = data_dir / "negative_features" / "validation_set_features.npy"
    val_mmap = None
    if val_path.exists():
        try:
            val_mmap = np.load(str(val_path), mmap_mode="r")
            logger.info(t("train.validation_loaded", count=len(val_mmap)))
            logger.info("  -> Validation array shape: %s, dtype: %s", val_mmap.shape, val_mmap.dtype)
        except Exception:
            pass

    # 5. Build Dataset and DataLoader
    dataset = WakeWordDataset(
        positive_features=pos_features,
        negative_features=neg_features,
        acav_mmap=acav_mmap,
        context_len=context_length,
        embedding_dim=embedding_dim,
        samples_per_epoch=steps * batch_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(train_device.type in ("cuda", "xpu")),
    )

    # 6. Initialize neural network model
    model = NPUOptimizedWakeWordDNN(
        context_length=context_length,
        embedding_dim=embedding_dim,
        layer_size=layer_size,
        dropout=dropout,
    ).to(train_device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    logger.info(t("train.start_training", steps=steps))
    start_time = time.time()
    running_loss = 0.0

    model.train()
    loader_iter = iter(loader)

    for step in range(1, steps + 1):
        try:
            x_batch, y_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x_batch, y_batch = next(loader_iter)

        x_batch = x_batch.to(train_device)
        y_batch = y_batch.to(train_device)

        optimizer.zero_grad()
        preds = model(x_batch)

        weights = torch.where(y_batch == 0.0, 2.5, 1.0)
        loss = torch.mean(weights * ((preds - y_batch) ** 2))
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if step % 200 == 0 or step == steps:
            avg_loss = running_loss / 200.0 if step % 200 == 0 else running_loss / (step % 200)
            running_loss = 0.0
            print(
                f"\r  {t('train.step_progress', step=step, steps=steps, loss=avg_loss, lr=scheduler.get_last_lr()[0])}",
                end="",
                flush=True,
            )

        if step % 2000 == 0 or step == steps:
            recall, faph = evaluate_model(
                model,
                pos_features,
                val_mmap,
                context_len=context_length,
                embedding_dim=embedding_dim,
                cutoff=0.5,
            )
            print(f"\n  {t('train.val_stats', step=step, recall=recall * 100, faph=faph)}")
            model.train()

    print()
    elapsed = time.time() - start_time
    logger.info(t("train.training_complete", elapsed=elapsed))

    # 7. Save weights, export NPU static ONNX model, and write metadata
    save_model_artifacts(
        model=model,
        output_dir=output_dir,
        model_name=model_name,
        cfg=cfg,
        context_length=context_length,
        embedding_dim=embedding_dim,
        opset=opset,
    )


if __name__ == "__main__":
    main()
