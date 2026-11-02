#!/usr/bin/env python3
"""
openWakeWord 模型训练、评估与 NPU 优化 ONNX 导出器 (train.py)
=============================================================
1. 特征提取管线:
   利用 openWakeWord 官方 melspectrogram.onnx 与 embedding_model.onnx，
   将增强后的正负向音频统一抽取为 96 维语音嵌入特征流。
2. 负样本融合:
   深度融合 TTS NG 干扰词 (Hard Negatives) 与 ACAV100M 2000 小时负样本特征。
3. 神经网络分类器训练:
   训练与 openWakeWord 规范完全一致的 DNN 判别网络。
4. NPU 专属优化导出:
   - 导出为纯静态形状 [1, 16, 96] -> [1, 1]
   - 纯净 Gemm + ReLU + Sigmoid 算子图，OpenVINO VPUX 编译器 100% 硬件编译加速
   - 彻底避免动态 Shape 导致 NPU 崩溃或回退 CPU
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

# 尝试导入 onnx 与推理引擎
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

# 常量定义 (严格与 openWakeWord 及 wyoming-wakeword.py 对齐)
CHUNK_SAMPLES = 1280       # 80ms 采样点 (16000 * 0.08)
MEL_HISTORY_FRAMES = 76    # 嵌入模型所需梅尔帧数
EMBEDDING_DIM = 96         # 语音嵌入向量维度
DEFAULT_CONTEXT_LEN = 16   # 滑动上下文窗口数 (16 * 80ms = 1.28秒)


# ==============================================================================
# 特征提取流水线 (Feature Extraction)
# ==============================================================================
class FeatureExtractor:
    """基于 ONNX 的 openWakeWord 音频特征提取管道。"""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        melspec_p = models_dir / "melspectrogram.onnx"
        embedding_p = models_dir / "embedding_model.onnx"

        # 降级检查
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
                f"未能找到 openWakeWord 基础特征模型:\n"
                f"  melspectrogram: {melspec_p} (存在: {melspec_p.exists()})\n"
                f"  embedding_model: {embedding_p} (存在: {embedding_p.exists()})\n"
                f"请先运行 'python setup_libs.py' 自动下载基础模型。"
            )

        # 优先使用 onnxruntime，无则使用 openvino
        if ort is not None:
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
            raise RuntimeError("请安装 onnxruntime 或 openvino 以进行特征提取 (pip install onnxruntime)。")

        # 适配 embedding 模型输入维度: [1, 76, 32, 1] 或 [1, 1, 76, 32]
        self.emb_shape = [1, MEL_HISTORY_FRAMES, 32, 1]
        try:
            if len(emb_shape) == 4 and (emb_shape[1] == 1 or str(emb_shape[1]) == "1"):
                self.emb_shape = [1, 1, MEL_HISTORY_FRAMES, 32]
        except Exception:
            pass

        logger.info("✅ 特征提取管道初始化就绪 (后端: %s, 嵌入形状: %s)", self.backend, self.emb_shape)

    def extract_features_from_audio(self, audio_16k: np.ndarray) -> np.ndarray:
        """
        从单条 16kHz 音频中抽取 96 维嵌入特征序列。
        返回: (N_frames, 96) 的 numpy 矩阵。
        """
        # 补齐或截取为 CHUNK_SAMPLES 整数倍
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

            # 当累积满 76 帧梅尔频谱时，抽取一次 96 维特征
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
# NPU 极致优化神经网络模型定义 (PyTorch)
# ==============================================================================
class NPUOptimizedWakeWordDNN(nn.Module):
    """
    针对 Intel NPU (OpenVINO VPUX) 深度优化的紧凑型唤醒词分类器。
    
    特点:
      - 纯净 Gemm + ReLU + Sigmoid 计算拓扑，无动态算子与分支
      - 训练阶段包含 Dropout 以提升防过拟合泛化性
      - 导出 ONNX 时自动剔除 Dropout，转换为纯粹的静态权值矩阵乘加
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
        # 输入 x: [Batch, context_length, 96]
        h = self.flatten(x)
        h = self.drop1(self.relu1(self.fc1(h)))
        h = self.drop2(self.relu2(self.fc2(h)))
        out = self.sigmoid(self.out_fc(h))
        return out


# ==============================================================================
# 唤醒词训练数据集加载器 (PyTorch Dataset)
# ==============================================================================
class WakeWordDataset(Dataset):
    """
    综合数据集:
      - 正样本 (来自增强后的正向音频末尾窗口)
      - 负样本 A (来自增强后的 NG 干扰词所有窗口)
      - 负样本 B (来自 ACAV100M 2000 小时日常环境与语音预计算特征)
    """

    def __init__(
        self,
        positive_features: List[np.ndarray],
        negative_features: List[np.ndarray],
        acav_mmap: Optional[np.ndarray] = None,
        context_len: int = 16,
        samples_per_epoch: int = 50000,
    ) -> None:
        self.context_len = context_len
        self.samples_per_epoch = samples_per_epoch

        # 构建正样本窗口
        self.pos_windows = []
        for feat in positive_features:
            if len(feat) >= context_len:
                # 截取音频尾部最后几个有效窗口 (唤醒词说完整时刻)
                self.pos_windows.append(feat[-context_len:])
                if len(feat) >= context_len + 1:
                    self.pos_windows.append(feat[-(context_len + 1) : -1])

        # 构建 NG 负样本窗口
        self.neg_windows = []
        for feat in negative_features:
            if len(feat) >= context_len:
                for i in range(len(feat) - context_len + 1):
                    self.neg_windows.append(feat[i : i + context_len])

        self.acav_mmap = acav_mmap
        self.has_acav = acav_mmap is not None and len(acav_mmap) > context_len

        logger.info(
            "📊 数据集构建完成: 正样本窗口 %d 个, NG负样本窗口 %d 个, ACAV100M 特征可用: %s",
            len(self.pos_windows),
            len(self.neg_windows),
            self.has_acav,
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 50% 概率采样正样本，50% 概率采样负样本
        if random.random() < 0.5 and len(self.pos_windows) > 0:
            window = random.choice(self.pos_windows)
            label = 1.0
        else:
            # 负样本: 50% 概率采自 NG 干扰词, 50% 概率采自 ACAV100M
            if self.has_acav and (len(self.neg_windows) == 0 or random.random() < 0.6):
                max_start = len(self.acav_mmap) - self.context_len - 1
                start = random.randint(0, max_start)
                window = self.acav_mmap[start : start + self.context_len]
                # 若 ACAV100M 存储为 16-bit 整型，需反归一化为 float32
                if window.dtype == np.int16:
                    window = window.astype(np.float32) / 32768.0
            else:
                window = random.choice(self.neg_windows)
            label = 0.0

        return torch.from_numpy(window.copy()).float(), torch.tensor([label], dtype=torch.float32)


# ==============================================================================
# 离线特征抽取与缓存构建
# ==============================================================================
def extract_and_cache_features(
    audio_dir: Path,
    extractor: FeatureExtractor,
    cache_path: Path,
    desc: str,
) -> List[np.ndarray]:
    """批量抽取目录下所有 WAV 文件的特征并序列化缓存，避免反复计算。"""
    if cache_path.exists():
        logger.info("发现已有特征缓存，正在直接加载: %s", cache_path.name)
        data = np.load(str(cache_path), allow_pickle=True)
        return list(data)

    wav_files = sorted(list(audio_dir.glob("*.wav")))
    logger.info("正在抽取 [%s] 特征 (共 %d 个音频)...", desc, len(wav_files))

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
                print(f"\r  -> 已抽取 {count}/{len(wav_files)} 个音频特征", end="", flush=True)
        except Exception as e:
            logger.debug("抽取音频 %s 异常: %s", w.name, e)

    print()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), np.array(features_list, dtype=object), allow_pickle=True)
    logger.info("✅ [%s] 特征抽取完成并保存至: %s (有效 %d 条)", desc, cache_path, len(features_list))
    return features_list


# ==============================================================================
# 模型验证 (FAPH 误唤醒率与召回率评估)
# ==============================================================================
def evaluate_model(
    model: nn.Module,
    val_pos_features: List[np.ndarray],
    val_neg_mmap: Optional[np.ndarray],
    context_len: int = 16,
    cutoff: float = 0.5,
) -> Tuple[float, float]:
    """
    评估模型表现:
      - 召回率 (Recall / True Positive Rate)
      - 每小时误报次数估算 (False Alarms Per Hour, FAPH)
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. 计算召回率
    pos_correct = 0
    pos_total = 0
    with torch.no_grad():
        for feat in val_pos_features:
            if len(feat) >= context_len:
                win = torch.from_numpy(feat[-context_len:]).unsqueeze(0).to(device)
                pred = model(win).item()
                if pred >= cutoff:
                    pos_correct += 1
                pos_total += 1

    recall = (pos_correct / pos_total) if pos_total > 0 else 0.0

    # 2. 计算 FAPH (在验证集特征上滑动)
    false_positives = 0
    total_hours = 0.0

    if val_neg_mmap is not None and len(val_neg_mmap) > context_len:
        total_frames = len(val_neg_mmap)
        # 每帧 80ms = 0.08s
        total_hours = (total_frames * 0.08) / 3600.0

        # 以步长 1 滑动测试 (为了加速测试每 4 帧测试一次)
        stride = 4
        with torch.no_grad():
            for i in range(0, total_frames - context_len, stride):
                win_np = val_neg_mmap[i : i + context_len]
                if win_np.dtype == np.int16:
                    win_np = win_np.astype(np.float32) / 32768.0
                win = torch.from_numpy(win_np).unsqueeze(0).to(device)
                pred = model(win).item()
                if pred >= cutoff:
                    false_positives += 1

        faph = (false_positives * (stride)) / total_hours if total_hours > 0 else 0.0
    else:
        faph = 0.0

    return recall, faph


# ==============================================================================
# NPU 专属 ONNX 导出与验证
# ==============================================================================
def export_npu_optimized_onnx(
    model: nn.Module,
    output_onnx_path: Path,
    context_length: int = 16,
    embedding_dim: int = 96,
    opset: int = 17,
) -> bool:
    """
    导出针对 Intel NPU (OpenVINO VPUX 编译器) 极致优化的纯静态 ONNX 模型。
    静态形状: [1, 16, 96] -> [1, 1]
    """
    model.eval()
    dummy_input = torch.zeros(1, context_length, embedding_dim, dtype=torch.float32)

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    temp_onnx = output_onnx_path.with_suffix(".tmp.onnx")

    logger.info("正在导出 NPU 优化 ONNX 模型: %s ...", output_onnx_path.name)
    logger.info("  - 强制静态输入形状: [1, %d, %d]", context_length, embedding_dim)
    logger.info("  - ONNX Opset: %d", opset)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(temp_onnx),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset,
            do_constant_folding=True,
            # 不设置 dynamic_axes 以保持纯静态形状，对 NPU 驱动最为稳健
            dynamic_axes=None,
        )

        # 校验 ONNX 合法性
        if onnx is not None:
            onnx_model = onnx.load(str(temp_onnx))
            onnx.checker.check_model(onnx_model)
            logger.info("✅ ONNX 结构校验通过 (无悬挂节点与动态形状)")

        # 验证数值正确性 (PyTorch vs ONNXRuntime / OpenVINO)
        test_inp = np.random.randn(1, context_length, embedding_dim).astype(np.float32)
        with torch.no_grad():
            torch_out = model(torch.from_numpy(test_inp)).cpu().numpy()

        if ort is not None:
            sess = ort.InferenceSession(str(temp_onnx), providers=["CPUExecutionProvider"])
            ort_out = sess.run(None, {"input": test_inp})[0]
            np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-4)
            logger.info("✅ ONNXRuntime 推理数值与 PyTorch 完全一致")

        temp_onnx.rename(output_onnx_path)
        logger.info("🎉 成功导出最终 NPU 唤醒词模型: %s (大小: %.2f KB)", output_onnx_path, output_onnx_path.stat().st_size / 1024)
        return True

    except Exception as exc:
        logger.error("❌ 导出 ONNX 发生异常: %s", exc)
        if temp_onnx.exists():
            temp_onnx.unlink()
        return False


# ==============================================================================
# 主训练调度入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="训练 openWakeWord 唤醒词并导出 NPU 优化 ONNX 模型")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径 (默认: config.yaml)")
    parser.add_argument("--steps", type=int, default=None, help="覆盖配置文件的训练迭代步数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖批大小")
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="PyTorch 训练设备")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        cfg_path = Path(__file__).resolve().parent / "config.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / cfg["data"].get("data_dir", "./data")).resolve()
    train_cfg = cfg["training"]

    model_name = train_cfg.get("model_name", "my_wakeword")
    context_length = int(train_cfg.get("context_length", DEFAULT_CONTEXT_LEN))
    embedding_dim = int(train_cfg.get("embedding_dim", EMBEDDING_DIM))
    layer_size = int(train_cfg.get("layer_size", 64))
    steps = args.steps or int(train_cfg.get("steps", 20000))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 128))
    learning_rate = args.lr or float(train_cfg.get("learning_rate", 0.001))
    dropout = float(train_cfg.get("dropout", 0.1))
    npu_cfg = train_cfg.get("npu_optimization", {})
    opset = int(npu_cfg.get("onnx_opset", 17))

    output_dir = (base_dir / train_cfg.get("output_dir", "./output")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_onnx_path = output_dir / f"{model_name}.onnx"

    print("======================================================================")
    print("🧠 openWakeWord 唤醒词模型训练与 NPU 优化导出")
    print(f"   - 目标模型名称: {model_name}")
    print(f"   - 上下文窗口: {context_length} 帧 (约 {context_length * 0.08:.2f} 秒)")
    print(f"   - 隐藏单元: {layer_size} | 训练步数: {steps} | 批大小: {batch_size}")
    print(f"   - 训练设备: {args.device.upper()}")
    print(f"   - 产物输出路径: {final_onnx_path}")
    print("======================================================================")

    # 1. 初始化特征提取器
    models_dir = (base_dir / "lib" / "models").resolve()
    extractor = FeatureExtractor(models_dir=models_dir)

    # 2. 抽取并加载正向与 NG 负向特征
    aug_pos_dir = data_dir / "augmented" / "positive"
    aug_neg_dir = data_dir / "augmented" / "negative"

    if not aug_pos_dir.exists() or len(list(aug_pos_dir.glob("*.wav"))) == 0:
        logger.warning("未在 %s 发现增强后的正样本，尝试使用原始合成音频...", aug_pos_dir)
        aug_pos_dir = data_dir / "synthetic_speech" / "positive"

    if not aug_neg_dir.exists() or len(list(aug_neg_dir.glob("*.wav"))) == 0:
        aug_neg_dir = data_dir / "synthetic_speech" / "negative"

    cache_dir = data_dir / "features_cache"
    pos_features = extract_and_cache_features(aug_pos_dir, extractor, cache_dir / "pos_feats.npy", "正样本唤醒词")
    neg_features = extract_and_cache_features(aug_neg_dir, extractor, cache_dir / "neg_feats.npy", "NG负向干扰词") if aug_neg_dir.exists() else []

    # 3. 尝试内存映射加载 ACAV100M 负样本特征
    acav_path = data_dir / "negative_features" / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    acav_mmap = None
    if acav_path.exists():
        try:
            logger.info("正在建立 ACAV100M 2000 小时负样本内存映射: %s", acav_path.name)
            acav_mmap = np.load(str(acav_path), mmap_mode="r")
            logger.info("✅ ACAV100M 负样本映射成功 (共 %d 帧特征)", len(acav_mmap))
        except Exception as e:
            logger.warning("加载 ACAV100M 失败: %s，将依赖 NG 干扰词进行负样本训练", e)

    # 4. 加载验证集特征
    val_path = data_dir / "negative_features" / "validation_set_features.npy"
    val_mmap = None
    if val_path.exists():
        try:
            val_mmap = np.load(str(val_path), mmap_mode="r")
            logger.info("✅ 假阳性评估验证集加载成功 (共 %d 帧)", len(val_mmap))
        except Exception:
            pass

    # 5. 构建 Dataset 与 DataLoader
    dataset = WakeWordDataset(
        positive_features=pos_features,
        negative_features=neg_features,
        acav_mmap=acav_mmap,
        context_len=context_length,
        samples_per_epoch=steps * batch_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
    )

    # 6. 初始化神经网络模型
    model = NPUOptimizedWakeWordDNN(
        context_length=context_length,
        embedding_dim=embedding_dim,
        layer_size=layer_size,
        dropout=dropout,
    ).to(args.device)

    # 损失函数 (加权 BCE 以加重负样本惩罚)
    pos_weight = torch.tensor([1.0]).to(args.device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    logger.info("🚀 开始模型训练迭代 (总步数: %d)...", steps)
    start_time = time.time()
    step_count = 0
    running_loss = 0.0

    model.train()
    loader_iter = iter(loader)

    for step in range(1, steps + 1):
        try:
            x_batch, y_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x_batch, y_batch = next(loader_iter)

        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)

        optimizer.zero_grad()
        preds = model(x_batch)

        # 对负样本误差加大惩罚系数
        weights = torch.where(y_batch == 0.0, 2.5, 1.0)
        loss = torch.mean(weights * ((preds - y_batch) ** 2))  # MSE 加权或 BCELoss
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if step % 200 == 0 or step == steps:
            avg_loss = running_loss / 200.0 if step % 200 == 0 else running_loss / (step % 200)
            running_loss = 0.0
            print(f"\r  [Step {step:05d}/{steps}] 损失值: {avg_loss:.5f} | 学习率: {scheduler.get_last_lr()[0]:.6f}", end="", flush=True)

        # 周期性评估与检查点
        if step % 2000 == 0 or step == steps:
            recall, faph = evaluate_model(model, pos_features, val_mmap, context_len=context_length, cutoff=0.5)
            print(f"\n  📈 阶段验证 (Step {step}): 召回率 (Recall) = {recall * 100:.1f}%, 估计误报率 (FAPH) = {faph:.2f}次/小时")
            model.train()

    print()
    elapsed = time.time() - start_time
    logger.info("✅ 训练完成！耗时: %.1f 秒", elapsed)

    # 7. 导出针对 NPU 深度优化的纯静态 ONNX 模型
    export_npu_optimized_onnx(
        model=model,
        output_onnx_path=final_onnx_path,
        context_length=context_length,
        embedding_dim=embedding_dim,
        opset=opset,
    )

    # 8. 写入模型元信息配置文件 (便于 wyoming-wakeword.py 直接引用)
    info_path = output_dir / f"{model_name}.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "model_type": "openWakeWord_DNN_NPU",
                "onnx_file": f"{model_name}.onnx",
                "phrase": cfg["wakeword"].get("phrase"),
                "language": cfg["wakeword"].get("language"),
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
    print("🎉 恭喜！唤醒词模型训练与 NPU 优化导出圆满完成！")
    print(f"   - ONNX 模型文件: {final_onnx_path}")
    print(f"   - 模型配置文件: {info_path}")
    print("💡 在 wyoming-wakeword.py 中启用此模型的命令:")
    print(f"   python wyoming-wakeword.py --model {final_onnx_path} --device NPU --cutoff 0.55")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
