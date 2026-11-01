#!/usr/bin/env python3
"""
Wyoming Wake Word (唤醒词检测) 服务
====================================
基于 OpenVINO 加速与 openWakeWord 模型的低延迟、高并发唤醒词检测服务。
专为 Home Assistant 语音助手生态打造，通过标准 Wyoming 协议实现无缝对接。

核心特性:
    1. 硬件全加速:
       - 深度优化支持 Intel NPU (神经网络处理单元，如 Core Ultra / Meteor Lake / Lunar Lake 等)
       - 自动处理 NPU 严格要求的静态形状重塑 (Static Shape Reshape)
       - 智能回退机制: 优先 NPU，失败时自动平滑回退至 GPU 或 CPU
    2. 多唤醒词并发支持 (Multi-Model):
       - 支持同时加载并并发检测多个不同的唤醒词 (例如同时监听 'Hey Jarvis', 'Alexa', '小爱同学' 等)
       - 共享音频特征提取管线 (MelSpectrogram + Speech Embedding)，降低多模型同时运行的算力开销
    3. 灵活的配置管理:
       - 支持 YAML / JSON 配置文件定义完整的服务器参数与模型列表
       - 支持为每个唤醒词单独自定义 cutoff (置信度阈值)、refractory (冷却时间)、device (推理设备)
       - 亦支持纯命令行参数极速启动 (无配置文件模式)
    4. 标准 Wyoming 协议兼容:
       - 完整支持 Describe, Detect, AudioChunk, AudioStop 等协议事件
       - 支持 Home Assistant 自动发现与模型列表呈现

使用示例:
    # 方式一: 使用 YAML 配置文件启动多模型检测 (推荐)
    python wyoming-wakeword.py --config wakeword_config.yaml

    # 方式二: 命令行直接指定模型并优先使用 NPU
    python wyoming-wakeword.py --model models/hey_jarvis_v0.1.onnx --device NPU --cutoff 0.6

    # 方式三: 自动加载指定目录下的所有唤醒词模型
    python wyoming-wakeword.py --model-dir ./models --device NPU
"""

import asyncio
from collections import deque
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

from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, WakeModel, WakeProgram
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.wake import Detect, Detection, NotDetected

# 可选 YAML 支持
try:
    import yaml
except ImportError:
    yaml = None

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logger = logging.getLogger("wyoming-wakeword")

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000          # 16kHz
CHUNK_SAMPLES = 1280         # openWakeWord 标准帧大小 (80ms = 1280 采样点)
CHUNK_BYTES = CHUNK_SAMPLES * 2  # 16-bit 线性 PCM 单通道 (2 字节/采样)
MEL_HISTORY_FRAMES = 76      # 嵌入模型所需梅尔频谱历史帧数
EMBEDDING_DIM = 96           # 语音嵌入向量维度
DEFAULT_CONTEXT_LEN = 16     # 分类器上下文嵌入帧数 (16 * 80ms = 1.28 秒)

# 默认 openWakeWord 官方特征提取模型下载链接
DEFAULT_MELSPEC_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx"
DEFAULT_EMBEDDING_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"


# ===========================================================================
# 配置数据结构
# ===========================================================================
@dataclass
class SingleModelConfig:
    """单个唤醒词模型的配置项。"""
    name: str
    model_path: str
    phrase: Optional[str] = None
    cutoff: float = 0.5
    refractory: float = 2.0
    device: Optional[str] = None
    languages: List[str] = field(default_factory=lambda: ["en"])


@dataclass
class WakewordServerConfig:
    """唤醒词服务端全局配置。"""
    host: str = "0.0.0.0"
    port: int = 10400
    log_level: str = "INFO"
    default_device: str = "NPU"
    default_cutoff: float = 0.5
    default_refractory: float = 2.0
    melspec_model_path: Optional[str] = None
    embedding_model_path: Optional[str] = None
    models: List[SingleModelConfig] = field(default_factory=list)


# ===========================================================================
# 辅助函数: 模型下载与路径查找
# ===========================================================================
def find_or_download_feature_model(
    model_name: str,
    custom_path: Optional[str],
    download_url: str,
    target_dir: Path,
) -> Path:
    """
    定位或自动下载 openWakeWord 基础特征模型 (melspectrogram / embedding_model)。

    查找顺序:
        1. 用户明确指定的路径
        2. target_dir / model_name
        3. openwakeword pip 包内置资源目录 (若已安装)
        4. 尝试从 GitHub Releases 自动下载
    """
    if custom_path:
        p = Path(custom_path).expanduser().resolve()
        if p.exists():
            return p
        logger.warning("用户指定的特征模型路径不存在: %s，尝试自动检索", custom_path)

    # 检查本地目录
    local_path = (target_dir / model_name).resolve()
    if local_path.exists():
        return local_path

    # 检查当前工作目录下的 models 目录
    cwd_model_path = Path("models") / model_name
    if cwd_model_path.exists():
        return cwd_model_path.resolve()

    # 尝试从 openwakeword 包定位
    try:
        import openwakeword
        pkg_dir = Path(openwakeword.__file__).parent
        pkg_model_path = pkg_dir / "resources" / "models" / model_name
        if pkg_model_path.exists():
            logger.info("在 openwakeword 模块中找到特征模型: %s", pkg_model_path)
            return pkg_model_path.resolve()
    except Exception:
        pass

    # 尝试自动从网络下载
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("本地未检测到特征模型 '%s'，准备从官方仓库下载...", model_name)
    try:
        logger.info("下载地址: %s -> 保存至: %s", download_url, local_path)
        urllib.request.urlretrieve(download_url, str(local_path))
        logger.info("'%s' 下载成功", model_name)
        return local_path
    except Exception as exc:
        raise RuntimeError(
            f"未能自动下载基础特征模型 '{model_name}' ({exc})。\n"
            f"请手动下载 {download_url} 并放置在 '{local_path}' 目录中，"
            f"或在配置中通过 feature_models 指定有效路径。"
        ) from exc


# ===========================================================================
# OpenVINO 硬件适配与模型管理
# ===========================================================================
class OpenVINOModelRunner:
    """
    OpenVINO 推理引擎包装器。
    
    特点:
        - 针对 NPU 强制静态形状转换 (避免 VPUX compiler 遇到动态形状直接崩溃)
        - 自动硬件回退策略 (NPU -> GPU -> CPU)
        - 低延迟属性配置
    """

    def __init__(self, core: ov.Core) -> None:
        self.core = core

    def compile_model_safe(
        self,
        model_path: Union[str, Path],
        requested_device: str,
        static_shape: Optional[List[int]] = None,
        input_name: Optional[str] = None,
    ) -> Tuple[ov.CompiledModel, str]:
        """
        读取并安全编译 OpenVINO 模型，支持针对 NPU 的静态形状重塑及自动硬件回退。

        返回:
            (compiled_model, actual_device)
        """
        path_str = str(model_path)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"模型文件不存在: {path_str}")

        logger.debug("读取模型: %s", path_str)
        model = self.core.read_model(path_str)

        # 针对 NPU 设备必须满足纯静态形状
        is_npu = "NPU" in requested_device.upper()
        if is_npu or static_shape is not None:
            self._ensure_static_shape(model, static_shape, input_name)

        # 规划回退链条
        device_chain = [requested_device]
        if "NPU" in requested_device.upper():
            device_chain.extend(["GPU", "CPU"])
        elif "GPU" in requested_device.upper():
            device_chain.append("CPU")

        # 去重保持顺序
        candidate_devices = []
        for d in device_chain:
            if d not in candidate_devices:
                candidate_devices.append(d)

        last_error = None
        for dev in candidate_devices:
            try:
                # 尝试为该设备设置低延迟属性
                try:
                    self.core.set_property(dev, {"PERFORMANCE_HINT": "LATENCY"})
                except Exception:
                    pass

                # 若回退到非 NPU 设备且之前未静态化，直接编译
                logger.info("正在尝试在设备 [%s] 上编译模型: %s", dev, os.path.basename(path_str))
                compiled = self.core.compile_model(model, dev)
                logger.info("✅ 成功在 [%s] 上加载模型: %s", dev, os.path.basename(path_str))
                return compiled, dev
            except Exception as e:
                logger.warning("⚠️ 在设备 [%s] 上编译模型失败: %s", dev, e)
                last_error = e

        raise RuntimeError(
            f"模型 {path_str} 在所有候选设备 {candidate_devices} 上均编译失败！"
        ) from last_error

    @staticmethod
    def _ensure_static_shape(
        model: ov.Model,
        forced_shape: Optional[List[int]] = None,
        input_name: Optional[str] = None,
    ) -> None:
        """检查并确保模型输入节点为静态形状。若含动态维度，按规则重塑。"""
        reshape_dict = {}

        for input_node in model.inputs:
            name = input_name if input_name else input_node.get_any_name()
            pshape = input_node.get_partial_shape()

            if forced_shape:
                reshape_dict[name] = forced_shape
            elif pshape.is_dynamic:
                # 遇到动态形状，默认将批处理等动态维度修正为 1
                static_dims = []
                for dim in pshape:
                    if dim.is_dynamic:
                        static_dims.append(1)
                    else:
                        static_dims.append(dim.get_length())
                reshape_dict[name] = static_dims

        if reshape_dict:
            logger.info("对模型应用静态形状重塑 (NPU 兼容): %s", reshape_dict)
            try:
                model.reshape(reshape_dict)
            except Exception as e:
                logger.warning("模型静态形状重塑失败 (若非 NPU 运行可忽略): %s", e)


# ===========================================================================
# 活跃唤醒词实例
# ===========================================================================
class LoadedClassifier:
    """已加载并编译完成的单个唤醒词分类器。"""

    def __init__(
        self,
        config: SingleModelConfig,
        compiled_model: ov.CompiledModel,
        actual_device: str,
        context_len: int = DEFAULT_CONTEXT_LEN,
        is_end_to_end: bool = False,
    ) -> None:
        self.config = config
        self.compiled_model = compiled_model
        self.actual_device = actual_device
        self.context_len = context_len
        self.is_end_to_end = is_end_to_end
        self.last_triggered_time: float = 0.0

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def phrase(self) -> str:
        return self.config.phrase or self.config.name

    @property
    def cutoff(self) -> float:
        return self.config.cutoff

    @property
    def refractory(self) -> float:
        return self.config.refractory

    def predict(self, feature_input: np.ndarray) -> float:
        """执行推理并返回唤醒概率得分 (0.0 ~ 1.0)。"""
        out = self.compiled_model([feature_input])[0]
        # 处理输出可能是 [1, 1], [1, 2] (二分类 softmax) 或标量的情况
        if out.ndim >= 2 and out.shape[-1] == 2:
            return float(out[0, 1])
        return float(out.flat[0])

    def check_trigger(self, score: float, current_time: float) -> bool:
        """检查分数是否达到阈值，并遵循冷却时间规则。"""
        if score >= self.cutoff:
            if current_time - self.last_triggered_time >= self.refractory:
                self.last_triggered_time = current_time
                return True
            else:
                logger.debug(
                    "唤醒词 '%s' 分数达到 %.4f，但处于冷却期 (剩余 %.2fs)",
                    self.name,
                    score,
                    self.refractory - (current_time - self.last_triggered_time),
                )
        return False


# ===========================================================================
# 共享特征提取流水线 (MelSpectrogram + Speech Embedding)
# ===========================================================================
class AudioFeaturePipeline:
    """
    负责将 16kHz PCM 音频切片转换为 96 维 Speech Embeddings 的流水线。
    由所有并发运行的唤醒词模型共享。
    """

    def __init__(
        self,
        runner: OpenVINOModelRunner,
        melspec_path: Union[str, Path],
        embedding_path: Union[str, Path],
        default_device: str = "NPU",
    ) -> None:
        logger.info("正在初始化特征提取流水线...")

        # 1. 编译 MelSpectrogram 模型: 输入 [1, 1280] float32
        self.melspec_model, self.melspec_device = runner.compile_model_safe(
            model_path=melspec_path,
            requested_device=default_device,
            static_shape=[1, CHUNK_SAMPLES],
        )

        # 2. 编译 Embedding 模型: 输入 [1, 76, 32, 1] 或 [1, 1, 76, 32]
        # 读取模型结构判定 input layout
        raw_emb_model = runner.core.read_model(str(embedding_path))
        emb_shape = [1, MEL_HISTORY_FRAMES, 32, 1]
        for inp in raw_emb_model.inputs:
            ps = inp.get_partial_shape()
            if len(ps) == 4 and ps[1].is_static and ps[1].get_length() == 1:
                emb_shape = [1, 1, MEL_HISTORY_FRAMES, 32]
                break

        self.emb_shape = emb_shape
        self.embedding_model, self.embedding_device = runner.compile_model_safe(
            model_path=embedding_path,
            requested_device=default_device,
            static_shape=emb_shape,
        )

        logger.info(
            "特征提取流水线就绪: MelSpectrogram运行于 [%s], Embedding运行于 [%s]",
            self.melspec_device,
            self.embedding_device,
        )

    def compute_melspectrogram(self, audio_1280_samples: np.ndarray) -> np.ndarray:
        """
        计算单帧 1280 采样点 (~80ms) 音频的梅尔频谱并进行标准缩放。
        公式: scaled_mel = (melspec / 10.0) + 2.0
        """
        inp = audio_1280_samples.astype(np.float32).reshape(1, CHUNK_SAMPLES)
        raw_mel = self.melspec_model([inp])[0]
        # 归一化缩放
        scaled_mel = (raw_mel / 10.0) + 2.0
        return scaled_mel

    def compute_embedding(self, mel_frames_76: np.ndarray) -> np.ndarray:
        """
        由 76 帧梅尔频谱历史计算 96 维语音嵌入向量。
        """
        inp = mel_frames_76.astype(np.float32).reshape(self.emb_shape)
        emb = self.embedding_model([inp])[0]
        return emb.reshape(EMBEDDING_DIM)


# ===========================================================================
# 唤醒词引擎管理器 (协调多模型与特征提取)
# ===========================================================================
class WakeWordEngine:
    """管理所有已加载的模型并在流式音频上执行检测。"""

    def __init__(self, config: WakewordServerConfig) -> None:
        self.config = config
        self.core = ov.Core()
        self.runner = OpenVINOModelRunner(self.core)

        # 准备模型存储目录
        models_dir = Path("models").resolve()
        models_dir.mkdir(parents=True, exist_ok=True)

        # 检查是否有需要特征提取器的模型
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
            )

        # 加载所有唤醒词模型
        self.classifiers: Dict[str, LoadedClassifier] = {}
        for m_cfg in config.models:
            self._load_classifier(m_cfg)

        if not self.classifiers:
            logger.warning("未加载任何唤醒词模型！请在配置文件或命令行中指定模型。")

    def _load_classifier(self, m_cfg: SingleModelConfig) -> None:
        """加载、编译单个唤醒词分类器并注册。"""
        device = m_cfg.device or self.config.default_device
        path = Path(m_cfg.model_path).expanduser().resolve()
        if not path.exists():
            # 尝试在 models 目录下寻找
            candidate = Path("models") / m_cfg.model_path
            if candidate.exists():
                path = candidate.resolve()
            else:
                raise FileNotFoundError(f"未找到唤醒词模型文件: {m_cfg.model_path}")

        # 分析模型输入形状
        temp_model = self.core.read_model(str(path))
        inp_shape = temp_model.inputs[0].get_partial_shape()

        # 判断是否为端到端模型 (直接输入音频或频谱) 还是基于 96 维 Embedding
        is_e2e = False
        context_len = DEFAULT_CONTEXT_LEN
        static_shape = None

        if len(inp_shape) == 3 and inp_shape[-1] == EMBEDDING_DIM:
            # 标准 openWakeWord 分类器: [1, context_len, 96]
            if inp_shape[1].is_static:
                context_len = inp_shape[1].get_length()
            static_shape = [1, context_len, EMBEDDING_DIM]
            logger.info("唤醒词 '%s' (基于Embedding, 窗口=%d帧)", m_cfg.name, context_len)
        else:
            # 端到端或者其它结构
            is_e2e = True
            logger.info("唤醒词 '%s' 识别为端到端模型结构", m_cfg.name)

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
        )
        self.classifiers[m_cfg.name] = classifier
        logger.info(
            "已注册唤醒词: 名称='%s', 短语='%s', 阈值=%.2f, 冷却=%.1fs, 设备=[%s]",
            classifier.name,
            classifier.phrase,
            classifier.cutoff,
            classifier.refractory,
            classifier.actual_device,
        )


# ===========================================================================
# Wyoming 协议事件处理器
# ===========================================================================
class OpenVINOWakewordHandler(AsyncEventHandler):
    """
    负责处理 Wyoming 客户端的音频输入与唤醒事件通信。
    """

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

        # 当前会话指定的激活模型 (若客户端未指定则监听所有已加载模型)
        self.active_models: List[LoadedClassifier] = list(engine.classifiers.values())

        # 客户端连接状态与流式缓冲区
        self.raw_audio_buffer = bytearray()
        self.melspec_history = deque(maxlen=MEL_HISTORY_FRAMES)
        self.embedding_history = deque(maxlen=64)

    async def handle_event(self, event: Event) -> bool:
        """分发处理 Wyoming 协议事件。"""

        if Describe.is_type(event.type):
            # 服务自省请求，向 Home Assistant 返回所有支持的唤醒词及属性
            await self.write_event(self.wyoming_info.event())
            return True

        if Detect.is_type(event.type):
            # 客户端启动唤醒词检测会话
            detect = Detect.from_event(event)
            self._reset_session()

            if detect.names:
                # 客户端筛选了特定模型
                filtered = [
                    self.engine.classifiers[n]
                    for n in detect.names
                    if n in self.engine.classifiers
                ]
                if filtered:
                    self.active_models = filtered
                    logger.debug("客户端指定唤醒词范围: %s", [m.name for m in filtered])
                else:
                    logger.warning("客户端指定的模型均未加载: %s，保持监听全部模型", detect.names)
                    self.active_models = list(self.engine.classifiers.values())
            else:
                self.active_models = list(self.engine.classifiers.values())

            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.raw_audio_buffer.extend(chunk.audio)

            # 当累积满 1280 采样点 (2560 字节 = 80ms) 时进行一次滑窗特征提取与检测
            while len(self.raw_audio_buffer) >= CHUNK_BYTES:
                chunk_slice = self.raw_audio_buffer[:CHUNK_BYTES]
                del self.raw_audio_buffer[:CHUNK_BYTES]

                # 解析为 int16 PCM 数组
                audio_samples = np.frombuffer(chunk_slice, dtype=np.int16)

                # 处理特征提取流水线
                detected_model = await self._process_frame(audio_samples)
                if detected_model:
                    # 触发唤醒事件，告知客户端
                    logger.info(
                        "🎉 检测到唤醒词: '%s' (短语: '%s') 时间戳: %s",
                        detected_model.name,
                        detected_model.phrase,
                        chunk.timestamp,
                    )
                    await self.write_event(
                        Detection(
                            name=detected_model.name,
                            timestamp=chunk.timestamp,
                        ).event()
                    )
                    # 触发后适当衰减或清空最近的部分特征缓存，防止连续误触发
                    self._on_detected_cooldown()

            return True

        if AudioStop.is_type(event.type):
            self._reset_session()
            return True

        return True

    def _reset_session(self) -> None:
        """重置流式会话缓存。"""
        self.raw_audio_buffer.clear()
        self.melspec_history.clear()
        self.embedding_history.clear()

    def _on_detected_cooldown(self) -> None:
        """检测命中后的缓存防抖清理。"""
        # 清除最近几帧的 embedding，避免同一语音尾音再次打到高分
        for _ in range(min(8, len(self.embedding_history))):
            if self.embedding_history:
                self.embedding_history.pop()

    async def _process_frame(self, audio_samples: np.ndarray) -> Optional[LoadedClassifier]:
        """处理单帧音频切片，并在各个模型上运行推理。"""
        pipeline = self.engine.feature_pipeline
        now = time.time()

        # 1. 计算梅尔频谱
        if pipeline:
            mel = pipeline.compute_melspectrogram(audio_samples)
            self.melspec_history.append(mel)

            # 2. 累积到 76 帧梅尔频谱后，计算 96 维语音嵌入
            if len(self.melspec_history) >= MEL_HISTORY_FRAMES:
                mel_array = np.array(self.melspec_history, dtype=np.float32)
                embedding = pipeline.compute_embedding(mel_array)
                self.embedding_history.append(embedding)

        # 3. 遍历所有激活的模型并发评估
        for model in self.active_models:
            score = 0.0
            if model.is_end_to_end:
                # 端到端模型直接送入音频数据
                score = model.predict(audio_samples)
            else:
                # 基于 Embedding 的分类器
                if len(self.embedding_history) >= model.context_len:
                    # 取最近 context_len 帧嵌入 (通常为 16 帧)
                    emb_window = list(self.embedding_history)[-model.context_len:]
                    inp_emb = np.array(emb_window, dtype=np.float32).reshape(
                        1, model.context_len, EMBEDDING_DIM
                    )
                    score = model.predict(inp_emb)

            if score > 0.05:
                logger.debug("模型 [%s] 当前帧得分: %.4f (阈值: %.2f)", model.name, score, model.cutoff)

            # 4. 判断阈值与冷却时间
            if model.check_trigger(score, now):
                return model

        return None


# ===========================================================================
# 配置文件解析与构建
# ===========================================================================
def load_config_file(config_path: Path) -> WakewordServerConfig:
    """从 YAML 或 JSON 文件加载服务器与模型配置。"""
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    ext = config_path.suffix.lower()
    raw_dict = {}

    with open(config_path, "r", encoding="utf-8") as f:
        if ext in [".yaml", ".yml"]:
            if yaml is None:
                raise RuntimeError(
                    "解析 YAML 配置文件需要安装 pyyaml: pip install pyyaml"
                )
            raw_dict = yaml.safe_load(f) or {}
        elif ext == ".json":
            raw_dict = json.load(f) or {}
        else:
            raise ValueError(f"不支持的配置文件类型: {ext}，仅支持 .yaml, .yml 或 .json")

    server_sec = raw_dict.get("server", {})
    cfg = WakewordServerConfig(
        host=server_sec.get("host", "0.0.0.0"),
        port=server_sec.get("port", 10400),
        log_level=server_sec.get("log_level", "INFO"),
        default_device=raw_dict.get("default_device", "NPU"),
        default_cutoff=float(raw_dict.get("default_cutoff", 0.5)),
        default_refractory=float(raw_dict.get("default_refractory", 2.0)),
    )

    feat_sec = raw_dict.get("feature_models", {})
    cfg.melspec_model_path = feat_sec.get("melspec_model")
    cfg.embedding_model_path = feat_sec.get("embedding_model")

    raw_models = raw_dict.get("models", [])
    for item in raw_models:
        m_name = item.get("name")
        m_path = item.get("model_path")
        if not m_name or not m_path:
            logger.warning("跳过无效的模型配置项 (缺少 name 或 model_path): %s", item)
            continue

        model_entry = SingleModelConfig(
            name=m_name,
            model_path=m_path,
            phrase=item.get("phrase", m_name.replace("_", " ").title()),
            cutoff=float(item.get("cutoff", cfg.default_cutoff)),
            refractory=float(item.get("refractory", cfg.default_refractory)),
            device=item.get("device"),
            languages=item.get("languages", ["en"]),
        )
        cfg.models.append(model_entry)

    return cfg


# ===========================================================================
# 命令行 CLI 应用
# ===========================================================================
app = typer.Typer(help="Wyoming Wake Word - 基于 OpenVINO 与 openWakeWord 的多模型唤醒词服务")


@app.command()
def main(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="配置文件路径 (支持 YAML 或 JSON)",
    ),
    model: Optional[List[str]] = typer.Option(
        None,
        "--model",
        "-m",
        help="指定加载的一个或多个唤醒词模型路径 (可多次指定)",
    ),
    model_dir: Optional[Path] = typer.Option(
        None,
        "--model-dir",
        help="自动扫描并加载该目录下所有的 .onnx/.tflite/.xml 唤醒词模型",
    ),
    device: str = typer.Option(
        "NPU",
        "--device",
        "-d",
        help="默认推理计算设备 (优先使用 NPU，自动回退 GPU -> CPU)",
    ),
    cutoff: float = typer.Option(
        0.5,
        "--cutoff",
        help="默认唤醒置信度阈值 (0.0 ~ 1.0)",
    ),
    refractory: float = typer.Option(
        2.0,
        "--refractory",
        help="默认唤醒触发后的防重复冷却时间 (秒)",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="服务监听网络地址",
    ),
    port: int = typer.Option(
        10400,
        "--port",
        "-p",
        help="服务监听端口",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="日志记录级别 (DEBUG / INFO / WARNING / ERROR)",
    ),
) -> None:
    """启动基于 OpenVINO 的多模型唤醒词检测服务。"""

    # 1. 初始化基础日志
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 2. 确定配置来源 (文件 vs 命令行)
    if config and config.exists():
        logger.info("从配置文件加载服务参数: %s", config)
        server_config = load_config_file(config)
        # 命令行参数覆盖配置文件对应的顶层字段 (若用户显式提供)
        if host != "0.0.0.0":
            server_config.host = host
        if port != 10400:
            server_config.port = port
        if device != "NPU":
            server_config.default_device = device
    else:
        server_config = WakewordServerConfig(
            host=host,
            port=port,
            log_level=log_level,
            default_device=device,
            default_cutoff=cutoff,
            default_refractory=refractory,
        )

    # 3. 合并命令行中指定的唤醒词模型
    if model:
        for m_path in model:
            base_name = Path(m_path).stem
            # 移除常见的版本后缀如 _v0.1
            clean_name = base_name.split("_v")[0] if "_v" in base_name else base_name
            entry = SingleModelConfig(
                name=clean_name,
                model_path=m_path,
                phrase=clean_name.replace("_", " ").title(),
                cutoff=cutoff,
                refractory=refractory,
                device=device,
            )
            server_config.models.append(entry)

    # 4. 合并指定目录下扫描到的模型
    if model_dir and model_dir.is_dir():
        logger.info("正在扫描目录下的模型文件: %s", model_dir)
        for ext in ("*.onnx", "*.tflite", "*.xml"):
            for file_p in model_dir.glob(ext):
                # 排除基础特征模型
                if file_p.name in ["melspectrogram.onnx", "embedding_model.onnx"]:
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
                            device=device,
                        )
                    )

    if not server_config.models:
        logger.error(
            "未指定任何唤醒词模型！请通过 --config, --model 或 --model-dir 提供模型文件。"
        )
        sys.exit(1)

    # 5. 初始化唤醒词引擎
    engine = WakeWordEngine(server_config)

    # 6. 构建向 Home Assistant 宣告的 Wyoming 服务元数据
    wake_models = [
        WakeModel(
            name=m.name,
            description=f"OpenVINO 唤醒词: {m.phrase} (运行于 {m.actual_device})",
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
                description="基于 OpenVINO 加速与 openWakeWord 的多模型唤醒词服务 (支持 NPU/GPU/CPU)",
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

    # 7. 启动 Wyoming 异步 TCP 服务
    server_uri = f"tcp://{server_config.host}:{server_config.port}"
    server = AsyncServer.from_uri(server_uri)
    logger.info("=================================================================")
    logger.info("🚀 Wyoming 唤醒词服务已启动: %s", server_uri)
    logger.info("🎯 默认设备策略: %s (自动硬件回退)", server_config.default_device)
    logger.info("📋 活跃唤醒词列表 (%d 个):", len(engine.classifiers))
    for m in engine.classifiers.values():
        logger.info(
            "   - [%s] 短语: '%s' | 阈值: %.2f | 冷却: %.1fs | 运行设备: %s",
            m.name,
            m.phrase,
            m.cutoff,
            m.refractory,
            m.actual_device,
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
        logger.info("服务被用户主动中断，正在关闭...")


if __name__ == "__main__":
    app()
