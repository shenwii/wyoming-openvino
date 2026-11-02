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

from wyoming.audio import AudioChunk, AudioStart, AudioStop
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


def configure_logging(level_name: str) -> None:
    """统一配置全局日志级别，支持动态调整。"""
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
    cutoff: Optional[float] = None       # 若未指定则继承 WakewordServerConfig.default_cutoff
    refractory: Optional[float] = None   # 若未指定则继承 WakewordServerConfig.default_refractory
    device: Optional[str] = None         # 若未指定则继承 WakewordServerConfig.default_device
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
        effective_cutoff: float = 0.5,
        effective_refractory: float = 2.0,
    ) -> None:
        self.config = config
        self.compiled_model = compiled_model
        self.actual_device = actual_device
        self.context_len = context_len
        self.is_end_to_end = is_end_to_end
        self._effective_cutoff = effective_cutoff
        self._effective_refractory = effective_refractory
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
    def refractory(self) -> float:
        return self._effective_refractory

    @refractory.setter
    def refractory(self, value: float) -> None:
        self._effective_refractory = value

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
        # MelSpectrogram 属于极轻量的 DSP/FFT 频域转换 (耗时 <0.1ms)，在 CPU 上执行稳定性最高且编译极快
        # 若在 NPU 上编译易因复杂频域算子导致 Level-Zero 驱动挂起或超时
        melspec_dev = "CPU" if "NPU" in default_device.upper() else default_device
        self.melspec_model, self.melspec_device = runner.compile_model_safe(
            model_path=melspec_path,
            requested_device=melspec_dev,
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
        确保返回为 (-1, 32) 的 2D 频谱帧矩阵。
        """
        inp = audio_1280_samples.astype(np.float32).reshape(1, CHUNK_SAMPLES)
        raw_mel = self.melspec_model([inp])[0]
        # 归一化缩放
        scaled_mel = (raw_mel / 10.0) + 2.0
        return scaled_mel.reshape(-1, 32)

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
        cutoff = m_cfg.cutoff if m_cfg.cutoff is not None else self.config.default_cutoff
        refractory = m_cfg.refractory if m_cfg.refractory is not None else self.config.default_refractory

        path = Path(m_cfg.model_path).expanduser().resolve()
        if not path.exists():
            # 尝试在 models 目录下寻找
            candidate = Path("models") / m_cfg.model_path
            if candidate.exists():
                path = candidate.resolve()
            else:
                raise FileNotFoundError(f"未找到唤醒词模型文件: {m_cfg.model_path}")

        # 分析模型输入形状与类型
        temp_model = self.core.read_model(str(path))
        inp = temp_model.inputs[0]
        inp_shape = inp.get_partial_shape()

        # 判断模型架构类型
        is_e2e = False
        context_len = DEFAULT_CONTEXT_LEN
        static_shape = None

        if len(inp_shape) == 3 and inp_shape[-1] == EMBEDDING_DIM:
            # 标准 openWakeWord 分类器: [1, context_len, 96]
            if inp_shape[1].is_static:
                context_len = inp_shape[1].get_length()
            static_shape = [1, context_len, EMBEDDING_DIM]
            logger.info("唤醒词 '%s' (基于Embedding, 窗口=%d帧)", m_cfg.name, context_len)
        elif len(inp_shape) == 3 and inp_shape[-1] in (32, 40):
            # 这是 microWakeWord 或 2D 梅尔频谱模型 (例如 [1, 98, 32] 或 [1, 98, 40])
            raise ValueError(
                f"唤醒词模型 '{m_cfg.name}' ({path}) 的输入形状为 {inp_shape}，"
                f"属于梅尔频谱特征模型（如 microWakeWord）。\n"
                f"当前服务是基于 openWakeWord 架构设计的，分类器模型输入必须为 [1, 16, 96]（语音嵌入向量）。\n"
                f"请使用 openWakeWord 官方分类器模型（如 hey_jarvis_v0.1.onnx、alexa_v0.1.onnx 等），"
                f"或移除该非 openWakeWord 模型。"
            )
        elif len(inp_shape) == 2:
            # 端到端原始波形输入模型 [1, SAMPLES]
            is_e2e = True
            e2e_samples = inp_shape[1].get_length() if inp_shape[1].is_static else CHUNK_SAMPLES
            static_shape = [1, e2e_samples]
            logger.info("唤醒词 '%s' 识别为端到端音频模型结构 (输入长度: %d 采样点)", m_cfg.name, e2e_samples)
        else:
            raise ValueError(
                f"不支持的唤醒词模型输入形状: {inp_shape} (模型文件: {path})。\n"
                f"支持的类型: openWakeWord 分类器 [1, N, 96] 或 端到端音频模型 [1, N]。"
            )

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
        # 预先使用静音 (全零) 填充滑窗缓冲区，确保从收到第 1 秒语音起即可无延迟实时计算预测，无需盲等 7 秒缓冲
        self.melspec_buffer = np.zeros((MEL_HISTORY_FRAMES, 32), dtype=np.float32)
        self.embedding_buffer = np.zeros((DEFAULT_CONTEXT_LEN, EMBEDDING_DIM), dtype=np.float32)
        self.has_detected = False

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

        if AudioStart.is_type(event.type):
            # 音频流启动事件
            self._reset_session()
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
                detected_model = None
                try:
                    detected_model = await self._process_frame(audio_samples)
                except Exception as exc:
                    logger.error("音频帧处理或模型推理失败: %s", exc, exc_info=True)
                if detected_model:
                    self.has_detected = True
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
                    # 触发后清空上下文缓冲，防止连续误触发
                    self._on_detected_cooldown()

            return True

        if AudioStop.is_type(event.type):
            # 音频流结束: 若未检测到唤醒词，必须明确告知客户端 NotDetected，防止客户端死等导致超时！
            if not self.has_detected:
                logger.debug("音频流结束未触发唤醒，发送 NotDetected 事件")
                await self.write_event(NotDetected().event())
            self._reset_session()
            return True

        return True

    def _reset_session(self) -> None:
        """重置流式会话缓存。"""
        self.raw_audio_buffer.clear()
        self.melspec_buffer.fill(0)
        self.embedding_buffer.fill(0)
        self.has_detected = False

    def _on_detected_cooldown(self) -> None:
        """检测命中后的缓存防抖清理。"""
        # 清空 embedding 缓冲，避免同一语音尾音再次打出高分
        self.embedding_buffer.fill(0)

    async def _process_frame(self, audio_samples: np.ndarray) -> Optional[LoadedClassifier]:
        """处理单帧音频切片，并在各个模型上运行推理。"""
        pipeline = self.engine.feature_pipeline
        now = time.time()

        # 1. 计算梅尔频谱并滑窗滚入
        if pipeline:
            new_mels = pipeline.compute_melspectrogram(audio_samples)
            n_new = len(new_mels)
            if n_new >= MEL_HISTORY_FRAMES:
                self.melspec_buffer[:] = new_mels[-MEL_HISTORY_FRAMES:]
            else:
                self.melspec_buffer = np.roll(self.melspec_buffer, -n_new, axis=0)
                self.melspec_buffer[-n_new:] = new_mels

            # 2. 由 76 帧梅尔频谱计算 96 维语音嵌入
            embedding = pipeline.compute_embedding(self.melspec_buffer)

            # 滑窗滚入 1 帧新的 embedding
            self.embedding_buffer = np.roll(self.embedding_buffer, -1, axis=0)
            self.embedding_buffer[-1] = embedding

        # 3. 遍历所有激活的模型并发评估
        for model in self.active_models:
            score = 0.0
            if model.is_end_to_end:
                # 端到端原始音频模型: 确保形状为 [1, N] 且类型为 float32
                float_audio = audio_samples.astype(np.float32).reshape(1, -1)
                score = model.predict(float_audio)
            else:
                # 基于 Embedding 的分类器: 取最近 context_len 帧 (通常 16 帧)
                if len(self.embedding_buffer) >= model.context_len:
                    emb_window = self.embedding_buffer[-model.context_len:]
                    inp_emb = emb_window.reshape(1, model.context_len, EMBEDDING_DIM)
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
def _safe_float(val, default: float) -> float:
    """安全转换为浮点数，若为 None 或转换异常则返回默认值。"""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


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

    # 增强容错兼容性: 支持 server / feature_models 节，同时兼顾顶层配置与连字符写法
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

    melspec_model = feat_sec.get("melspec_model") or raw_dict.get("melspec_model")
    embedding_model = feat_sec.get("embedding_model") or raw_dict.get("embedding_model")

    cfg = WakewordServerConfig(
        host=str(host),
        port=port,
        log_level=str(log_level).upper(),
        default_device=str(default_device),
        default_cutoff=default_cutoff,
        default_refractory=default_refractory,
        melspec_model_path=str(melspec_model) if melspec_model else None,
        embedding_model_path=str(embedding_model) if embedding_model else None,
    )

    raw_models = raw_dict.get("models") or []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        m_name = item.get("name")
        m_path = item.get("model_path")
        if not m_name or not m_path:
            logger.warning("跳过无效的模型配置项 (缺少 name 或 model_path): %s", item)
            continue

        cutoff_raw = item.get("cutoff")
        refractory_raw = item.get("refractory")

        model_entry = SingleModelConfig(
            name=str(m_name),
            model_path=str(m_path),
            phrase=item.get("phrase", str(m_name).replace("_", " ").title()),
            # 若模型未显式配置 cutoff / refractory，保持 None 以动态继承服务全局默认值
            cutoff=_safe_float(cutoff_raw, cfg.default_cutoff) if cutoff_raw is not None else None,
            refractory=_safe_float(refractory_raw, cfg.default_refractory) if refractory_raw is not None else None,
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
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="默认推理计算设备 (优先使用 NPU，自动回退 GPU -> CPU) [缺省: NPU]",
    ),
    cutoff: Optional[float] = typer.Option(
        None,
        "--cutoff",
        help="默认唤醒置信度阈值 (0.0 ~ 1.0) [缺省: 0.5]",
    ),
    refractory: Optional[float] = typer.Option(
        None,
        "--refractory",
        help="默认唤醒触发后的防重复冷却时间 (秒) [缺省: 2.0]",
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help="服务监听网络地址 [缺省: 0.0.0.0]",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        "-p",
        help="服务监听端口 [缺省: 10400]",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="日志记录级别 (DEBUG / INFO / WARNING / ERROR) [缺省: INFO]",
    ),
    melspec_model: Optional[str] = typer.Option(
        None,
        "--melspec-model",
        help="MelSpectrogram 特征提取模型路径 (可选)",
    ),
    embedding_model: Optional[str] = typer.Option(
        None,
        "--embedding-model",
        help="Embedding 特征提取模型路径 (可选)",
    ),
) -> None:
    """启动基于 OpenVINO 的多模型唤醒词检测服务。"""

    # 1. 初始化基础日志 (若命令行指定了 log_level 则优先作为启动日志级别，否则使用 INFO)
    init_log_level = log_level or "INFO"
    configure_logging(init_log_level)

    # 2. 确定配置基准 (配置文件优先加载为基准，若无配置文件则采用内置默认值)
    if config:
        if not config.exists():
            logger.error("指定的配置文件不存在: %s", config)
            sys.exit(1)
        logger.info("从配置文件加载服务参数: %s", config)
        server_config = load_config_file(config)
    else:
        server_config = WakewordServerConfig()

    # 3. 统一合并命令行显式指定的参数 (命令行优先级高于配置文件)
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
    if refractory is not None:
        server_config.default_refractory = refractory
    if melspec_model is not None:
        server_config.melspec_model_path = melspec_model
    if embedding_model is not None:
        server_config.embedding_model_path = embedding_model

    # 4. 统一使最终确定的日志级别生效 (保证配置文件或命令行指定的 log_level 绝对生效)
    configure_logging(server_config.log_level)
    logger.debug("日志级别已生效并设置为: %s", server_config.log_level)

    # 5. 合并命令行中指定的唤醒词模型
    if model:
        for m_path in model:
            base_name = Path(m_path).stem
            # 移除常见的版本后缀如 _v0.1
            clean_name = base_name.split("_v")[0] if "_v" in base_name else base_name
            entry = SingleModelConfig(
                name=clean_name,
                model_path=m_path,
                phrase=clean_name.replace("_", " ").title(),
                cutoff=cutoff,         # 若命令行指定则使用，否则为 None (动态继承全局 default_cutoff)
                refractory=refractory, # 若命令行指定则使用，否则为 None (动态继承全局 default_refractory)
                device=device,         # 若命令行指定则使用，否则为 None (动态继承全局 default_device)
            )
            # 若配置中已存在同名模型，替换已有项；否则追加
            existing_idx = next(
                (i for i, m in enumerate(server_config.models) if m.name == clean_name),
                None,
            )
            if existing_idx is not None:
                logger.info("命令行指定的模型 '%s' 覆盖配置文件中的同名模型", clean_name)
                server_config.models[existing_idx] = entry
            else:
                server_config.models.append(entry)

    # 6. 合并指定目录下扫描到的模型
    if model_dir and model_dir.is_dir():
        logger.info("正在扫描目录下的模型文件: %s", model_dir)
        for ext in ("*.onnx", "*.tflite", "*.xml"):
            for file_p in sorted(model_dir.glob(ext)):
                # 排除基础特征提取模型 (如 melspectrogram / embedding_model)
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
