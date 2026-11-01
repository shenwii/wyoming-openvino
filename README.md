# wyoming-openvino

[ 中文 ](README.md) | [ English ](README_en.md) | [ 日本語 ](README_ja.md)

---

Wyoming 协议 + OpenVINO 推理引擎：在 Intel 神经网络处理单元（NPU）、独立显卡（Arc）、核显（iGPU）以及 CPU 上高效运行语音唤醒词检测（Wake Word）、语音识别（STT）和语音合成（TTS）微服务，无缝对接 Home Assistant。

## 🌟 核心特性

- **硬件全加速与 NPU 原生适配（Intel Core Ultra）**：
  - **Wake Word (openWakeWord)**：针对 Intel NPU（Meteor Lake / Lunar Lake / Arrow Lake）进行静态形状重塑（Static Shape Reshape）深度优化，提供极低功耗、全天候运行的唤醒词检测；支持 `NPU` -> `GPU` -> `CPU` 智能平滑回退。
  - **STT (Whisper)**：支持 `GPU`（Intel Arc 独显 / 核显）、`CPU` 与 `AUTO` 混合推理。
  - **TTS (Kokoro)**：推荐使用 `CPU`，自动根据 CPU 拓扑自适应优化。
- **多唤醒词并发检测与灵活配置（Multi-Model & YAML/JSON）**：
  - 支持同时监听多个唤醒词（如同时检测 `Hey Jarvis`、`Alexa`、`小爱同学` 等），底层共享特征提取管线，算力开销极低。
  - 支持 YAML / JSON 配置文件以及独立参数覆盖：每个唤醒词均可单独自定义 `cutoff`（置信度阈值）、`refractory`（冷却时间）与 `device`（指定设备）。
- **直接适配官方原生 OpenVINO 模型包**：
  - **TTS (Kokoro)**：直接指定从 Hugging Face 下载的官方 OpenVINO 目录（包含 `openvino_model.xml`、`openvino_model.bin` 以及 `voices/*.pt` 或 `voices/*.bin` 文件），无需额外格式转换。
  - **STT (Whisper)**：直接指定 Optimum/OpenVINO 导出的 Whisper 模型目录。
  - **Wake Word**：直接支持 openWakeWord 训练或导出的 `.onnx`、`.tflite` 与 OpenVINO `.xml` 模型。
- **极简命令行体验（统一使用 `--model-dir` 或 `--config`）**：
  - 各服务均支持极简启动命令与自动模型发现。
- **智能分句流式即时发声（借鉴 Kokoro-FastAPI 分词优化）**：
  - 缩写词与数字保护：保护 `Dr.`、`Mr.`、`etc.`、`e.g.` 与浮点数（如 `3.14`），绝不生硬断句。
  - 保留完整声学标点（`。！？!?`），使合成语音具备自然顿挫与语调起伏。
  - Wyoming 流式首句秒级出声，长文本合成告别十几秒等待。
- **发音人多音色线性加权融合（Voice Blending）**：
  - 借鉴 Kokoro-FastAPI 声音融合语法，支持 `zf_xiaobei(0.7)+af_heart(0.3)`、`af_bella(2)+af_sky(1)` 或 `zf_xiaobei+af_heart`，自动归一化权重并线性组合风格向量。
- **全自动多语言与协议动态指定**：
  - 命令行 `--language` 默认为 `auto`（完全可以不传）。
  - **STT**：优先遵循 Home Assistant 请求中指定的语言；若未指定，由 Whisper 神经网络原生自动听声辨语（支持中、英、日等 90+ 种语言）。
  - **TTS**：优先遵循 Home Assistant 选择的发音人与语言；自动根据声音前缀（如 `zf_` -> 中文、`jf_` -> 日文、`af_` -> 英文）或文本内容智能匹配最佳 G2P 引擎；中英混排时通过 `en_callable` 回调与 `EspeakFallback` 兜底，纯英文分句自动切换原生英文音标。

---

## 📦 环境要求与依赖安装

### 1. 系统要求
- Python >= 3.10
- Intel GPU 驱动与 OpenVINO Runtime

### 2. 依赖安装

根据你的部署场景，选择对应的精简依赖清单：

```bash
# 场景 A: 仅运行 STT (语音识别) 服务（极简轻量，仅需 OpenVINO GenAI 与 Wyoming）
pip install -r requirements-stt.txt

# 场景 B: 仅运行 TTS (语音合成) 服务（含 Kokoro、PyTorch 音色解析及中/英/日多语言分词依赖）
# 依赖含: pyopenjtalk, fugashi, unidic-lite, pypinyin, spacy 等
pip install -r requirements-tts.txt

# 场景 C: 仅运行 Wake Word (唤醒词) 服务（含 OpenVINO、Wyoming、PyYAML）
pip install -r requirements-wakeword.txt

# 场景 D: 全量安装 (STT + TTS + Wake Word)
pip install -r requirements.txt
```

---

## 🧠 模型文件说明

### 1. STT 模型 (Whisper)

Whisper 导出的 OpenVINO 目录结构通常为：
```
whisper-openvino/
├── openvino_model.xml
├── openvino_model.bin
├── config.json
├── tokenizer.json
└── ...
```

### 2. TTS 模型 (Kokoro)

从 Hugging Face（如 `Kokoro-82M-FP16-OpenVINO`）下载的官方目录结构通常为：
```
Kokoro-82M-FP16-OpenVINO/
├── openvino_model.xml     # 模型结构定义
├── openvino_model.bin     # 模型二进制权重
├── config.json            # 模型配置
├── openvino_config.json
├── voices/                # 发音人音色向量文件夹
│   ├── af_heart.pt        # 美音女声
│   ├── zf_xiaobei.pt      # 中文女声
│   ├── jf_teio.pt         # 日语音色
│   └── ...
└── ...
```
> [!NOTE]
> `voices/` 下的每一个文件均代表一个独立发音人。服务启动时会自动扫描该目录下的全部音色，并自动向 Home Assistant 完整注册。

### 3. Wake Word 唤醒词模型 (openWakeWord)

存放于 `models/` 目录中：
```
models/
├── melspectrogram.onnx      # 特征提取模型 1 (若本地不存在，服务会自动从官方 Releases 下载)
├── embedding_model.onnx     # 特征提取模型 2 (若本地不存在，服务会自动从官方 Releases 下载)
├── hey_jarvis_v0.1.onnx     # 官方预训练或自训练唤醒词模型
├── alexa_v0.1.onnx
└── ...
```
> [!NOTE]
> 支持 openWakeWord 导出的 `.onnx`、`.tflite` 与 OpenVINO 原生 `.xml` 模型。针对 Intel NPU，脚本内置自动静态形状转换与智能硬件平滑回退。

---

## 🚀 启动与使用

### 1. 启动 STT 语音识别服务

```bash
python wyoming-stt.py \
    --model-dir /path/to/whisper-openvino \
    --device GPU \
    --port 10020
```

### 2. 启动 TTS 语音合成服务

```bash
python wyoming-tts.py \
    --model-dir /path/to/Kokoro-82M-FP16-OpenVINO \
    --device CPU \
    --speed 0.85 \
    --port 10021
```

> [!TIP]
> **语速推荐**：Kokoro 默认语速 `1.0` 在朗读中文/日文时稍偏快。启动时添加 `--speed 0.85`（或 `0.8`）能够获得最自然悠扬、清晰自然的播报体验！
> **设备推荐**：由于 Kokoro 模型中算子的特性，TTS 推荐使用 `CPU` 设备，具有最佳兼容性与极佳的吞吐表现。

### 3. 启动 Wake Word 唤醒词检测服务

**方式 A: 基于 YAML 配置文件（推荐多模型并发检测）**
```bash
python wyoming-wakeword.py --config wakeword_config.yaml
```

**方式 B: 纯命令行快速指定模型并优先使用 NPU**
```bash
python wyoming-wakeword.py \
    --model models/hey_jarvis_v0.1.onnx \
    --device NPU \
    --cutoff 0.6 \
    --port 10400
```

**方式 C: 自动扫描目录下所有模型**
```bash
python wyoming-wakeword.py --model-dir ./models --device NPU
```

### 4. 本地快速测试工具

#### A. 测试 TTS 发音与生成 WAV (`test_tts.py`)
```bash
# 中文基础测试
python test_tts.py "我来测试一下语音合成的效果。" --model-dir /path/to/Kokoro-82M-FP16-OpenVINO -o test.wav

# 多音色融合测试（按 7:3 混合中文与英文音色）
python test_tts.py "今天天气真好，我们来测试 OpenVINO 语音合成。" \
    --model-dir /path/to/Kokoro-82M-FP16-OpenVINO \
    -v "zf_xiaobei(0.7)+af_heart(0.3)" \
    --speed 0.85 \
    -o blend.wav
```

#### B. 测试 Wake Word 唤醒词检测 (`test_wakeword.py`)
```bash
# 模式 1: 本地直接用 OpenVINO 评估 WAV 文件
python test_wakeword.py sample.wav --config wakeword_config.yaml --device NPU

# 模式 2: 作为客户端连接正在运行的 Wyoming 唤醒词服务
python test_wakeword.py sample.wav --server 127.0.0.1:10400
```

---

## ⚙️ 常用命令行参数对照

### STT 参数 (`wyoming-stt.py`)

| 参数 | 简写 | 默认值 | 说明 |
|---|---|---|---|
| `--model-dir` | `-m` | (必填) | Whisper 模型目录 |
| `--device` | `-d` | `GPU` | 推理设备 (`CPU` / `GPU` / `AUTO` / `MULTI:CPU,GPU`) |
| `--language` | `-l` | `auto` | 默认语言（优先遵循 HA 协议，其次自动检测） |
| `--port` | `-p` | `10020` | 服务监听端口 |
| `--host` | | `0.0.0.0` | 监听地址 |

### TTS 参数 (`wyoming-tts.py`)

| 参数 | 简写 | 默认值 | 说明 |
|---|---|---|---|
| `--model-dir` | `-m` | (必填) | Kokoro 模型目录（自动加载 xml/bin 和 voices 目录） |
| `--device` | `-d` | `CPU` | 推理设备 (`CPU` 推荐 / `GPU` / `AUTO`) |
| `--language` | `-l` | `auto` | 默认语言（优先遵循 HA 协议，其次按音色/文本自动识别） |
| `--speed` | `-s` | `1.0` | 语速倍率（推荐 `0.85` 或 `0.8`；>1.0 加速，<1.0 减速） |
| `--threads` | `-t` | `None` | OpenVINO 推理线程数（默认自动根据 CPU 拓扑自适应优化） |
| `--port` | `-p` | `10021` | 服务监听端口 |

### Wake Word 参数 (`wyoming-wakeword.py`)

| 参数 | 简写 | 默认值 | 说明 |
|---|---|---|---|
| `--config` | `-c` | `None` | 配置文件路径（YAML 或 JSON），支持定义多模型与各项独立参数 |
| `--model` | `-m` | `None` | 唤醒词模型路径（可多次指定以加载多个模型） |
| `--model-dir` | | `None` | 自动扫描并加载该目录下所有 `.onnx` / `.tflite` / `.xml` 唤醒词模型 |
| `--device` | `-d` | `NPU` | 默认推理设备 (`NPU` 优先 / `GPU` / `CPU` / `AUTO`，自动硬件回退) |
| `--cutoff` | | `0.5` | 默认唤醒阈值 (0.0 ~ 1.0) |
| `--refractory` | | `2.0` | 默认唤醒冷却时间 (秒) |
| `--port` | `-p` | `10400` | 服务监听端口 (HA Wyoming openWakeWord 官方端口 10400) |
| `--host` | | `0.0.0.0` | 监听地址 |

---

## 🏠 Home Assistant 配置

在 Home Assistant 中进入 **配置** -> **设备与服务** -> **添加集成** -> 搜索 **Wyoming Protocol**：

1. 添加 STT：输入服务 IP，端口填 `10020`
2. 添加 TTS：输入服务 IP，端口填 `10021`
3. 添加 Wake Word：输入服务 IP，端口填 `10400`

添加完成后，在 Home Assistant 的「语音助手 (Assist)」中，即可自由选择中、英、日等语言以及对应的 Kokoro 发音人，并开启极低功耗的高灵敏度唤醒词！

## 📜 许可证

GNU General Public License v3.0 (GPL v3)
