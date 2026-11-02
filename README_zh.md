# wyoming-openvino

[ English ](README.md) | [ 中文 ](README_zh.md) | [ 日本語 ](README_ja.md) | [ 한국어 ](README_ko.md)

---

Wyoming 协议 + Intel OpenVINO：针对 Intel NPU (Core Ultra)、Arc 独立显卡、核显 (iGPU) 以及 CPU 深度优化的高性能唤醒词检测 (Wake Word)、语音识别 (STT) 与语音合成 (TTS) 微服务体系，无缝对接 Home Assistant 语音助手。

## 🌟 核心特性

- **硬件全加速与原生 NPU 优化 (Intel Core Ultra)**:
  - **唤醒词 (openWakeWord)**: 专为 Intel NPU (Meteor Lake / Lunar Lake / Arrow Lake) 深度优化，支持自动静态形状重塑，超低功耗常驻监听，具备平滑的 `NPU` -> `GPU` -> `CPU` 自动硬件回退。
  - **语音识别 (Whisper)**: 优先推荐运行于 `GPU` (Arc 独显 / 核显)，兼顾 `CPU` 与 `AUTO` 调度。
  - **语音合成 (Kokoro)**: 优先推荐运行于 `CPU`，具备自适应 CPU 线程调度与极佳的算子兼容性。
- **多模型并发唤醒词检测与 YAML/JSON 配置**:
  - 单路音频流同时监听多个唤醒词 (如 `Hey Jarvis`, `Alexa`, 自定义模型等)，底层共享特征提取流水线。
  - 支持 YAML/JSON 配置文件精细化定义每个模型的 `cutoff` (阈值)、`refractory` (冷却时间) 和 `device` (执行设备)。
- **智能流式切句与即时生成 (Kokoro TTS)**:
  - 缩写与浮点数保护：防止对 `Dr.`, `Mr.`, `etc.`, `3.14` 等发生粗暴生硬截断。
  - 完整保留句尾标点 (`。！？!?`)，使语音生成具备自然的抑扬顿挫。
  - 分句级流式输出：首句在 100~300ms 内即刻发声，通过 Wyoming TCP 实时推流，彻底告别整段等待。
- **加权多音色融合 (Voice Blending)**:
  - 支持 Kokoro-FastAPI 风格语法，如 `zf_xiaobei(0.7)+af_heart(0.3)`、`af_bella(2)+af_sky(1)` 动态融合多发音人音色。
- **全方位国际化 (i18n)**:
  - 原生支持英文 (`en`)、中文 (`zh`)、日文 (`ja`)、韩文 (`ko`) 4 门语言。
  - 控制台日志输出与 Home Assistant 内显示的 Wyoming 服务元数据均可通过环境变量 (`LANG`, `LC_ALL`, `WYOMING_LANG`) 动态切换。
- **全面容器化支持 (Docker)**:
  - 拆分为独立的 `wyoming-stt`, `wyoming-tts`, `wyoming-wakeword` 目录，每个子项目具备独立 `Dockerfile`、`requirements.txt` 与 Shell 启动脚本，统一基于 `docker.io/openvino/ubuntu24_runtime:latest` 基础镜像构建。

---

## 📁 项目目录结构

```
wyoming-openvino/
├── wyoming-stt/              # 语音识别服务 (Whisper OpenVINO)
│   ├── wyoming-stt.py        # Python 服务脚本
│   ├── wyoming-stt.sh        # Shell 入口启动脚本
│   ├── i18n.py               # 国际化引擎
│   ├── lang/                 # 语言映射包 (en.json, zh.json, ja.json, ko.json)
│   ├── requirements.txt      # STT 依赖清单
│   └── Dockerfile            # 容器构建文件
│
├── wyoming-tts/              # 语音合成服务 (Kokoro OpenVINO)
│   ├── wyoming-tts.py        # Python 服务脚本
│   ├── wyoming-tts.sh        # Shell 入口启动脚本
│   ├── i18n.py               # 国际化引擎
│   ├── lang/                 # 语言映射包 (en.json, zh.json, ja.json, ko.json)
│   ├── requirements.txt      # TTS 依赖清单
│   ├── Dockerfile            # 轻量容器构建文件 (默认，不含 PyTorch)
│   └── Dockerfile.torch      # 包含 PyTorch 支持的容器构建文件
│
├── wyoming-wakeword/         # 唤醒词服务 (openWakeWord OpenVINO)
│   ├── wyoming-wakeword.py   # Python 服务脚本
│   ├── wyoming-wakeword.sh   # Shell 入口启动脚本
│   ├── wakeword_config.yaml  # 多模型服务配置文件
│   ├── i18n.py               # 国际化引擎
│   ├── lang/                 # 语言映射包 (en.json, zh.json, ja.json, ko.json)
│   ├── requirements.txt      # 唤醒词依赖清单
│   └── Dockerfile            # 容器构建文件
│
├── wakeword_trainer/         # 唤醒词训练流水线工具箱
├── README.md                 # 英文文档 (默认)
├── README_zh.md              # 中文文档
├── README_ja.md              # 日文文档
├── README_ko.md              # 韩文文档
└── LICENSE                   # GPL-3.0 开源协议
```

---

## 📦 安装与直接运行

### 1. Wyoming STT (语音识别)

```bash
cd wyoming-stt
pip install -r requirements.txt

# 使用 GPU 加速并强制输出简体中文
./wyoming-stt.sh --model-dir /path/to/whisper-openvino --device GPU --to-simplified --port 10020

# 默认设置 (自动识别多语言)
./wyoming-stt.sh --model-dir /path/to/whisper-openvino --device GPU --port 10020
```

### 2. Wyoming TTS (语音合成)

```bash
cd wyoming-tts
pip install -r requirements.txt

# 在 CPU 上运行 Kokoro TTS (推荐 0.85 语速获得最自然听感)
./wyoming-tts.sh --model-dir /path/to/Kokoro-82M-FP16-OpenVINO --device CPU --speed 0.85 --port 10021
```

### 3. Wyoming Wake Word (唤醒词)

```bash
cd wyoming-wakeword
pip install -r requirements.txt

# 方式 A: 使用 YAML 配置文件启动多模型检测 (推荐)
./wyoming-wakeword.sh --config wakeword_config.yaml

# 方式 B: 极速单模型在 Intel NPU 上运行
./wyoming-wakeword.sh --model models/hey_jarvis_v0.1.onnx --device NPU --cutoff 0.6 --port 10400

# 方式 C: 自动扫描目录下所有模型
./wyoming-wakeword.sh --model-dir ./models --device NPU --port 10400
```

---

## 🐳 Docker 容器部署

所有微服务统一使用官方 `docker.io/openvino/ubuntu24_runtime:latest` 镜像作为基础容器，免去复杂的 OpenVINO 环境配置。

### 1. 构建 Docker 镜像

```bash
# 构建 STT 镜像
docker build -t wyoming-stt ./wyoming-stt

# 构建 TTS 镜像 (默认轻量版，不包含 PyTorch，体积缩小约 2GB)
docker build -t wyoming-tts ./wyoming-tts

# 或构建包含 PyTorch 的 TTS 镜像 (支持直接加载复杂 .pt 声音权重)
docker build -f ./wyoming-tts/Dockerfile.torch -t wyoming-tts:torch ./wyoming-tts

# 构建 Wake Word 镜像
docker build -t wyoming-wakeword ./wyoming-wakeword
```

### 2. 运行容器

#### STT 容器 (通过 `/dev/dri` 映射 GPU)
```bash
docker run -d \
  --name wyoming-stt \
  --restart unless-stopped \
  --net host \
  --device /dev/dri \
  -v /path/to/whisper-openvino:/models/whisper:ro \
  -e LANG=zh_CN.UTF-8 \
  wyoming-stt \
  --model-dir /models/whisper \
  --device GPU \
  --to-simplified \
  --port 10020
```

#### TTS 容器
```bash
docker run -d \
  --name wyoming-tts \
  --restart unless-stopped \
  --net host \
  -v /path/to/Kokoro-82M-FP16-OpenVINO:/models/kokoro:ro \
  -e LANG=zh_CN.UTF-8 \
  wyoming-tts \
  --model-dir /models/kokoro \
  --device CPU \
  --speed 0.85 \
  --port 10021
```

#### Wake Word 容器 (通过 `/dev/accel` 映射 Intel NPU)
```bash
docker run -d \
  --name wyoming-wakeword \
  --restart unless-stopped \
  --net host \
  --device /dev/accel \
  -v /path/to/models:/app/models \
  -v $(pwd)/wyoming-wakeword/wakeword_config.yaml:/app/wakeword_config.yaml:ro \
  -e LANG=zh_CN.UTF-8 \
  wyoming-wakeword \
  --config /app/wakeword_config.yaml
```

---

## 🌐 国际化与语言设置 (i18n)

每个子服务内均包含多语言支持模块，并将语言文件独立于 `lang/` 文件夹（**英文**: `en.json`、**中文**: `zh.json`、**日文**: `ja.json`、**韩文**: `ko.json`）。如需扩充其它语言（如德文 `de.json`），只需在对应的 `lang/` 目录下新增对应 JSON 映射即可，无需修改任何 Python 逻辑。

语言切换完全通过标准环境变量控制，无需附加繁琐的命令行参数：
- `WYOMING_LANG` (最高优先级)
- `LC_ALL`
- `LC_MESSAGES`
- `LANG` (如 `zh_CN.UTF-8`, `ja_JP.UTF-8`, `ko_KR.UTF-8`, `en_US.UTF-8`)

生效范围：
1. 终端与日志输出 (`logger.info`, `logger.warning` 等)
2. Wyoming 协议向 Home Assistant 宣告的服务元数据 (`AsrProgram`, `TtsProgram`, `WakeProgram`)

示例：
```bash
# 以日文模式启动 TTS 服务
LANG=ja_JP.UTF-8 ./wyoming-tts.sh --model-dir /models/kokoro

# 以中文模式启动 STT 服务
LANG=zh_CN.UTF-8 ./wyoming-stt.sh --model-dir /models/whisper
```

---

## 🏠 Home Assistant 集成配置

在 Home Assistant 中，进入 **设置** -> **设备与服务** -> **添加集成** -> 搜索 **Wyoming Protocol**：

1. **添加 STT**: 输入主机 IP，端口 `10020`
2. **添加 TTS**: 输入主机 IP，端口 `10021`
3. **添加 Wake Word**: 输入主机 IP，端口 `10400`

添加完成后，前往 **语音助手 (Assist)** 配置您的专属语音流水线：
- 语音合成选择 `kokoro-openvino`
- 语音识别选择 `wyoming-openvino-stt`
- 唤醒词选择 `wyoming-openvino-wakeword`

---

## 📜 开源协议

GNU General Public License v3.0 (GPL v3)
