# wyoming-openvino

Wyoming 协议 + OpenVINO 推理引擎：在 Intel GPU/CPU 上运行语音识别（STT）和语音合成（TTS）服务，无缝对接 Home Assistant。

## 🌟 核心特性

- **多语言全自动推测与协议动态指定**：
  - **无需强制指定语言**：命令行 `--language` 默认为 `auto`（完全可以不传）。
  - **优先遵循 Wyoming 协议**：当 Home Assistant 配置的助理管线（如中文助理、日文助理、英文助理）发起请求时，服务会自动提取协议中的语言进行针对性识别与合成。
  - **自动兜底与检测**：
    - **STT (Whisper)**：若协议未传语言，由 Whisper 神经网络原生自动检测输入的说话语言（支持中、英、日等 90+ 种语言）。
    - **TTS (Kokoro)**：若协议未传语言，自动根据选用的声音前缀（如 `zf_` -> 中文、`jf_` -> 日文、`af_` -> 英文）或文本字符（平假名/片假名/汉字/拉丁字母）自动路由至最合适的 G2P 引擎。
- **Intel 硬件加速**：
  - 充分利用 Intel 独立显卡（Arc 系列）、核显（iGPU）或 CPU。
  - 支持 `GPU`、`CPU`、`AUTO` 与 `MULTI:CPU,GPU` 混合推理。
- **Home Assistant 深度集成**：
  - 自动向 HA 宣告支持的完整语言列表与声音声库（中/英/日等）。

---

## 📦 环境要求与安装

### 1. 系统要求
- Python >= 3.10
- Intel GPU 驱动与 OpenVINO Runtime

### 2. 安装依赖
```bash
pip install -r requirements.txt
```
> [!TIP]
> `requirements.txt` 中已包含中文（`pypinyin`, `jieba`）与日文（`fugashi`, `unidic-lite`）所需的 G2P 解析依赖。

---

## 🧠 模型准备

### 1. STT 模型 (Whisper)

使用 `optimum-cli` 将 Whisper 多语言模型导出为 OpenVINO 格式：

```bash
pip install optimum[openvino]

# 导出多语言模型（以 whisper-large-v3-turbo 为例）
optimum-cli export openvino \
    --model openai/whisper-large-v3-turbo \
    --task automatic-speech-recognition \
    ./models/whisper-large-v3-turbo-openvino
```

### 2. TTS 模型 (Kokoro)

从 [Kokoro / kokoro-onnx](https://github.com/hexgrad/kokoro) 获取：
1. 模型权重：`kokoro.onnx`（OpenVINO 原生直接支持加载 ONNX）
2. 声音风格文件：`voices.npy`（内置中英日多语言声音，如 `zf_*` 中文、`jf_*` 日文、`af_*` 英文等）

---

## 🚀 启动与使用

### 1. 启动 STT 语音识别服务

```bash
# 最简启动：无需指定语言，完全由 HA 协议指定或由 Whisper 自动检测！
python wyoming-stt.py \
    --model-dir ./models/whisper-large-v3-turbo-openvino \
    --device GPU \
    --port 10020
```

### 2. 启动 TTS 语音合成服务

```bash
# 最简启动：无需指定语言和声音，自动注册全部声音并由 HA 或文本内容自动识别！
python wyoming-tts.py \
    --model-path ./models/kokoro.onnx \
    --voices-path ./models/voices.npy \
    --device GPU \
    --port 10021
```

---

## ⚙️ 命令行参数对照

### STT 参数 (`wyoming-stt.py`)

| 参数 | 简写 | 默认值 | 说明 |
|---|---|---|---|
| `--model-dir` | `-m` | (必填) | Whisper 模型目录 |
| `--device` | `-d` | `GPU` | 推理设备 (`CPU` / `GPU` / `AUTO` / `MULTI:CPU,GPU`) |
| `--language` | `-l` | `auto` | 默认语言（优先遵循 Wyoming 协议请求；若协议未指定则自动检测） |
| `--port` | `-p` | `10020` | 服务监听端口 |
| `--host` | | `0.0.0.0` | 监听地址 |

### TTS 参数 (`wyoming-tts.py`)

| 参数 | 简写 | 默认值 | 说明 |
|---|---|---|---|
| `--model-path` | `-m` | (必填) | Kokoro 模型文件路径 (.onnx 或 .xml) |
| `--voices-path` | `-v` | (必填) | 声音文件路径 (.npy) |
| `--voice-name` | | `None` (自动取首个声音) | 默认声音名称（HA 请求中指定的声音具有更高优先级） |
| `--device` | `-d` | `GPU` | 推理设备 (`CPU` / `GPU` / `AUTO` / `MULTI:CPU,GPU`) |
| `--language` | `-l` | `auto` | 默认语言（优先遵循 HA 协议，其次按声音前缀/文本内容自动识别） |
| `--speed` | `-s` | `1.0` | 语速倍率 |
| `--port` | `-p` | `10021` | 服务监听端口 |

---

## 🏠 Home Assistant 配置

在 Home Assistant 中进入 **配置** -> **设备与服务** -> **添加集成** -> 搜索 **Wyoming Protocol**：

1. 添加 STT：输入主机 IP，端口填 `10020`
2. 添加 TTS：输入主机 IP，端口填 `10021`

添加完成后，在 Home Assistant 的「语音助手 (Assist)」中，即可自由选择中、英、日等语言以及对应的 Kokoro 发音人。

## 📜 许可证

GNU General Public License v3.0 (GPL v3)
