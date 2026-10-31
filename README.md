# wyoming-openvino

Wyoming 协议 + OpenVINO 推理引擎：在 Intel GPU/CPU 上运行语音识别（STT）和语音合成（TTS）服务，无缝对接 Home Assistant。

## 项目简介

本项目提供两个独立的 Wyoming 协议服务：

- **STT 服务** (`wyoming-stt.py`)：基于 Whisper 模型的语音转文字
- **TTS 服务** (`wyoming-tts.py`)：基于 Kokoro 模型的文字转语音

所有推理均通过 [OpenVINO](https://github.com/openvinotoolkit/openvino) 引擎运行，支持 Intel 集显/独显 (GPU)、CPU 以及混合推理 (AUTO/MULTI)，解决了目前主流语音项目仅支持 NVIDIA CUDA 的问题。

## 环境要求

- Python >= 3.10
- Intel GPU 驱动（使用 GPU 推理时需要）
- [OpenVINO Runtime](https://docs.openvino.ai/2024/get-started/install-openvino.html)

## 安装

```bash
pip install -r requirements.txt
```

## 模型准备

### STT（Whisper）

使用 `optimum-cli` 将 Whisper 模型导出为 OpenVINO 格式：

```bash
pip install optimum[openvino]

# 导出模型（以 whisper-large-v3 为例）
optimum-cli export openvino \
    --model openai/whisper-large-v3 \
    --task automatic-speech-recognition \
    ./models/whisper-large-v3-openvino
```

或者从 HuggingFace 下载预转换的 OpenVINO 模型（搜索 `openvino whisper`）。

### TTS（Kokoro）

需要准备：
1. **模型文件**：Kokoro 的 OpenVINO IR 文件（`.xml` + `.bin`）或 ONNX 文件（`.onnx`）
2. **声音文件**：声音风格向量文件（`.npy`，字典格式 `{voice_name: style_vector}`）

可以从 [kokoro-onnx](https://github.com/hexgrad/kokoro-onnx) 获取 ONNX 模型，OpenVINO 可以直接加载 ONNX 格式。

## 使用方法

### 启动 STT 服务

```bash
python wyoming-stt.py \
    --model-dir ./models/whisper-large-v3-openvino \
    --device GPU \
    --language zh \
    --port 10020
```

### 启动 TTS 服务

```bash
python wyoming-tts.py \
    --model-path ./models/kokoro.onnx \
    --voices-path ./models/voices.npy \
    --voice-name af_heart \
    --device GPU \
    --language zh \
    --port 10021
```

### 查看所有参数

```bash
python wyoming-stt.py --help
python wyoming-tts.py --help
```

## 主要参数说明

### STT 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-dir` / `-m` | （必填） | OpenVINO Whisper 模型目录 |
| `--device` / `-d` | `GPU` | 推理设备：`CPU` / `GPU` / `AUTO` / `MULTI:CPU,GPU` |
| `--language` / `-l` | `zh` | 识别语言代码 |
| `--port` / `-p` | `10020` | 服务端口 |
| `--model-name` | `whisper` | 模型名称（用于 HA 显示） |

### TTS 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` / `-m` | （必填） | Kokoro 模型文件路径 |
| `--voices-path` / `-v` | （必填） | 声音风格文件路径 |
| `--voice-name` | `af_heart` | 声音名称 |
| `--device` / `-d` | `GPU` | 推理设备 |
| `--language` / `-l` | `zh` | 合成语言代码 |
| `--speed` / `-s` | `1.0` | 语速倍率 |
| `--sample-rate` | `24000` | 音频采样率 |
| `--port` / `-p` | `10021` | 服务端口 |

## Home Assistant 集成

在 Home Assistant 的 Wyoming 集成中添加：

- **STT**：`tcp://your-host:10020`
- **TTS**：`tcp://your-host:10021`

## 推理设备说明

OpenVINO 支持多种推理设备，通过 `--device` 参数指定：

| 设备 | 说明 |
|------|------|
| `CPU` | 使用 CPU 推理 |
| `GPU` | 使用 Intel 集成显卡或独立显卡 |
| `AUTO` | 自动选择最优设备 |
| `MULTI:CPU,GPU` | CPU 和 GPU 混合推理，提高吞吐量 |

## 许可证

GPL v3
