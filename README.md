# wyoming-openvino

Wyoming 协议 + OpenVINO 推理引擎：在 Intel GPU/CPU 上运行语音识别（STT）和语音合成（TTS）服务，无缝对接 Home Assistant。

## 🌟 核心特性

- **直接支持官方原生 OpenVINO 模型包**：
  - **TTS (Kokoro)**：直接指定从 Hugging Face 下载的官方 OpenVINO 目录（包含 `openvino_model.xml`、`openvino_model.bin` 以及 `voices/*.bin` 文件），无需任何额外格式转换。
  - **STT (Whisper)**：直接指定 Optimum/OpenVINO 导出的 Whisper 模型目录。
- **命令行极致简单（统一使用 `--model-dir`）**：
  - STT 和 TTS 均只需一个 `--model-dir` 参数传入模型文件夹路径即可，自动查找模型与所有发音人音色。
- **全自动多语言与协议动态指定**：
  - 命令行 `--language` 默认为 `auto`（完全可以不传）。
  - **STT**：优先遵循 Home Assistant 请求中指定的语言；若未指定，由 Whisper 神经网络原生自动听声辨语（支持中、英、日等 90+ 种语言）。
  - **TTS**：优先遵循 Home Assistant 选择的发音人与语言；自动根据声音前缀（如 `zf_` -> 中文、`jf_` -> 日文、`af_` -> 英文）或文本内容（平假名/片假名/汉字/字母）智能匹配最佳 G2P 引擎。
- **Intel 硬件加速**：
  - 充分利用 Intel 独立显卡（Arc 系列）、核显（iGPU）或 CPU。
  - 支持 `GPU`、`CPU`、`AUTO` 与 `MULTI:CPU,GPU` 混合推理。

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

从 Hugging Face 下载的官方 Kokoro OpenVINO 目录结构通常为：
```
Kokoro-82M-OpenVINO/
├── openvino_model.xml     # 模型结构定义
├── openvino_model.bin     # 模型二进制权重
├── config.json            # 模型配置
├── openvino_config.json
├── voices/                # 发音人音色向量文件夹
│   ├── af_heart.bin       # 美音女声
│   ├── zf_xiaobei.bin     # 中文女声
│   ├── jf_teio.bin        # 日语女声
│   └── ...
└── ...
```
> [!NOTE]
> 每一个 `voices/*.bin` 文件代表一个独立的音色。服务启动时会自动扫描该目录下的全部音色，并自动向 Home Assistant 完整注册。

---

## 🚀 启动与使用

### 1. 启动 STT 语音识别服务

```bash
# 只需指定模型目录与推理设备：
python wyoming-stt.py \
    --model-dir /path/to/whisper-openvino \
    --device GPU \
    --port 10020
```

### 2. 启动 TTS 语音合成服务

```bash
# 只需指定下载的 Kokoro 官方目录即可：
python wyoming-tts.py \
    --model-dir /path/to/Kokoro-82M-OpenVINO \
    --device GPU \
    --port 10021
```

---

## ⚙️ 命令行常用参数对照

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
| `--device` | `-d` | `GPU` | 推理设备 (`CPU` / `GPU` / `AUTO` / `MULTI:CPU,GPU`) |
| `--language` | `-l` | `auto` | 默认语言（优先遵循 HA 协议，其次按音色/文本自动识别） |
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
