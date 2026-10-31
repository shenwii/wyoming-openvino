# wyoming-openvino

Wyoming 协议 + OpenVINO 推理引擎：在 Intel GPU/CPU 上运行语音识别（STT）和语音合成（TTS）服务，无缝对接 Home Assistant。

## 🌟 核心特性

- **多语言全面支持**：
  - **STT (Whisper)**：原生支持 **中文 (zh)**、**英语 (en)**、**日语 (ja)** 及 90+ 种多国语言，支持动态按请求切换语言与 `auto` 自动语言检测。
  - **TTS (Kokoro)**：内置多语言 G2P 管理器，无缝支持 **中文 (zh)**、**英语 (en)**、**日语 (ja)**，并可扩展支持西班牙语 (es)、法语 (fr)、印地语 (hi)、意大利语 (it)、巴西葡萄牙语 (pt) 等。自动按请求音色与文本特征匹配音素转换。
- **Intel 硬件加速**：
  - 充分利用 Intel 独立显卡（Arc 系列）、核显（iGPU）或 CPU。
  - 支持 `GPU`、`CPU`、`AUTO` 与 `MULTI:CPU,GPU` 混合推理。
- **Home Assistant 深度集成**：
  - 符合 Wyoming 标准协议，自动向 HA 宣告支持的完整语言列表与声音声库。

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
# 默认中文（支持在请求中动态指定 en / ja，或设为 auto）
python wyoming-stt.py \
    --model-dir ./models/whisper-large-v3-turbo-openvino \
    --device GPU \
    --language zh \
    --port 10020
```

- `--language` 可以设为 `zh`、`en`、`ja` 或 `auto`（自动检测说话语言）。
- 当 Home Assistant 的不同语音助理（如日文助手或英文助手）发起请求时，服务会自动切换为对应的语言识别，无需重启。

### 2. 启动 TTS 语音合成服务

```bash
python wyoming-tts.py \
    --model-path ./models/kokoro.onnx \
    --voices-path ./models/voices.npy \
    --voice-name zf_xiaobei \
    --device GPU \
    --language zh \
    --port 10021
```

- 服务会自动扫描 `voices.npy` 中的全部音色，并自动为每个音色匹配中/英/日等语种，向 Home Assistant 完整注册。
- 服务内置智能语言推断：
  1. 优先使用 Home Assistant 请求中指定的语言。
  2. 若未显式指定，根据声音名称前缀推断（如 `zf_` -> 中文, `af_` -> 英文, `jf_` -> 日文）。
  3. 亦可直接分析待合成文本的字符（包含平假名/片假名走日文 G2P，包含中文字走中文 G2P，英文字母走英文 G2P）。

---

## ⚙️ 常用命令行参数对照

### STT 参数 (`wyoming-stt.py`)

| 参数 | 简写 | 默认值 | 说明 |
|---|---|---|---|
| `--model-dir` | `-m` | (必填) | Whisper 模型目录 |
| `--device` | `-d` | `GPU` | 推理设备 (`CPU` / `GPU` / `AUTO` / `MULTI:CPU,GPU`) |
| `--language` | `-l` | `zh` | 默认识别语言代码 (`zh`, `en`, `ja`, `auto` 等) |
| `--port` | `-p` | `10020` | 服务监听端口 |
| `--host` | | `0.0.0.0` | 监听地址 |

### TTS 参数 (`wyoming-tts.py`)

| 参数 | 简写 | 默认值 | 说明 |
|---|---|---|---|
| `--model-path` | `-m` | (必填) | Kokoro 模型文件路径 (.onnx 或 .xml) |
| `--voices-path` | `-v` | (必填) | 声音文件路径 (.npy) |
| `--voice-name` | | `af_heart` | 默认声音名称 |
| `--device` | `-d` | `GPU` | 推理设备 (`CPU` / `GPU` / `AUTO` / `MULTI:CPU,GPU`) |
| `--language` | `-l` | `zh` | 默认语言 (`zh`, `en`, `ja`, `es`, `fr`, `hi`, `it`, `pt`) |
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
