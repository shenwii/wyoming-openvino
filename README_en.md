# wyoming-openvino

[ 中文 ](README.md) | [ English ](README_en.md) | [ 日本語 ](README_ja.md)

---

Wyoming Protocol + Intel OpenVINO: High-performance Wake Word detection, Speech-to-Text (STT), and Text-to-Speech (TTS) microservices optimized for Intel NPUs (Core Ultra), Arc GPUs, integrated graphics (iGPUs), and CPUs, seamlessly integrated with Home Assistant.

## 🌟 Key Features

- **Full Hardware Acceleration & Native NPU Optimization (Intel Core Ultra)**:
  - **Wake Word (openWakeWord)**: Deeply optimized for Intel NPUs (Meteor Lake / Lunar Lake / Arrow Lake) with automatic static shape reshaping; ultra-low-power, always-on listening with graceful `NPU` -> `GPU` -> `CPU` fallback.
  - **STT (Whisper)**: Accelerated on `GPU` (Arc dGPU / iGPU), `CPU`, and `AUTO`.
  - **TTS (Kokoro)**: Recommended on `CPU`, with adaptive CPU thread scheduling.
- **Concurrent Multi-Model Wake Word Detection & YAML/JSON Config**:
  - Concurrently listen for multiple wake phrases (e.g. `Hey Jarvis`, `Alexa`, custom models) in a single audio stream with shared feature extraction pipelines.
  - Granular configuration via YAML or JSON: customizable `cutoff` (threshold), `refractory` (cooldown), and target `device` per model.
- **Native Support for Official OpenVINO Model Hubs**:
  - **TTS (Kokoro)**: Directly points to official OpenVINO directories from Hugging Face (containing `openvino_model.xml`, `openvino_model.bin`, and `voices/*.pt` or `voices/*.bin`), with zero extra conversion needed.
  - **STT (Whisper)**: Direct support for Whisper models exported via Optimum/OpenVINO.
  - **Wake Word**: Direct support for `.onnx`, `.tflite`, and OpenVINO `.xml` models trained or exported from openWakeWord.
- **Unified & Minimal CLI Experience**:
  - Simple commands with automatic model discovery via `--config`, `--model-dir`, or `--model`.
- **Smart Chunking & Low-Latency Streaming (Learned from Kokoro-FastAPI)**:
  - Abbreviation and Decimal Guard: Prevents improper sentence breaking on `Dr.`, `Mr.`, `etc.`, `e.g.`, or floating numbers (e.g., `3.14`).
  - Acoustic Punctuation Retention: Preserves trailing punctuation (`。！？!?`) so Kokoro generates natural intonation, cadences, and pauses.
  - Sentence-level streaming playback: First sentence speaks in ~1 second via Wyoming audio chunks, completely eliminating long waits for long paragraphs.
- **Weighted Voice Blending / Mixing**:
  - Supports Kokoro-FastAPI voice blending syntax such as `zf_xiaobei(0.7)+af_heart(0.3)`, `af_bella(2)+af_sky(1)`, or `zf_xiaobei+af_heart` to dynamically mix speaker timbres with normalized weights.
- **Dynamic Multilingual Support & Auto Language Routing**:
  - `--language` defaults to `auto` (can be omitted entirely).
  - **STT**: Prioritizes language specified by Home Assistant Wyoming requests; falls back to Whisper's native automatic language detection (supporting 90+ languages including English, Chinese, and Japanese).
  - **TTS**: Automatically routes G2P engines based on the requested voice prefix (`zf_*` -> Chinese, `jf_*` -> Japanese, `af_*` -> English) or through text character heuristics (Kanji/Kana/Latin). Chinese-English mixed text routes embedded English through `en_callable` with `EspeakFallback` fallback, while pure English clauses seamlessly use native English phonemes.

---

## 📦 Requirements & Installation

### 1. System Requirements
- Python >= 3.10
- Intel GPU Drivers & OpenVINO Runtime

### 2. Dependency Installation

Choose the minimal requirement profile based on your deployment needs:

```bash
# Profile A: Run STT (Speech-to-Text) only (Lightweight, OpenVINO GenAI & Wyoming only)
pip install -r requirements-stt.txt

# Profile B: Run TTS (Text-to-Speech) only (Includes Kokoro, voicepack parser, and G2P dependencies)
pip install -r requirements-tts.txt

# Profile C: Run Wake Word detection only (OpenVINO, Wyoming, PyYAML)
pip install -r requirements-wakeword.txt

# Profile D: Full installation (STT + TTS + Wake Word)
pip install -r requirements.txt
```

---

## 🧠 Model Preparation

### 1. STT Model (Whisper)

A standard OpenVINO Whisper model directory:
```
whisper-openvino/
├── openvino_model.xml
├── openvino_model.bin
├── config.json
├── tokenizer.json
└── ...
```

### 2. TTS Model (Kokoro)

Standard Hugging Face layout (e.g., `Kokoro-82M-FP16-OpenVINO`):
```
Kokoro-82M-FP16-OpenVINO/
├── openvino_model.xml     # Network topology
├── openvino_model.bin     # Model weights
├── config.json            # Model configuration
├── openvino_config.json
├── voices/                # Speaker voicepack directory
│   ├── af_heart.pt        # American English female
│   ├── zf_xiaobei.pt      # Chinese female
│   ├── jf_teio.pt         # Japanese voice
│   └── ...
└── ...
```
> [!NOTE]
> Every file inside `voices/` represents an individual speaker voice. The service automatically scans and advertises all voices to Home Assistant upon startup.

### 3. Wake Word Models (openWakeWord)

Located inside the `models/` folder:
```
models/
├── melspectrogram.onnx      # Feature extraction backbone (auto-downloaded if not found)
├── embedding_model.onnx     # Feature extraction backbone (auto-downloaded if not found)
├── hey_jarvis_v0.1.onnx     # Trained or pre-trained wake word classifier
├── alexa_v0.1.onnx
└── ...
```
> [!NOTE]
> Supports `.onnx`, `.tflite`, and OpenVINO `.xml` wake models. Includes automatic static shape reshaping for Intel Core Ultra NPU compatibility with fallback to GPU/CPU.

---

## 🚀 Quick Start

### 1. Start STT (Speech-to-Text) Service

```bash
python wyoming-stt.py \
    --model-dir /path/to/whisper-openvino \
    --device GPU \
    --port 10020
```

### 2. Start TTS (Text-to-Speech) Service

```bash
python wyoming-tts.py \
    --model-dir /path/to/Kokoro-82M-FP16-OpenVINO \
    --device CPU \
    --speed 0.85 \
    --port 10021
```

> [!TIP]
> **Speed Recommendation**: The default `1.0` speed can sound slightly hurried for Chinese and Japanese. Setting `--speed 0.85` (or `0.8`) delivers a significantly more natural and relaxed pacing.
> **Device Recommendation**: `CPU` is recommended for Kokoro-82M-FP16 due to IR operator compatibility and fast response latency.

### 3. Start Wake Word Service

**Option A: Using YAML configuration file (Recommended for multi-model detection)**
```bash
python wyoming-wakeword.py --config wakeword_config.yaml
```

**Option B: CLI fast launch targeting NPU**
```bash
python wyoming-wakeword.py \
    --model models/hey_jarvis_v0.1.onnx \
    --device NPU \
    --cutoff 0.6 \
    --port 10400
```

**Option C: Auto-scan model directory**
```bash
python wyoming-wakeword.py --model-dir ./models --device NPU
```

### 4. Standalone Testing Tools

#### A. Test TTS Speech & Export WAV (`test_tts.py`)
```bash
# Basic Mandarin synthesis
python test_tts.py "我来测试一下语音合成的效果。" --model-dir /path/to/Kokoro-82M-FP16-OpenVINO -o test.wav

# Voice blending (70% Chinese female + 30% English female)
python test_tts.py "今天天气真好，我们来测试 OpenVINO 语音合成。" \
    --model-dir /path/to/Kokoro-82M-FP16-OpenVINO \
    -v "zf_xiaobei(0.7)+af_heart(0.3)" \
    --speed 0.85 \
    -o blend.wav
```

#### B. Test Wake Word Detection (`test_wakeword.py`)
```bash
# Mode 1: Evaluate a WAV file directly using local OpenVINO engine
python test_wakeword.py sample.wav --config wakeword_config.yaml --device NPU

# Mode 2: Test as a client connecting to a live Wyoming Wake Word server
python test_wakeword.py sample.wav --server 127.0.0.1:10400
```

---

## ⚙️ CLI Options Reference

### STT Options (`wyoming-stt.py`)

| Option | Flag | Default | Description |
|---|---|---|---|
| `--model-dir` | `-m` | (Required) | Directory containing OpenVINO Whisper model files |
| `--device` | `-d` | `GPU` | Inference target (`CPU` / `GPU` / `AUTO` / `MULTI:CPU,GPU`) |
| `--language` | `-l` | `auto` | Default language (prioritizes HA Wyoming request; auto-detects otherwise) |
| `--port` | `-p` | `10020` | Server listening port |
| `--host` | | `0.0.0.0` | Server bind address |

### TTS Options (`wyoming-tts.py`)

| Option | Flag | Default | Description |
|---|---|---|---|
| `--model-dir` | `-m` | (Required) | Kokoro model directory (automatically locates xml/bin and voices/) |
| `--device` | `-d` | `CPU` | Inference target (`CPU` recommended / `GPU` / `AUTO`) |
| `--language` | `-l` | `auto` | Default language (prioritizes HA Wyoming request, or infers from text/voice) |
| `--speed` | `-s` | `1.0` | Speech speed multiplier (recommended `0.85` or `0.8` for natural pacing) |
| `--port` | `-p` | `10021` | Server listening port |

### Wake Word Options (`wyoming-wakeword.py`)

| Option | Flag | Default | Description |
|---|---|---|---|
| `--config` | `-c` | `None` | Path to YAML or JSON configuration file |
| `--model` | `-m` | `None` | Path to wake word model (can be specified multiple times) |
| `--model-dir` | | `None` | Automatically scan and load all models in directory |
| `--device` | `-d` | `NPU` | Inference target (`NPU` prioritized / `GPU` / `CPU` / `AUTO` with fallback) |
| `--cutoff` | | `0.5` | Default wake score threshold (0.0 ~ 1.0) |
| `--refractory` | | `2.0` | Cooldown period after detection in seconds |
| `--port` | `-p` | `10400` | Server listening port (HA Wyoming openWakeWord default: 10400) |
| `--host` | | `0.0.0.0` | Server bind address |
| `--log-level` | | `INFO` | Logging verbosity (`DEBUG` / `INFO` / `WARNING` / `ERROR`) |
| `--melspec-model` | | `None` | Path to custom MelSpectrogram feature extractor model |
| `--embedding-model` | | `None` | Path to custom Embedding feature extractor model |

---

## 🏠 Home Assistant Integration

In Home Assistant, navigate to **Settings** -> **Devices & Services** -> **Add Integration** -> search for **Wyoming Protocol**:

1. Add STT: Host IP, Port `10020`
2. Add TTS: Host IP, Port `10021`
3. Add Wake Word: Host IP, Port `10400`

Once added, navigate to **Voice Assistants (Assist)** to select your voice models and enable wake word detection with ultra-low latency.

## 📜 License

GNU General Public License v3.0 (GPL v3)
