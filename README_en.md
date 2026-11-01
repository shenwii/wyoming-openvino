# wyoming-openvino

[ 中文 ](README.md) | [ English ](README_en.md) | [ 日本語 ](README_ja.md)

---

Wyoming Protocol + Intel OpenVINO: High-performance Speech-to-Text (STT) and Text-to-Speech (TTS) microservices optimized for Intel Arc GPUs, integrated graphics (iGPUs), and CPUs, seamlessly integrated with Home Assistant.

## 🌟 Key Features

- **Native Support for Official OpenVINO Model Hubs**:
  - **TTS (Kokoro)**: Directly points to official OpenVINO directories from Hugging Face (containing `openvino_model.xml`, `openvino_model.bin`, and `voices/*.pt` or `voices/*.bin`), with zero extra conversion needed.
  - **STT (Whisper)**: Direct support for Whisper models exported via Optimum/OpenVINO.
- **Unified & Minimal CLI Experience**:
  - Both STT and TTS services accept a single `--model-dir` parameter to automatically resolve model graphs, weights, and all available speaker voicepacks.
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
- **Intel Hardware Acceleration**:
  - Supports `GPU` (Intel Arc dGPU / iGPU), `CPU`, `AUTO`, and `MULTI:CPU,GPU` hybrid execution.

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

# Profile C: Run both STT and TTS
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

### 3. Standalone Testing & WAV Export (`test_tts.py`)

Test speech synthesis and voice blending locally without Home Assistant and output standard WAV audio files:

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

---

## 🏠 Home Assistant Integration

In Home Assistant, navigate to **Settings** -> **Devices & Services** -> **Add Integration** -> search for **Wyoming Protocol**:

1. Add STT: Host IP, Port `10020`
2. Add TTS: Host IP, Port `10021`

Once added, navigate to **Voice Assistants (Assist)** to freely select available languages and Kokoro speakers.

## 📜 License

GNU General Public License v3.0 (GPL v3)
