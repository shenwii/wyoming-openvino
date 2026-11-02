# wyoming-openvino

[ English ](README.md) | [ 中文 ](README_zh.md) | [ 日本語 ](README_ja.md) | [ 한국어 ](README_ko.md)

---

Wyoming Protocol + Intel OpenVINO: High-performance Wake Word detection, Speech-to-Text (STT), and Text-to-Speech (TTS) microservices optimized for Intel NPUs (Core Ultra), Arc GPUs, integrated graphics (iGPUs), and CPUs, seamlessly integrated with Home Assistant.

## 🌟 Key Features

- **Full Hardware Acceleration & Native NPU Optimization (Intel Core Ultra)**:
  - **Wake Word (openWakeWord)**: Deeply optimized for Intel NPUs (Meteor Lake / Lunar Lake / Arrow Lake) with automatic static shape reshaping; ultra-low-power, always-on listening with graceful `NPU` -> `GPU` -> `CPU` fallback.
  - **STT (Whisper)**: Accelerated on `GPU` (Arc dGPU / iGPU), `CPU`, and `AUTO`.
  - **TTS (Kokoro)**: Optimized on `CPU`, with adaptive CPU thread scheduling.
- **Concurrent Multi-Model Wake Word Detection & YAML/JSON Config**:
  - Concurrently listen for multiple wake phrases (e.g. `Hey Jarvis`, `Alexa`, custom models) in a single audio stream with shared feature extraction pipelines.
  - Granular configuration via YAML or JSON: customizable `cutoff` (threshold), `refractory` (cooldown), and target `device` per model.
- **Smart Chunking & Low-Latency Streaming (Kokoro TTS)**:
  - Abbreviation and Decimal Guard: Prevents improper sentence breaking on `Dr.`, `Mr.`, `etc.`, `e.g.`, or floating-point numbers (`3.14`).
  - Trailing Punctuation Preservation: Retains sentence marks (`。！？!?`) so speech exhibits natural intonation and cadences.
  - Sentence-level streaming playback: First clause generates within ~100-300ms, streaming audio chunks immediately over Wyoming TCP to eliminate paragraph waiting.
- **Weighted Voice Blending / Mixing**:
  - Supports Kokoro-FastAPI voice blending syntax such as `zf_xiaobei(0.7)+af_heart(0.3)`, `af_bella(2)+af_sky(1)`, or `zf_xiaobei+af_heart` to dynamically mix speaker timbres.
- **Dynamic Multilingual & Built-in Internationalization (i18n)**:
  - Native support for English (`en`), Chinese (`zh`), Japanese (`ja`), and Korean (`ko`).
  - Console logs and Wyoming protocol service descriptions automatically adapt to the system locale (`LANG`, `LC_ALL`, or `WYOMING_LANG`).
- **Container Ready (Docker)**:
  - Fully decoupled subprojects (`wyoming-stt`, `wyoming-tts`, `wyoming-wakeword`), each with its own `Dockerfile`, `requirements.txt`, and executable shell entrypoint based on `docker.io/openvino/ubuntu24_runtime:latest`.

---

## 📁 Repository Structure

```
wyoming-openvino/
├── wyoming-stt/              # STT Service (Whisper OpenVINO)
│   ├── wyoming-stt.py        # Python service
│   ├── wyoming-stt.sh        # Shell entrypoint command
│   ├── i18n.py               # Internationalization engine
│   ├── lang/                 # Translation catalogs (en.json, zh.json, ja.json, ko.json)
│   ├── requirements.txt      # STT dependencies
│   └── Dockerfile            # Container definition
│
├── wyoming-tts/              # TTS Service (Kokoro OpenVINO)
│   ├── wyoming-tts.py        # Python service
│   ├── wyoming-tts.sh        # Shell entrypoint command
│   ├── i18n.py               # Internationalization engine
│   ├── lang/                 # Translation catalogs (en.json, zh.json, ja.json, ko.json)
│   ├── requirements.txt      # TTS dependencies
│   └── Dockerfile            # Multi-stage build container definition
│
├── wyoming-wakeword/         # Wake Word Service (openWakeWord OpenVINO)
│   ├── wyoming-wakeword.py   # Python service
│   ├── wyoming-wakeword.sh   # Shell entrypoint command
│   ├── wakeword_config.yaml  # Multi-model server configuration
│   ├── i18n.py               # Internationalization engine
│   ├── lang/                 # Translation catalogs (en.json, zh.json, ja.json, ko.json)
│   ├── requirements.txt      # Wake Word dependencies
│   └── Dockerfile            # Container definition
│
├── wakeword_trainer/         # End-to-end wake word training toolkit
├── README.md                 # English documentation (default)
├── README_zh.md              # Chinese documentation
├── README_ja.md              # Japanese documentation
├── README_ko.md              # Korean documentation
└── LICENSE                   # GPL-3.0 License
```

---

## 📦 Installation & Direct Execution

### 1. Wyoming STT (Speech-to-Text)

```bash
cd wyoming-stt
pip install -r requirements.txt

# Run with GPU acceleration and force Simplified Chinese output
./wyoming-stt.sh --model-dir /path/to/whisper-openvino --device GPU --to-simplified --port 10020

# Run with default settings (auto-detect language)
./wyoming-stt.sh --model-dir /path/to/whisper-openvino --device GPU --port 10020
```

### 2. Wyoming TTS (Text-to-Speech)

```bash
cd wyoming-tts
pip install -r requirements.txt

# Run Kokoro TTS on CPU (recommended 0.85 speed for natural cadence)
./wyoming-tts.sh --model-dir /path/to/Kokoro-82M-FP16-OpenVINO --device CPU --speed 0.85 --port 10021
```

### 3. Wyoming Wake Word (openWakeWord)

```bash
cd wyoming-wakeword
pip install -r requirements.txt

# Option A: Start using YAML config (Recommended for multiple wake words)
./wyoming-wakeword.sh --config wakeword_config.yaml

# Option B: Fast launch on Intel NPU
./wyoming-wakeword.sh --model models/hey_jarvis_v0.1.onnx --device NPU --cutoff 0.6 --port 10400

# Option C: Scan directory for models
./wyoming-wakeword.sh --model-dir ./models --device NPU --port 10400
```

---

## 🐳 Docker Deployment

All microservices use `docker.io/openvino/ubuntu24_runtime:latest` as the base image for out-of-the-box OpenVINO runtime support.

### 1. Build Docker Images

```bash
# Build STT image
docker build -t wyoming-stt ./wyoming-stt

# Build TTS image (Multi-stage build handles pyopenjtalk compilation)
docker build -t wyoming-tts ./wyoming-tts

# Build Wake Word image
docker build -t wyoming-wakeword ./wyoming-wakeword
```

### 2. Run Containers

#### STT Container (GPU access via `/dev/dri`)
```bash
docker run -d \
  --name wyoming-stt \
  --restart unless-stopped \
  --net host \
  --device /dev/dri \
  -v /path/to/whisper-openvino:/models/whisper:ro \
  -e LANG=en_US.UTF-8 \
  wyoming-stt \
  --model-dir /models/whisper \
  --device GPU \
  --port 10020
```

#### TTS Container
```bash
docker run -d \
  --name wyoming-tts \
  --restart unless-stopped \
  --net host \
  -v /path/to/Kokoro-82M-FP16-OpenVINO:/models/kokoro:ro \
  -e LANG=en_US.UTF-8 \
  wyoming-tts \
  --model-dir /models/kokoro \
  --device CPU \
  --speed 0.85 \
  --port 10021
```

#### Wake Word Container (Intel NPU access via `/dev/accel`)
```bash
docker run -d \
  --name wyoming-wakeword \
  --restart unless-stopped \
  --net host \
  --device /dev/accel \
  -v /path/to/models:/app/models \
  -v $(pwd)/wyoming-wakeword/wakeword_config.yaml:/app/wakeword_config.yaml:ro \
  -e LANG=en_US.UTF-8 \
  wyoming-wakeword \
  --config /app/wakeword_config.yaml
```

---

## 🌐 Internationalization (i18n)

Each microservice includes an i18n engine with translations cleanly decoupled into `lang/` directories (**English**: `en.json`, **Chinese**: `zh.json`, **Japanese**: `ja.json`, **Korean**: `ko.json`). Adding a new language simply requires placing a new JSON file (e.g. `lang/de.json`) into the respective `lang/` folder without touching Python code.

Language selection is controlled automatically via standard environment variables:
- `WYOMING_LANG` (highest priority)
- `LC_ALL`
- `LC_MESSAGES`
- `LANG` (e.g. `zh_CN.UTF-8`, `ja_JP.UTF-8`, `ko_KR.UTF-8`, `en_US.UTF-8`)

The selected language applies to:
1. Console and log outputs (`logger.info`, `logger.warning`, etc.)
2. Wyoming protocol service metadata descriptions (`AsrProgram`, `TtsProgram`, `WakeProgram`) displayed inside Home Assistant.

Example:
```bash
# Output logs and Wyoming metadata in Japanese
LANG=ja_JP.UTF-8 ./wyoming-tts.sh --model-dir /models/kokoro

# Output logs and Wyoming metadata in Chinese
LANG=zh_CN.UTF-8 ./wyoming-stt.sh --model-dir /models/whisper
```

---

## 🏠 Home Assistant Integration

In Home Assistant, navigate to **Settings** -> **Devices & Services** -> **Add Integration** -> search for **Wyoming Protocol**:

1. **Add STT**: Host IP, Port `10020`
2. **Add TTS**: Host IP, Port `10021`
3. **Add Wake Word**: Host IP, Port `10400`

Once added, go to **Settings** -> **Voice Assistants (Assist)** to configure your pipeline:
- Select `kokoro-openvino` as the TTS engine
- Select `wyoming-openvino-stt` as the Speech-to-Text engine
- Select `wyoming-openvino-wakeword` as the Wake Word engine

---

## 📜 License

GNU General Public License v3.0 (GPL v3)
