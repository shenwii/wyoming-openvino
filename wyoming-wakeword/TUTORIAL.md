# Wyoming Wake Word Quickstart Tutorial

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

This tutorial provides a copy-paste ready, plug-and-play Docker deployment guide for **Wyoming Wake Word (openWakeWord)** accelerated by Intel OpenVINO, featuring support for Intel NPU, GPU, and CPU hardware acceleration as well as concurrent multi-model listening.

## 1. Prepare Models and Directory

Create a `models` directory in your working directory:

```bash
mkdir -p models
```

- Place your trained custom wake word model (`.onnx` file exported from `wakeword_trainer`) into `./models`.
- If using standard pre-trained models (e.g. `hey_jarvis_v0.1.onnx`, `alexa_v0.1.onnx`), the service will automatically download the necessary acoustic feature extractors (`melspectrogram.onnx` and `embedding_model.onnx`) on initial launch.

---

## 2. Obtain and Edit `wakeword_config.yaml`

Download the default configuration file:

```bash
wget https://raw.githubusercontent.com/shenwii/wyoming-openvino/refs/heads/main/wyoming-wakeword/wakeword_config.yaml -O wakeword_config.yaml
```

Open `wakeword_config.yaml` in an editor and add your model under the `models` list:

```yaml
models:
  - name: "hey_jarvis"
    phrase: "Hey Jarvis"
    model_path: "models/hey_jarvis_v0.1.onnx"
    cutoff: 0.60
    trigger_level: 2
    sliding_window_size: 3
    trailing_fall: true
    refractory: 2.0
    device: "NPU"
```

---

## 3. Docker Compose Configuration (`docker-compose.yml`)

Create a `docker-compose.yml` file in the same directory:

```yaml
services:
  wyoming-wakeword:
    image: ghcr.io/shenwii/wyoming-wakeword
    container_name: wyoming-wakeword
    ports:
      - "10400:10400"
    devices:
      - /dev/dri/renderD128:/dev/dri/renderD128
      - /dev/accel/accel0:/dev/accel/accel0
    volumes:
      - ./models:/app/models
      - ./wakeword_config.yaml:/app/wakeword_config.yaml
    environment:
      - LANG=en_US.UTF-8
    command:
      - --config
      - wakeword_config.yaml
    restart: unless-stopped
```

### Key Details & Tips
- **Container Working Directory (`WORKDIR`)**: The container's default workdir is `/app`. By mounting to `/app/models` and `/app/wakeword_config.yaml`, you can cleanly use relative paths like `models/hey_jarvis_v0.1.onnx` and `wakeword_config.yaml`.
- **Hardware Device Mapping (`devices`)**:
  - `/dev/accel/accel0:/dev/accel/accel0`: Maps the Intel Core Ultra NPU for ultra-low-power, always-on listening. If your system lacks an NPU, remove or comment out this line.
  - `/dev/dri/renderD128:/dev/dri/renderD128`: Maps Intel iGPU or Arc discrete GPU.
  - The service automatically falls back gracefully: `NPU` -> `GPU` -> `CPU`.

---

## 4. Configuration Parameter Breakdown

| Parameter | Default | Explanation & Tuning Advice |
| :--- | :--- | :--- |
| `cutoff` | `0.70` | **Confidence score threshold** (0.0 to 1.0). Typically set between 0.60 and 0.80. Lower values increase sensitivity (and false positives); higher values require louder/clearer pronunciation. |
| `trigger_level` | `2` | **Consecutive hit count**. Sudden transient noise (like TV background audio) can spike above 0.9 for a single frame. Requiring N consecutive matching frames (each frame is 80ms) suppresses false-positive glitches. |
| `sliding_window_size` | `3` | **Rolling average window size**. Maintains a rolling average over the last N frames (1 frame = 80ms, 3 frames ≈ 240ms) to smooth out isolated spikes. |
| `trailing_fall` | `true` | **Peak Fall-Off Release Mechanism (Anti-Early-Truncation)**. Prevents dirty STT data; see detailed explanation below. |
| `max_trailing_frames` | `8` | **Max waiting frames past peak** (~640ms). Acts as a deadlock safety guard in case sustained loud speech or high background noise prevents the score from dropping. |
| `trailing_raw_cutoff` | `0.20` | **Instant drop threshold**. When using larger sliding windows, the rolling average lags behind speech termination. If the latest raw frame score plunges below this threshold, the event triggers immediately. |
| `refractory` | `2.0` | **Cooldown period (seconds)**. Ignores detections for this model during the specified duration to prevent double triggers. |

### 💡 Why `trailing_fall` is Critical
When a wake word is relatively long (~1 second), a standard detection engine might hit the `cutoff` threshold early (e.g. at 500ms, when only half the phrase has been spoken).
- **Without `trailing_fall`**: The engine triggers Home Assistant immediately. HA halts listening for the wake word and starts the STT audio capture right away. The remaining half of the wake word ends up in the STT buffer as "dirty audio", corrupting transcription (e.g. "Jarvis turn on the lights" becomes transcribed as "is turn on the lights").
- **With `trailing_fall: true`**: Once the score meets the threshold, the service does not return immediately. Instead, it enters a `PENDING` state and tracks the score contour (e.g. `0.1 -> 0.5 -> 0.7 -> 0.9 -> 0.9 -> 0.7 -> 0.3`). Only when the speech completes and the score drops off the peak does it release the event. This perfectly aligns with the user's natural 0.3s speech pause before giving commands, ensuring pristine audio sent to STT!

---

## 5. Start Service & Home Assistant Integration

Start the container:
```bash
docker compose up -d
```

View logs:
```bash
docker compose logs -f wyoming-wakeword
```

### Home Assistant Integration
1. In Home Assistant, go to **Settings** -> **Devices & Services** -> **Add Integration**.
2. Search for **Wyoming Protocol**, enter your host IP and port `10400`.
3. Go to **Settings** -> **Voice Assistants** -> select your assistant pipeline, and choose your wake word model under **Wake word**.
