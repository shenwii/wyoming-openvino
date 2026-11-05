# Wyoming TTS (Text-to-Speech) Quickstart Tutorial

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

This tutorial provides a copy-paste ready, plug-and-play Docker deployment guide for **Wyoming TTS (Kokoro-82M)** accelerated by Intel OpenVINO.

## 1. Download Model

This example uses the **Kokoro-82M INT8 quantized model** (combining ultra-low RAM footprint with high-fidelity speech synthesis).

> 💡 **Model Options**: If you prefer the unquantized FP16 version, you can substitute `Echo9Zulu/Kokoro-82M-FP16-OpenVINO`.

Run the following command to create the `models` directory and download the model:

```bash
mkdir -p models
docker run --rm -v ./models:/models docker.io/library/python:3.11 bash -c '
pip install -U huggingface_hub
hf download OpenVINO/Kokoro-82M-int8-ov --local-dir /models/Kokoro-82M-int8-ov
'
```

---

## 2. Docker Compose Configuration (`docker-compose.yml`)

Create a `docker-compose.yml` file in your working directory:

```yaml
services:
  wyoming-tts:
    image: ghcr.io/shenwii/wyoming-tts
    container_name: wyoming-tts
    ports:
      - "10021:10021"
    volumes:
      - ./models:/models
    environment:
      - LANG=en_US.UTF-8
    command:
      - --model-dir
      - /models/Kokoro-82M-int8-ov
      - --device
      - CPU
      - --port
      - "10021"
      - --speed
      - "0.85"
    restart: unless-stopped
```

### Key Details & Tips
- **Port**: The standard default port for TTS is `10021` (STT uses `10020`).
- **Matching Model Directory**: The download folder `/models/Kokoro-82M-int8-ov` must strictly match the path given to `--model-dir`.
- **Recommended Device (`--device CPU`)**:
  - Kokoro TTS is strongly recommended to run on **CPU**. OpenVINO's CPU backend (leveraging AVX2/AVX-512/AMX) generates the first audio chunk in just 100~300ms without GPU shader compilation stutter.
  - If you specifically prefer GPU execution, map `devices: - /dev/dri/renderD128:/dev/dri/renderD128` and set `--device` to `GPU`.
- **Speech Speed (`--speed 0.85`)**: Kokoro's baseline speed can be slightly fast; `0.85` delivers the most natural conversational cadence.
- **Voice Selection & Blending (`--voice`)**:
  - English voices: `af_heart` (female), `af_bella` (female), `am_adam` (male), `am_michael` (male), `bf_emma` (British female), `bm_george` (British male).
  - Voice blending is supported directly in the command, e.g.:
    `- --voice`
    `- af_heart(0.7)+am_adam(0.3)`

---

## 3. Start Service

Launch the container in the background using Docker Compose:

```bash
docker compose up -d
```

Check the logs to ensure the engine and voices initialized properly:

```bash
docker compose logs -f wyoming-tts
```

---

## 4. Home Assistant Integration

1. In Home Assistant, navigate to **Settings** -> **Devices & Services** -> **Add Integration**.
2. Search for and select **Wyoming Protocol**.
3. Enter your host IP address and port `10021`, then submit.
4. Go to **Settings** -> **Voice Assistants** -> select your assistant pipeline, and choose `kokoro-openvino` under **Text-to-speech**.
