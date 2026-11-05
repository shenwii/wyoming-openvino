# Wyoming STT (Speech-to-Text) Quickstart Tutorial

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

This tutorial provides a copy-paste ready, plug-and-play Docker deployment guide for **Wyoming STT (Whisper)** accelerated by Intel OpenVINO.

## 1. Download Model

Whisper models must be in the OpenVINO Intermediate Representation (IR) format. This example uses the **Whisper Medium FP16** model (which offers an ideal balance between transcription accuracy and resource utilization).

> 💡 **Other Model Sizes & Quantization**: If you require a smaller/faster footprint (e.g. `base`, `small`) or higher accuracy (`large-v3-turbo`, `large-v3`), browse the [OpenVINO Hugging Face Organization](https://huggingface.co/OpenVINO) and swap the model ID.

Run the following command to create the `models` directory and download the model:

```bash
mkdir -p models
docker run --rm -v ./models:/models docker.io/library/python:3.11 bash -c '
pip install -U huggingface_hub
hf download OpenVINO/whisper-medium-fp16-ov --local-dir /models/whisper-medium-fp16-ov
'
```

---

## 2. Docker Compose Configuration (`docker-compose.yml`)

Create a `docker-compose.yml` file in your working directory:

```yaml
services:
  wyoming-stt:
    image: ghcr.io/shenwii/wyoming-stt
    container_name: wyoming-stt
    ports:
      - "10020:10020"
    devices:
      - /dev/dri/renderD128:/dev/dri/renderD128
    volumes:
      - ./models:/models
    environment:
      - LANG=en_US.UTF-8
    command:
      - --model-dir
      - /models/whisper-medium-fp16-ov
      - --device
      - GPU
      - --port
      - "10020"
    restart: unless-stopped
```

### Key Details & Tips
- **Port**: The standard default port for STT is `10020` (TTS uses `10021`).
- **Hardware Acceleration (`devices`)**:
  - `/dev/dri/renderD128:/dev/dri/renderD128`: Passes through Intel integrated graphics (iGPU) or Intel Arc discrete GPU for GPU acceleration.
  - If running in a CPU-only environment, omit the `devices` section and set `--device` to `CPU` in `command`.
- **Language Detection & Switching**:
  - By default, Whisper automatically detects the spoken language (`--language auto`). You can lock to a specific language with `--language en` if desired.
- **Chinese Conversion (`--to-simplified`)**: If transcribing Chinese, Whisper frequently outputs Traditional Chinese; pass `--to-simplified` to force Simplified Chinese output.
- **Environment (`LANG=en_US.UTF-8`)**: Sets console logs and Wyoming metadata to English.

---

## 3. Start Service

Launch the container in the background using Docker Compose:

```bash
docker compose up -d
```

View the logs to verify that the model has loaded successfully:

```bash
docker compose logs -f wyoming-stt
```

---

## 4. Home Assistant Integration

1. In Home Assistant, navigate to **Settings** -> **Devices & Services** -> **Add Integration**.
2. Search for and select **Wyoming Protocol**.
3. Enter your host IP address and port `10020`, then submit.
4. Go to **Settings** -> **Voice Assistants** -> select your assistant pipeline, and choose `wyoming-openvino-stt` under **Speech-to-text**.
