# Custom Wake Word Training Pipeline (wakeword_trainer) Quickstart Tutorial

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

This tutorial provides a copy-paste ready, Docker-based guide for **end-to-end training of custom wake word models**.
Powered by dual-engine synthetic speech generation (Kokoro OpenVINO + Microsoft Edge-TTS), acoustic room simulation (MIT RIR impulse responses), and negative sample mining, this tool automatically trains and exports **Intel NPU-optimized ONNX models** and **ESPHome / microWakeWord-compatible TFLite models**.

---

## 1. Clone Repository

```bash
git clone --depth=1 'https://github.com/shenwii/wyoming-openvino'
cd wyoming-openvino
```

---

## 2. Copy and Configure `config.yaml`

Copy the default template:

```bash
cp wakeword_trainer/config.sample.yaml ./config.yaml
```

Open `./config.yaml` in an editor and modify the top `wakeword` section:

```yaml
wakeword:
  phrase: "hey jarvis"          # Target wake word
  friend_name: "hey_jarvis_v1"  # Identifier (exports {friend_name}.onnx and {friend_name}.tflite)
  candidate_phrases:            # Aliases / acoustic variations
    - "jarvis jarvis"
    - "hi jarvis"
    - "hello jarvis"
  hard_negatives:               # Near-miss negative distractors (crucial for false trigger defense)
    - "hey harvest"
    - "hey service"
    - "turn on the lights"
    - "turn off the lights"
  language: "en"                # Language code (en, zh, ja, ko, etc.)
```

> 💡 **Training Device**: We recommend keeping `training.device` as `"cpu"` (fast training speed, zero driver conflicts).

---

## 3. Launch End-to-End Automated Training via Docker

Run the following command to download dependencies, fetch the offline Kokoro TTS model, and execute the entire pipeline inside a clean Docker container:

```bash
docker run --rm \
  --name wakeword-trainer \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  -v $(pwd):/app \
  -v $(pwd)/config.yaml:/app/wakeword_trainer/config.yaml \
  -d docker.io/library/python:3.11 bash -c '
cd /app/wakeword_trainer
apt update && apt install -y ffmpeg
pip install --upgrade pip
pip install -U huggingface_hub
hf download OpenVINO/Kokoro-82M-int8-ov --local-dir ../Kokoro-82M-int8-ov
pip install -r requirements.txt
pip install -r ../wyoming-tts/requirements.txt
python run_pipeline.py --config config.yaml
'
```

> 💡 **Tip**:
> - If your machine does not have an Intel GPU, simply omit the `--device /dev/dri/renderD128:/dev/dri/renderD128` line.
> - The container runs detached in the background.

---

## 4. Monitor Training Progress

Follow real-time output:

```bash
docker logs -f wakeword-trainer
```

The pipeline executes through these automated stages:
1. **Download Feature Models & Datasets**: Downloads openWakeWord base feature extractors, MIT room impulse responses (RIR), and background noise.
2. **Dual-Engine Synthetic Speech Generation**: Local Kokoro TTS and Microsoft Edge-TTS synthesize thousands of positive and negative samples with perturbations.
3. **Room Acoustic Simulation & Augmentation**: Convolves room acoustics and mixes real ambient noise.
4. **Neural Network Training & Multi-Format Export**: Trains the classifier and exports models.

---

## 5. Trained Model Deployment

Once finished, artifacts are saved in `./wakeword_trainer/output/`:

- **`{friend_name}.onnx`**:
  Intel NPU / GPU / CPU static shape ONNX model.
  👉 **Usage**: Copy to `wyoming-wakeword/models/` and add to `wakeword_config.yaml` to run in Home Assistant.
- **`{friend_name}.tflite`**:
  Standard 32-bit floating-point TensorFlow Lite model.
  👉 **Usage**: Deploy to smart microphones running **ESPHome microWakeWord** (e.g. ESP32-S3).
- **`{friend_name}.json`**:
  Metadata with recommended `cutoff` threshold settings.
- **`{friend_name}.pt`**:
  Raw PyTorch checkpoint for instant re-exporting (`python train.py --export-only`).
