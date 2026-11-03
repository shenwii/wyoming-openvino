# Wyoming / openWakeWord NPU Wake Word Training Suite

An end-to-end training suite based on **[openWakeWord](https://github.com/dscripka/openWakeWord)** and **[MicroWakeWordV2Trainer](https://github.com/JohnnyPrimus/MicroWakeWordV2Trainer)**, customized for **Intel NPU (OpenVINO VPUX / Core Ultra)** and hardware acceleration.

---

## 🌟 Key Features

1. **Centralized User Configuration (`config.yaml`)**:
   - Customize everything in the `wakeword` section: target phrase, `friend_name` (model identifier without spaces), aliases, and hard negatives (near-misses).
   - Multi-language support (`en`, `zh`, `ja`, etc.).
2. **Dual-Engine Synthetic Speech Generation (`generate_tts.py`)**:
   - **Local Engine (Kokoro OpenVINO TTS)**: High-speed local offline neural synthesis with voice blending and speed variations.
   - **Cloud Engine (Microsoft Edge-TTS)**: High-fidelity neural voices with pitch, rate, and volume perturbations.
3. **Realistic Acoustic Simulation (`augment_data.py`)**:
   - Convolves MIT Environmental RIR (Room Impulse Responses) to simulate room acoustics.
   - Mixes ambient background noise at random SNRs (5~25dB).
   - Random time shifting and silence padding for robustness.
4. **Hard Negative & ACAV100M Dataset Blending**:
   - Synthesizes confusable near-misses as hard negatives to strongly suppress false positives.
   - Supports openWakeWord ACAV100M 2000-hour negative speech features.
5. **Intel NPU Dedicated Optimization (`train.py`)**:
   - **Static Input Shape**: Exports strict static `[1, 16, 96] -> [1, 1]` ONNX models, preventing NPU driver crashes and dynamic reallocation latency.
   - **Pure Operator Graph**: Uses `Gemm (Linear) + ReLU + Sigmoid` for 100% NPU hardware acceleration with zero CPU fallback.
   - **Flexible Hardware Acceleration**: Supports training on CPU, CUDA, or Intel GPU (via PyTorch 2.4+ native `xpu` or IPEX) and feature extraction on OpenVINO `GPU`.

---

## 📁 Directory Structure

```text
wakeword_trainer/
├── config.sample.yaml     # Sample configuration file (English default)
├── config.sample_zh.yaml  # Sample configuration file (Chinese)
├── config.sample_ja.yaml  # Sample configuration file (Japanese)
├── config.sample_ko.yaml  # Sample configuration file (Korean)
├── requirements.txt       # Training suite dependencies
├── i18n.py                # Internationalization module
├── lang/                  # i18n translation catalogs (en, zh, ja, ko)
├── setup_libs.py          # Downloads base models (melspectrogram.onnx, embedding_model.onnx)
├── download_data.py       # Downloads RIRs, noise datasets, and negative features
├── generate_tts.py        # Dual-engine synthetic speech generation
├── augment_data.py        # Acoustic simulation and audio augmentation
├── train.py               # Feature extraction, DNN training, and NPU ONNX export
├── run_pipeline.py        # Automated end-to-end pipeline runner
├── README.md              # Documentation (English default)
├── data/                  # [Generated] Synthetic speech, augmented audio, RIRs, noise
├── lib/                   # [Generated] Base models and upstream repositories
└── output/                # [Generated] Trained ONNX models and metadata JSON
```

---

## 🛠️ Installation

Install the Python dependencies:

```bash
cd wakeword_trainer
pip install -r requirements.txt
```

> **Note**:
> - Audio processing requires `ffmpeg` (Ubuntu/Debian: `sudo apt install ffmpeg`).
> - Local Kokoro TTS requires the OpenVINO environment. If Kokoro is not found, Edge-TTS is automatically used.

---

## ⚙️ Configuration (`config.yaml`)

Copy one of the sample configuration files:
```bash
cp config.sample.yaml config.yaml
```

The user only needs to edit the `wakeword` section:
```yaml
wakeword:
  phrase: "hey jarvis"            # Primary target phrase
  friend_name: "hey_jarvis_v1"    # Friendly identifier without spaces (determines output filename)
  candidate_phrases:              # Positive aliases
    - "jarvis jarvis"
    - "hi jarvis"
  hard_negatives:                 # Confusable words & near-misses
    - "hey harvest"
    - "hey service"
    - "turn on the lights"
  language: "en"                  # Language code
```

Hardware device configuration (under `training:`):
```yaml
training:
  # "auto", "cpu", "cuda", or "xpu" (Intel Arc / Core Ultra GPU via PyTorch XPU)
  device: "auto"

  # "auto", "CPU", or "GPU" (Intel GPU via OpenVINO)
  feature_device: "auto"
```

---

## 🚀 Training Instructions

### Method 1: Automated End-to-End Run (Recommended)

Run the full pipeline:
```bash
python run_pipeline.py --config config.yaml
```

For a quick test with fewer samples:
```bash
python run_pipeline.py --quick-mode
```

---

### Method 2: Step-by-Step Execution

1. **Setup base feature models**:
   ```bash
   python setup_libs.py --models-only
   ```
2. **Download acoustic data**:
   ```bash
   python download_data.py --rirs --noise
   ```
3. **Generate synthetic speech**:
   ```bash
   python generate_tts.py --config config.yaml
   ```
4. **Augment audio with RIR and noise**:
   ```bash
   python augment_data.py --config config.yaml
   ```
5. **Train DNN and export models (ONNX + TFLite)**:
   ```bash
   python train.py --config config.yaml --steps 20000 --batch-size 128
   ```
   > Training outputs full artifacts in `output/`:
   > - `output/{friend_name}.onnx`: Intel NPU / OpenVINO static acceleration model
   > - `output/{friend_name}.tflite`: Standard float32 TensorFlow Lite model
   > - `output/{friend_name}.pt`: PyTorch weights checkpoint (for fast re-export)
   > - `output/{friend_name}.json`: Model metadata & recommended cutoff thresholds

---

## 📦 TFLite Model Export & Deployment (microWakeWord / ESPHome)

The training suite natively supports exporting **TensorFlow Lite (`.tflite`)** models, fully compatible with **Home Assistant openWakeWord**, **ESPHome**, and **microWakeWord** edge devices.

### 1. Requirements

Exporting `.tflite` requires TensorFlow (included in `requirements.txt`):
```bash
pip install tensorflow>=2.13.0
```
*(If TensorFlow is not installed, the trainer gracefully skips TFLite export and saves ONNX. You can export TFLite later in seconds using `--export-tflite` without re-training!)*

### 2. Standalone Export & Quantization Commands (Instant, No Re-training)

Using saved PyTorch weights (`output/{friend_name}.pt`), export desired formats anytime:

```bash
# Option A: Export all formats (ONNX + TFLite Float32)
python train.py --config config.yaml --export-only

# Option B: Export only TFLite model (Float32, ~390 KB)
python train.py --config config.yaml --export-tflite

# Option C: Export with INT8 dynamic range quantization (~100 KB, perfect for ESP32 / microcontrollers)
python train.py --config config.yaml --export-tflite --quantize
# Outputs: output/{friend_name}.tflite (Float32) and output/{friend_name}_int8.tflite (INT8)

# Option D: Export from custom checkpoint weights
python train.py --config config.yaml --export-only --checkpoint output/my_model.pt
```

### 3. Model Format Comparison & Selection

| Format | Typical Size | Recommended Hardware & Target | Runtime Environment |
| :--- | :---: | :--- | :--- |
| **`.onnx`** | ~400 KB | **Intel NPU** (Core Ultra Meteor/Lunar/Arrow Lake), Intel iGPU/Arc GPU, x86/x64 Servers | `wyoming-wakeword`, OpenVINO Runtime, ONNXRuntime |
| **`.tflite` (Float32)** | ~390 KB | Raspberry Pi, Linux/Windows Edge boxes, Home Assistant Host | Home Assistant `openWakeWord` addon, `wyoming-wakeword` (natively supported by OpenVINO) |
| **`_int8.tflite` (INT8)** | **~100 KB** | **ESP32-S3** microcontrollers, memory-constrained IoT nodes | **ESPHome**, **microWakeWord** firmware |

---

## 🎯 Multi-Platform Deployment Guide

### 1. Wyoming Wake Word Service (Intel NPU / GPU / CPU)

`wyoming-wakeword` is built on OpenVINO, which **natively supports directly loading both `.onnx` and `.tflite` models**:

```bash
cd ../wyoming-wakeword

# Run ONNX model (preferred on Intel NPU)
python wyoming-wakeword.py \
    --model ../wakeword_trainer/output/hey_jarvis_v1.onnx \
    --device NPU \
    --cutoff 0.55 \
    --refractory 2.0

# Run TFLite model (OpenVINO native acceleration)
python wyoming-wakeword.py \
    --model ../wakeword_trainer/output/hey_jarvis_v1.tflite \
    --device NPU \
    --cutoff 0.55
```

### 2. Home Assistant openWakeWord Addon

Copy the generated `{friend_name}.tflite` (or `{friend_name}.onnx`) into Home Assistant's custom model directory:
```bash
cp output/hey_jarvis_v1.tflite /share/openwakeword/
```
Reload openWakeWord in Home Assistant, and select your custom wake word in the Voice Assistant settings.

### 3. ESPHome / microWakeWord (ESP32-S3 Satellite)

Upload `{friend_name}_int8.tflite` to your ESPHome configuration:
```yaml
# ESPHome YAML example
micro_wake_word:
  models:
    - model: "hey_jarvis_v1_int8.tflite"
      probability_cutoff: 0.55
      sliding_window_average_size: 2
```

---

## 🔍 Pre-flight Diagnostics & Cache Management

```bash
# 1. Audit all 8 dataset components in seconds before training
python train.py --check-data

# 2. Force re-extracting features (invalidating stale or empty cache)
python train.py --recompute-features
```
