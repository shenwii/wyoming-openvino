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
5. **Train DNN and export NPU-optimized ONNX**:
   ```bash
   python train.py --config config.yaml --steps 20000 --batch-size 128
   ```
6. **Export-only from trained weights (no re-training needed)**:
   ```bash
   python train.py --config config.yaml --export-only
   ```

Outputs are saved in `output/{friend_name}.onnx`, `output/{friend_name}.pt`, and `output/{friend_name}.json`.

---

## 🎯 Integration with Wyoming Wake Word

Run directly with `wyoming-wakeword`:
```bash
cd ../wyoming-wakeword
python wyoming-wakeword.py \
    --model ../wakeword_trainer/output/hey_jarvis_v1.onnx \
    --device NPU \
    --cutoff 0.55 \
    --refractory 2.0
```
