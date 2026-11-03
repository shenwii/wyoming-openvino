# Wyoming / openWakeWord NPU 호출어 학습 스위트

**[openWakeWord](https://github.com/dscripka/openWakeWord)** 및 **[MicroWakeWordV2Trainer](https://github.com/JohnnyPrimus/MicroWakeWordV2Trainer)** 기반으로 제작된, **Intel NPU (OpenVINO VPUX / Core Ultra)** 및 하드웨어 가속에 특화된 엔드투엔드 호출어 학습 스위트입니다.

---

## 🌟 주요 특징

1. **통합 사용자 설정 (`config.yaml`)**:
   - `wakeword` 블록에 모든 핵심 설정 집약: 대상 문구, `friend_name` (공백 없는 모델 식별자), 변형 문구, 오발화 억제 부정어(Hard Negatives).
   - 다국어 지원 (`en`, `zh`, `ja` 등).
2. **듀얼 엔진 TTS 합성 (`generate_tts.py`)**:
   - **로컬 엔진 (Kokoro OpenVINO TTS)**: 초고속 오프라인 합성, 음성 블렌딩 지원.
   - **클라우드 엔진 (Microsoft Edge-TTS)**: 고품질 클라우드 음성.
3. **현실적인 음향 시뮬레이션 (`augment_data.py`)**:
   - MIT Environmental RIR 실내 잔향 합성.
   - 생활 환경 소음 랜덤 SNR(5~25dB) 믹싱.
4. **Intel NPU 전용 최적화 (`train.py`)**:
   - **완전 정적 형상**: `[1, 16, 96] -> [1, 1]` 정적 ONNX 모델 내보내기.
   - **순수 연산자 그래프**: `Gemm + ReLU + Sigmoid`로 100% NPU 하드웨어 가속 달성.
   - **하드웨어 가속 지원**: CPU, CUDA 및 Intel GPU (PyTorch 2.4+ `xpu` / IPEX) 학습 지원.

---

## 📁 디렉토리 구조

```text
wakeword_trainer/
├── config.sample.yaml     # 설정 샘플 (영어 기본)
├── config.sample_zh.yaml  # 설정 샘플 (중국어)
├── config.sample_ja.yaml  # 설정 샘플 (일본어)
├── config.sample_ko.yaml  # 설정 샘플 (한국어)
├── requirements.txt       # 의존 패키지 목록
├── i18n.py                # 다국어 엔진
├── lang/                  # 번역 카탈로그 (en, zh, ja, ko)
├── setup_libs.py          # 특징 추출 기본 모델 다운로드
├── download_data.py       # RIR 및 소음 데이터셋 다운로드
├── generate_tts.py        # TTS 합성 데이터 생성
├── augment_data.py        # 음향 시뮬레이션 및 데이터 증강
├── train.py               # DNN 학습 및 NPU 최적화 ONNX 내보내기
├── run_pipeline.py        # 자동 파이프라인 러너
└── README.md              # 설명 문서
```

---

## 🛠️ 설치

```bash
cd wakeword_trainer
pip install -r requirements.txt
```

---

## ⚙️ 설정 (`config.yaml`)

```bash
cp config.sample_ko.yaml config.yaml
```

`wakeword` 섹션만 편집하면 됩니다:
```yaml
wakeword:
  phrase: "hey jarvis"            # 대상 호출어
  friend_name: "hey_jarvis_v1"    # 모델 식별자 ({friend_name}.onnx로 저장)
  candidate_phrases:              # 변형 문구
    - "jarvis jarvis"
  hard_negatives:                 # 오발화 방지 부정어
    - "hey harvest"
    - "불 켜줘"
  language: "en"                  # 언어 코드
```

---

## 🚀 학습 실행

전체 파이프라인 자동 실행:
```bash
python run_pipeline.py --config config.yaml
```

빠른 테스트 모드:
```bash
python run_pipeline.py --quick-mode
```

---

## 📦 TFLite 모델 내보내기 (microWakeWord / ESPHome)

본 학습 도구는 **TensorFlow Lite (`.tflite`)** 포맷 내보내기를 지원하여 **Home Assistant openWakeWord**, **ESPHome**, **microWakeWord** 임베디드 기기와 완벽히 호환됩니다.

```bash
# 전체 포맷(ONNX + TFLite Float32) 내보내기 (재학습 불필요)
python train.py --config config.yaml --export-only

# TFLite 전용 내보내기 (Float32, 약 390 KB)
python train.py --config config.yaml --export-tflite

# INT8 양자화 TFLite 내보내기 (약 100 KB, ESP32-S3 / 마이크로컨트롤러 최적화)
python train.py --config config.yaml --export-tflite --quantize
```

---

## 🎯 모델 배포 안내

### 1. Wyoming Wake Word 서비스 (Intel NPU / GPU / CPU)

OpenVINO 런타임을 통해 **`.onnx` 및 `.tflite` 모델을 모두 직접 실행**할 수 있습니다:

```bash
cd ../wyoming-wakeword
# ONNX 모델 실행 (Intel NPU 권장)
python wyoming-wakeword.py \
    --model ../wakeword_trainer/output/hey_jarvis_v1.onnx \
    --device NPU \
    --cutoff 0.55 \
    --refractory 2.0

# TFLite 모델 실행 (OpenVINO 네이티브 가속)
python wyoming-wakeword.py \
    --model ../wakeword_trainer/output/hey_jarvis_v1.tflite \
    --device NPU \
    --cutoff 0.55
```

### 2. ESPHome / microWakeWord (ESP32-S3)

생성된 양자화 모델 `{friend_name}_int8.tflite`를 ESPHome에 설정:
```yaml
micro_wake_word:
  models:
    - model: "hey_jarvis_v1_int8.tflite"
      probability_cutoff: 0.55
      sliding_window_average_size: 2
```
