# 호출어 모델 학습 파이프라인 (wakeword_trainer) 빠른 시작 튜토리얼

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

이 튜토리얼은 Docker 환경에서 **사용자 정의 호출어(Wake Word) 모델을 엔드투엔드로 완전 자동 학습**할 수 있는 가이드입니다.
Kokoro OpenVINO와 Microsoft Edge-TTS 듀얼 엔진을 통한 음성 자동 합성, 실제 방의 반향 시뮬레이션(MIT RIR), 일상 배경 노이즈 믹싱을 거쳐 **Intel NPU에 최적화된 ONNX 모델** 및 **ESPHome / microWakeWord 호환 TFLite 모델**을 원클릭으로 생성합니다.

---

## 1. 코드 저장소 클론

```bash
git clone --depth=1 'https://github.com/shenwii/wyoming-openvino'
cd wyoming-openvino
```

---

## 2. 설정 파일 복사 및 편집

한국어용 예제 설정 파일을 복사합니다:

```bash
cp wakeword_trainer/config.sample_ko.yaml ./config.yaml
```

편집기로 `./config.yaml`을 열고 상단의 `wakeword` 항목을 원하는 호출어로 수정합니다:

```yaml
wakeword:
  phrase: "안녕 지니"           # 목표 호출어
  friend_name: "annyeongjini_v1" # 모델 식별자 ({friend_name}.onnx 및 {friend_name}.tflite로 출력)
  candidate_phrases:            # 발음 변형 및 연음 보정용 후보 단어
    - "지니 지니"
    - "하이 지니"
  hard_negatives:               # 오인식 방지용 유사 발음 및 일상 명령어 (필수 방어선)
    - "안녕 미니"
    - "안녕 토니"
    - "불 켜줘"
    - "불 꺼줘"
    - "오늘 날씨 어때"
  language: "ko"                # 언어 코드 (ko, en, zh, ja 등)
```

> 💡 **권장 학습 장치**: `training.device`는 `"cpu"`로 유지하는 것을 권장합니다 (학습 속도가 빠르고 드라이버 충돌이 없습니다).

---

## 3. Docker 원클릭 자동 학습 실행

다음 단일 명령을 실행하면 Docker 컨테이너 내에서 필수 의존성을 설치하고, Kokoro 로컬 오프라인 음성 합성 모델을 다운로드한 후 학습 파이프라인 전체를 자동으로 실행합니다:

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

> 💡 **안내**:
> - 시스템에 Intel 그래픽 카드가 없는 경우 `--device /dev/dri/renderD128:/dev/dri/renderD128` 라인을 제거하세요.
> - 컨테이너는 백그라운드에서 안전하게 실행됩니다.

---

## 4. 실시간 학습 진행 상황 확인

다음 명령어로 실시간 로그를 확인합니다:

```bash
docker logs -f wakeword-trainer
```

자동화 파이프라인은 다음 단계를 순서대로 수행합니다:
1. **기초 특징 추출 모델 및 데이터셋 다운로드** (openWakeWord 기초 모델, MIT 룸 임펄스 응답, 환경 잡음 등).
2. **듀얼 엔진 음성 대량 합성** (Kokoro TTS 및 Edge-TTS를 사용해 수천 건의 긍정/부정 음성 자동 생성).
3. **룸 어쿠스틱 시뮬레이션 및 데이터 증강** (공간 반향 및 배경 잡음 합성).
4. **신경망 학습 및 멀티 포맷 모델 출력**.

---

## 5. 생성된 모델 배포 및 활용

학습이 완료되면 생성된 결과물은 `./wakeword_trainer/output/` 디렉토리에 저장됩니다:

- **`{friend_name}.onnx`**:
  Intel NPU / GPU / CPU 최적화 정적 형상 ONNX 모델.
  👉 **활용**: `wyoming-wakeword/models/`로 복사하고 `wakeword_config.yaml`에 등록하여 Home Assistant 상시 감지에 사용합니다.
- **`{friend_name}.tflite`**:
  표준 32비트 부동소수점 TensorFlow Lite 모델.
  👉 **활용**: ESP32-S3 기반 **ESPHome microWakeWord** 스마트 마이크 기기에 배포합니다.
- **`{friend_name}.json`**:
  권장 `cutoff` 감지 임계값 등이 포함된 설정 메타데이터.
- **`{friend_name}.pt`**:
  PyTorch 원본 가중치 파일 (재합성 없이 모델만 즉시 재변환할 때 사용).
