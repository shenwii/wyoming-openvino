# wyoming-openvino

[ English ](README.md) | [ 中文 ](README_zh.md) | [ 日本語 ](README_ja.md) | [ 한국어 ](README_ko.md)

---

Wyoming 프로토콜 + Intel OpenVINO: Intel NPU (Core Ultra), Arc 외장 그래픽, 내장 그래픽 (iGPU) 및 CPU에 최적화된 고성능 호출어 감지 (Wake Word), 음성 인식 (STT), 음성 합성 (TTS) 마이크로서비스로 Home Assistant와 원활하게 연동됩니다.

## 🌟 주요 특징

- **하드웨어 완전 가속 및 네이티브 NPU 최적화 (Intel Core Ultra)**:
  - **호출어 (openWakeWord)**: Intel NPU (Meteor Lake / Lunar Lake / Arrow Lake)를 위해 정적 형태 리셰이프를 자동 처리하며, 초저전력 상시 대기 및 `NPU` -> `GPU` -> `CPU` 자동 하드웨어 대체를 지원합니다.
  - **음성 인식 (STT - Whisper)**: `GPU` (Arc 외장/내장 그래픽), `CPU`, `AUTO` 실행을 지원합니다.
  - **음성 합성 (TTS - Kokoro)**: `CPU` 실행을 권장하며 적응형 스레드 스케줄링을 지원합니다.
- **다중 모델 동시 감지 및 YAML/JSON 설정**:
  - 단일 오디오 스트림에서 여러 호출어 (`Hey Jarvis`, `Alexa`, 사용자 정의 모델 등)를 동시에 청취하며 특성 추출 파이프라인을 공유합니다.
  - YAML 또는 JSON 설정 파일로 모델별 `cutoff` (임계값), `refractory` (대기 시간), `device` (실행 장치)를 개별 지정할 수 있습니다.
- **스마트 스트리밍 절 분할 및 즉시 생성 (Kokoro TTS)**:
  - 약어 및 소수점 보호 (`Dr.`, `Mr.`, `3.14` 등의 비정상적인 분단 방지).
  - 문장 부호 (`。！？!?`)를 보존하여 자연스러운 억양과 운율을 표현합니다.
  - 절 단위 스트리밍 출력: 첫 번째 절을 100~300ms 이내에 즉시 발화 시작하여 Wyoming TCP로 스트리밍합니다.
- **가중 음색 블렌딩 (Voice Blending)**:
  - Kokoro-FastAPI 스타일 구문 지원 (예: `zf_xiaobei(0.7)+af_heart(0.3)`, `af_bella(2)+af_sky(1)` 등).
- **국제화 (i18n) 완벽 지원**:
  - 영어 (`en`), 중국어 (`zh`), 일본어 (`ja`), 한국어 (`ko`) 4개 언어 지원.
  - 콘솔 출력 및 Home Assistant에 전달되는 서비스 정보가 환경 변수 (`LANG`, `LC_ALL`, `WYOMING_LANG`)에 따라 동적으로 변경됩니다.
- **Docker 컨테이너 지원**:
  - `wyoming-stt`, `wyoming-tts`, `wyoming-wakeword` 하위 디렉터리로 완전히 분리되어 각각 독립된 `Dockerfile`, `requirements.txt`, 셸 스크립트를 제공하며, 기본 이미지는 `docker.io/openvino/ubuntu24_runtime:latest`를 사용합니다.

---

## 📁 디렉터리 구조

```
wyoming-openvino/
├── wyoming-stt/              # 음성 인식 서비스 (Whisper OpenVINO)
│   ├── wyoming-stt.py        # Python 서비스 스크립트
│   ├── wyoming-stt.sh        # 셸 진입점 스크립트
│   ├── i18n.py               # 국제화 엔진
│   ├── lang/                 # 언어 카탈로그 (en.json, zh.json, ja.json, ko.json)
│   ├── requirements.txt      # STT 의존성 목록
│   └── Dockerfile            # 컨테이너 정의
│
├── wyoming-tts/              # 음성 합성 서비스 (Kokoro OpenVINO)
│   ├── wyoming-tts.py        # Python 서비스 스크립트
│   ├── wyoming-tts.sh        # 셸 진입점 스크립트
│   ├── i18n.py               # 국제화 엔진
│   ├── lang/                 # 언어 카탈로그 (en.json, zh.json, ja.json, ko.json)
│   ├── requirements.txt      # TTS 의존성 목록
│   └── Dockerfile            # 다단계 빌드 컨테이너 정의
│
├── wyoming-wakeword/         # 호출어 서비스 (openWakeWord OpenVINO)
│   ├── wyoming-wakeword.py   # Python 서비스 스크립트
│   ├── wyoming-wakeword.sh   # 셸 진입점 스크립트
│   ├── wakeword_config.yaml  # 다중 모델 설정 파일
│   ├── i18n.py               # 국제화 엔진
│   ├── lang/                 # 언어 카탈로그 (en.json, zh.json, ja.json, ko.json)
│   ├── requirements.txt      # 호출어 의존성 목록
│   └── Dockerfile            # 컨테이너 정의
│
├── wakeword_trainer/         # 호출어 훈련 도구 모음
├── README.md                 # 영어 문서 (기본)
├── README_zh.md              # 중국어 문서
├── README_ja.md              # 일본어 문서
├── README_ko.md              # 한국어 문서
└── LICENSE                   # GPL-3.0 라이선스
```

---

## 📦 설치 및 직접 실행

### 1. Wyoming STT (음성 인식)

```bash
cd wyoming-stt
pip install -r requirements.txt

# GPU 가속 실행
./wyoming-stt.sh --model-dir /path/to/whisper-openvino --device GPU --port 10020
```

### 2. Wyoming TTS (음성 합성)

```bash
cd wyoming-tts
pip install -r requirements.txt

# CPU 실행 (자연스러운 속도로 0.85 권장)
./wyoming-tts.sh --model-dir /path/to/Kokoro-82M-FP16-OpenVINO --device CPU --speed 0.85 --port 10021
```

### 3. Wyoming Wake Word (호출어)

```bash
cd wyoming-wakeword
pip install -r requirements.txt

# 방법 A: YAML 구성 파일로 시작 (권장)
./wyoming-wakeword.sh --config wakeword_config.yaml

# 방법 B: Intel NPU에서 단일 모델 직접 실행
./wyoming-wakeword.sh --model models/hey_jarvis_v0.1.onnx --device NPU --cutoff 0.6 --port 10400
```

---

## 🐳 Docker 컨테이너 배포

모든 서비스는 `docker.io/openvino/ubuntu24_runtime:latest`를 기본 컨테이너로 사용합니다.

### 1. Docker 이미지 빌드

```bash
# STT 이미지 빌드
docker build -t wyoming-stt ./wyoming-stt

# TTS 이미지 빌드 (다단계 빌드로 pyopenjtalk 자동 컴파일)
docker build -t wyoming-tts ./wyoming-tts

# Wake Word 이미지 빌드
docker build -t wyoming-wakeword ./wyoming-wakeword
```

### 2. 컨테이너 실행

#### STT 컨테이너 (GPU `/dev/dri` 매핑)
```bash
docker run -d \
  --name wyoming-stt \
  --restart unless-stopped \
  --net host \
  --device /dev/dri \
  -v /path/to/whisper-openvino:/models/whisper:ro \
  -e LANG=ko_KR.UTF-8 \
  wyoming-stt \
  --model-dir /models/whisper \
  --device GPU \
  --port 10020
```

#### TTS 컨테이너
```bash
docker run -d \
  --name wyoming-tts \
  --restart unless-stopped \
  --net host \
  -v /path/to/Kokoro-82M-FP16-OpenVINO:/models/kokoro:ro \
  -e LANG=ko_KR.UTF-8 \
  wyoming-tts \
  --model-dir /models/kokoro \
  --device CPU \
  --speed 0.85 \
  --port 10021
```

#### Wake Word 컨테이너 (Intel NPU `/dev/accel` 매핑)
```bash
docker run -d \
  --name wyoming-wakeword \
  --restart unless-stopped \
  --net host \
  --device /dev/accel \
  -v /path/to/models:/app/models \
  -v $(pwd)/wyoming-wakeword/wakeword_config.yaml:/app/wakeword_config.yaml:ro \
  -e LANG=ko_KR.UTF-8 \
  wyoming-wakeword \
  --config /app/wakeword_config.yaml
```

---

## 🌐 국제화 (i18n) 설정

각 마이크로서비스는 다국어 지원 모듈을 갖추고 있으며 번역 텍스트는 `lang/` 디렉터리에 분리되어 있습니다 (**영어**: `en.json`, **중국어**: `zh.json`, **일본어**: `ja.json`, **한국어**: `ko.json`). 새로운 언어(예: 독일어 `de.json`)를 추가할 때도 Python 코드를 수정할 필요 없이 해당 디렉터리에 JSON 파일만 추가하면 됩니다.

로그 및 Wyoming 서비스 메타데이터 언어는 표준 환경 변수를 통해 자동 감지됩니다:
- `WYOMING_LANG` (최우선 순위)
- `LC_ALL`
- `LC_MESSAGES`
- `LANG` (예: `ko_KR.UTF-8`, `en_US.UTF-8`, `zh_CN.UTF-8`, `ja_JP.UTF-8`)

---

## 🏠 Home Assistant 연동

Home Assistant에서 **설정** -> **기기 및 서비스** -> **통합 구성요소 추가** -> **Wyoming Protocol** 검색:

1. **STT 추가**: 호스트 IP, 포트 `10020`
2. **TTS 추가**: 호스트 IP, 포트 `10021`
3. **Wake Word 추가**: 호스트 IP, 포트 `10400`

추가 후 **음성 어시스턴트 (Assist)**에서 해당 모델을 선택하여 사용하십시오.

---

## 📜 라이선스

GNU General Public License v3.0 (GPL v3)
