# Wyoming Wake Word (호출어 감지) 빠른 시작 튜토리얼

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

이 튜토리얼은 Intel OpenVINO로 가속화된 **Wyoming Wake Word (openWakeWord)** 서비스를 Docker 컨테이너 환경에서 복사-붙여넣기로 즉시 실행할 수 있는 가이드입니다. Intel NPU, GPU, CPU 하드웨어 가속 및 다중 모델 동시 감지를 지원합니다.

## 1. 모델 및 디렉토리 준비

작업 디렉토리에 모델 저장용 `models` 폴더를 생성합니다:

```bash
mkdir -p models
```

- `wakeword_trainer`로 직접 학습하여 내보낸 사용자 정의 모델(`.onnx` 파일)을 `./models` 디렉토리에 배치합니다.
- 공식 사전 학습 모델(`hey_jarvis_v0.1.onnx`, `alexa_v0.1.onnx` 등)을 사용하는 경우, 서비스 실행 시 기본 음향 특징 추출 모델(`melspectrogram.onnx`, `embedding_model.onnx`)이 자동으로 다운로드됩니다.

---

## 2. 설정 파일 다운로드 및 편집 (`wakeword_config.yaml`)

한국어 설명 주석이 포함된 설정 파일을 다운로드합니다:

```bash
wget https://raw.githubusercontent.com/shenwii/wyoming-openvino/refs/heads/main/wyoming-wakeword/wakeword_config_ko.yaml -O wakeword_config.yaml
```

편집기로 `wakeword_config.yaml`을 열고 `models` 목록에 호출어 모델을 추가합니다:

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

## 3. `docker-compose.yml` 작성

동일한 디렉토리에 `docker-compose.yml` 파일을 작성합니다:

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
      - LANG=ko_KR.UTF-8
    command:
      - --config
      - wakeword_config.yaml
    restart: unless-stopped
```

### 핵심 설정 안내 및 주의사항
- **작업 디렉토리 (`WORKDIR`)**: 컨테이너의 기본 작업 경로는 `/app`입니다. `./models:/app/models`와 `./wakeword_config.yaml:/app/wakeword_config.yaml`로 마운트하면 상대 경로로 간편하게 지정할 수 있습니다.
- **디바이스 마운트 (`devices`)**:
  - `/dev/accel/accel0:/dev/accel/accel0`: Intel Core Ultra NPU를 활용한 초저전력 상시 감지용입니다. 시스템에 NPU가 없다면 해당 라인을 삭제하거나 주석 처리하세요.
  - `/dev/dri/renderD128:/dev/dri/renderD128`: Intel 내장 GPU 또는 Arc 외장 GPU 가속용입니다.
  - 서비스 내부에서 `NPU` -> `GPU` -> `CPU` 순으로 자동 하드웨어 폴백이 지원됩니다.

---

## 4. 설정 파라미터 핵심 상세 안내

| 파라미터명 | 기본값 | 역할 및 튜닝 가이드 |
| :--- | :--- | :--- |
| `cutoff` | `0.70` | **신뢰도 점수 임계값** (0.0 ~ 1.0). 보통 0.60 ~ 0.80 사이로 설정합니다. 너무 낮으면 오인식이 증가하고, 너무 높으면 발음을 아주 크게 해야 합니다. |
| `trigger_level` | `2` | **연속 감지 프레임 수**. TV 소리 등 돌발 잡음으로 인해 순간 0.9 이상의 피크가 발생할 때 오작동을 방지하기 위해 연속 N개 프레임(1프레임=80ms) 동안 기준을 만족해야 트리거됩니다. |
| `sliding_window_size` | `3` | **이동 평균 윈도우 크기**. 최근 N개 프레임의 점수를 이동 평균(3프레임 ≈ 240ms)하여 일시적인 노이즈를 완화합니다. |
| `trailing_fall` | `true` | **피크 하강 릴리즈 메커니즘 (조기 끊김 방지 핵심 기능)**. 호출어가 잘려 STT로 잘못 입력되는 문제를 방지합니다 (하단 상세 설명 참고). |
| `max_trailing_frames` | `8` | **피크 하강 대기 최대 프레임 수** (~640ms). 긴 발음이나 지속적인 잡음으로 인해 점수가 내려가지 않을 때 멈춤을 방지하는 타임아웃 보호 장치입니다. |
| `trailing_raw_cutoff` | `0.20` | **급락 즉시 릴리즈 기준**. 이동 평균의 지연 시간을 건너뛰고, 발음이 끝나 단일 프레임 원시 점수가 급락할 경우 즉시 이벤트를 발생시킵니다. |
| `refractory` | `2.0` | **쿨다운 방지 시간(초)**. 감지 후 지정된 시간 동안 중복 반응을 방지합니다. |

### 💡 `trailing_fall` (피크 하강 릴리즈) 기능이 꼭 필요한 이유
호출어 발음 길이가 긴 편일 때(약 1초), 이동 평균 창 크기에 따라 앞부분(예: 500ms 부근)에서 이미 점수가 `cutoff`를 넘는 경우가 발생합니다.
- **이 기능이 꺼져 있는 경우**: 중간에 즉시 트리거되어 Home Assistant로 넘어가고, 호출어 뒷부분 음성이 사용자 명령어 앞머리로 인식되어 STT에 오염된 데이터가 들어갑니다.
- **`trailing_fall: true` 활성화 시**: 점수가 기준에 도달해도 즉시 끊지 않고 점수 추이(`0.1 -> 0.7 -> 0.9 -> 0.9 -> 0.7 -> 0.3`)를 관찰합니다. 발음이 완료되어 점수가 하강 곡선을 그릴 때 비로소 릴리즈하므로, 자연스러운 음성 정지 시점에 정확히 잘려 깨끗한 음성이 STT로 전달됩니다.

---

## 5. 서비스 실행 및 Home Assistant 연동

컨테이너 백그라운드 시작:
```bash
docker compose up -d
```

로그 확인:
```bash
docker compose logs -f wyoming-wakeword
```

### Home Assistant 연동
1. Home Assistant 관리자 화면에서 **설정** -> **기기 및 서비스** -> **통합 구성요소 추가**로 이동합니다.
2. **Wyoming Protocol**을 검색하고 호스트 IP와 포트 `10400`을 입력합니다.
3. **음성 어시스턴트** 설정에서 등록된 호출어 모델을 선택합니다.
