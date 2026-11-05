# Wyoming TTS (음성 합성) 빠른 시작 튜토리얼

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

이 튜토리얼은 Intel OpenVINO로 가속화된 **Wyoming TTS (Kokoro-82M)** 서비스를 Docker 컨테이너 환경에서 복사-붙여넣기로 즉시 실행할 수 있는 가이드입니다.

## 1. 모델 다운로드

본 예제에서는 메모리 점유율이 매우 낮으면서도 고품질 음성을 생성하는 **Kokoro-82M INT8 양자화 모델**을 사용합니다.

> 💡 **모델 선택 옵션**: 비양자화 원본 FP16 모델을 원하시면 `Echo9Zulu/Kokoro-82M-FP16-OpenVINO`로 변경하실 수 있습니다.

작업 디렉토리에서 다음 명령을 실행하여 `models` 폴더를 생성하고 모델을 다운로드합니다:

```bash
mkdir -p models
docker run --rm -v ./models:/models docker.io/library/python:3.11 bash -c '
pip install -U huggingface_hub
hf download OpenVINO/Kokoro-82M-int8-ov --local-dir /models/Kokoro-82M-int8-ov
'
```

---

## 2. `docker-compose.yml` 작성

동일한 디렉토리에 `docker-compose.yml` 파일을 작성합니다:

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
      - LANG=ko_KR.UTF-8
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

### 핵심 설정 안내 및 주의사항
- **포트**: TTS의 표준 기본 포트는 `10021`입니다 (STT 기본 포트는 `10020`).
- **모델 경로 일치**: 다운로드된 디렉토리 경로인 `/models/Kokoro-82M-int8-ov`와 command의 `--model-dir` 옵션 경로가 반드시 정확히 일치해야 합니다.
- **권장 실행 디바이스 (`--device CPU`)**:
  - Kokoro TTS는 **CPU**에서 실행하는 것을 적극 권장합니다. OpenVINO의 CPU 명령어 최적화(AVX2/AVX-512/AMX) 덕분에 GPU 셰이더 컴파일 지연 없이 100~300ms 이내에 첫 음성 조각이 즉각 출력됩니다.
  - GPU 환경에서 구동하려면 `devices: - /dev/dri/renderD128:/dev/dri/renderD128` 항목을 추가하고 `--device`를 `GPU`로 설정하세요.
- **말하기 속도 (`--speed 0.85`)**: Kokoro의 기본 재생 속도는 다소 빠른 편이므로, `0.85`로 지정하면 가장 자연스럽고 편안한 어조를 얻을 수 있습니다.
- **화자(음색) 지정 및 가중치 블렌딩 (`--voice`)**:
  - 다국어 화자 지원 및 복수 화자 합성(Blending) 지원 (예: `- --voice` 및 `- zf_xiaobei(0.7)+af_heart(0.3)`).

---

## 3. 서비스 실행

Docker Compose를 사용하여 백그라운드에서 서비스를 시작합니다:

```bash
docker compose up -d
```

실행 로그를 확인하여 모델이 성공적으로 로드되었는지 확인합니다:

```bash
docker compose logs -f wyoming-tts
```

---

## 4. Home Assistant 연동

1. Home Assistant 관리자 화면에서 **설정** -> **기기 및 서비스** -> **통합 구성요소 추가**로 이동합니다.
2. **Wyoming Protocol**을 검색하고 선택합니다.
3. TTS 서비스가 실행 중인 호스트 IP와 포트 `10021`을 입력하고 제출합니다.
4. **설정** -> **음성 어시스턴트 (Voice Assistants)** 파이프라인 설정으로 이동하여, **텍스트를 음성으로 변환 (Text-to-speech)** 항목에서 `kokoro-openvino`를 선택합니다.
