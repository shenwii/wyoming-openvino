# Wyoming STT (음성 인식) 빠른 시작 튜토리얼

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

이 튜토리얼은 Intel OpenVINO로 가속화된 **Wyoming STT (Whisper)** 서비스를 Docker 컨테이너 환경에서 복사-붙여넣기로 즉시 실행할 수 있는 가이드입니다.

## 1. 모델 다운로드

Whisper 모델은 OpenVINO IR(Intermediate Representation) 형식이어야 합니다. 본 예제에서는 인식 정확도와 하드웨어 리소스 효율의 균형이 뛰어난 **Whisper Medium FP16** 모델을 사용합니다.

> 💡 **다른 크기 및 양자화 모델**: 더 가볍고 빠른 속도를 원하시면(`base`, `small`) 또는 더 높은 정확도를 원하시면(`large-v3-turbo` 등), Hugging Face의 [OpenVINO 공식 조직 페이지](https://huggingface.co/OpenVINO)에서 모델 ID를 확인하여 변경하실 수 있습니다.

작업 디렉토리에서 다음 명령을 실행하여 `models` 폴더를 생성하고 모델을 다운로드합니다:

```bash
mkdir -p models
docker run --rm -v ./models:/models docker.io/library/python:3.11 bash -c '
pip install -U huggingface_hub
hf download OpenVINO/whisper-medium-fp16-ov --local-dir /models/whisper-medium-fp16-ov
'
```

---

## 2. `docker-compose.yml` 작성

동일한 디렉토리에 `docker-compose.yml` 파일을 작성합니다:

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
      - LANG=ko_KR.UTF-8
    command:
      - --model-dir
      - /models/whisper-medium-fp16-ov
      - --device
      - GPU
      - --port
      - "10020"
    restart: unless-stopped
```

### 핵심 설정 안내 및 주의사항
- **포트**: STT의 표준 기본 포트는 `10020`입니다 (TTS 기본 포트는 `10021`).
- **하드웨어 가속 (`devices`)**:
  - `/dev/dri/renderD128:/dev/dri/renderD128`: Intel 내장 그래픽(iGPU) 또는 Intel Arc 외장 그래픽을 컨테이너에 전달하여 GPU 가속을 적용합니다.
  - 순수 CPU 환경에서 실행할 경우 `devices` 섹션을 삭제하고 command의 `--device` 값을 `CPU`로 변경하세요.
- **언어 설정**:
  - 기본값은 자동 다국어 감지(`--language auto`)입니다. 한국어로 고정하려면 `--language ko`를 추가할 수 있습니다.
- **환경 변수 (`LANG=ko_KR.UTF-8`)**: 콘솔 로그 및 Home Assistant 연동 메타데이터를 한국어로 출력합니다.

---

## 3. 서비스 실행

Docker Compose를 사용하여 백그라운드에서 서비스를 시작합니다:

```bash
docker compose up -d
```

실행 로그를 확인하여 모델이 성공적으로 로드되었는지 확인합니다:

```bash
docker compose logs -f wyoming-stt
```

---

## 4. Home Assistant 연동

1. Home Assistant 관리자 화면에서 **설정** -> **기기 및 서비스** -> **통합 구성요소 추가**로 이동합니다.
2. **Wyoming Protocol**을 검색하고 선택합니다.
3. STT 서비스가 실행 중인 호스트 IP와 포트 `10020`을 입력하고 제출합니다.
4. **설정** -> **음성 어시스턴트 (Voice Assistants)** 파이프라인 설정으로 이동하여, **음성을 텍스트로 변환 (Speech-to-text)** 항목에서 `wyoming-openvino-stt`를 선택합니다.
