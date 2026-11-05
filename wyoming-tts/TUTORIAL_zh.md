# Wyoming TTS (语音合成) 极速抄作业教程

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

本教程提供基于 Intel OpenVINO 加速的 **Wyoming TTS (Kokoro-82M)** 容器化无脑部署流程，复制命令即可直接运行。

## 1. 下载模型

示例选用 **Kokoro-82M INT8 量化模型**（兼顾极低内存占用与高保真合成音质）。

> 💡 **模型选项**: 如需未量化的原生 FP16 模型，也可使用 `Echo9Zulu/Kokoro-82M-FP16-OpenVINO`。

在当前工作目录下执行以下命令创建 `models` 目录并下载模型文件：

```bash
mkdir -p models
docker run --rm -v ./models:/models docker.io/library/python:3.11 bash -c '
pip install -U huggingface_hub
hf download OpenVINO/Kokoro-82M-int8-ov --local-dir /models/Kokoro-82M-int8-ov
'
```

---

## 2. 编写 `docker-compose.yml`

在同一目录下创建 `docker-compose.yml` 文件：

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
      - LANG=zh_CN.UTF-8
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

### 关键细节说明与踩坑提醒
- **端口**: TTS 默认标准端口为 `10021`（STT 默认端口为 `10020`）。
- **模型路径匹配**: 下载保存的目录路径为 `/models/Kokoro-82M-int8-ov`，command 中的 `--model-dir` 必须与其严格保持一致。
- **推荐运行设备 (`--device CPU`)**:
  - Kokoro TTS 优先强烈推荐运行在 **CPU** 上，具备 OpenVINO 针对 CPU 指令集（AVX2/AVX-512/AMX）的深度自适应调度，首句发声延迟仅约 100~300ms，且避免 GPU 算子编译开销。
  - 若确需运行在 GPU 上，可挂载 `devices: - /dev/dri/renderD128:/dev/dri/renderD128` 并将 `--device` 设为 `GPU`。
- **语速调节 (`--speed 0.85`)**: Kokoro 默认语速偏快，设为 `0.85` 听感最为自然舒适。
- **音色指定与加权融合 (`--voice`)**:
  - 中文常用音色：`zf_xiaobei`（女声）、`zm_yunjian`（男声）、`zf_xiaoni`（女声）、`zm_yunxi`（男声）。
  - 支持多发音人音色加权融合，例如 command 中追加 `- --voice` 和 `- zf_xiaobei(0.7)+af_heart(0.3)`。

---

## 3. 启动服务

使用 Docker Compose 一键在后台启动服务：

```bash
docker compose up -d
```

查看运行日志，确认模型是否加载成功：

```bash
docker compose logs -f wyoming-tts
```

---

## 4. 接入 Home Assistant

1. 打开 Home Assistant，依次进入 **配置 (Settings)** -> **设备与服务 (Devices & Services)** -> **添加集成 (Add Integration)**。
2. 搜索并选择 **Wyoming Protocol**。
3. 填入运行 TTS 服务的机器 IP 地址，端口填写 `10021`，点击提交。
4. 前往 **语音助手 (Voice Assistants)** -> 选择流水线，在 **文本转语音 (Text-to-speech)** 下拉列表中选择 `kokoro-openvino`。
