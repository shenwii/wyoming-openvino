# Wyoming STT (语音识别) 极速抄作业教程

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

本教程提供基于 Intel OpenVINO 加速的 **Wyoming STT (Whisper)** 容器化无脑部署流程，复制命令即可直接运行。

## 1. 下载模型

Whisper 模型需使用 OpenVINO IR 格式。本示例选用 **Whisper Medium FP16** 精度模型（在精度与显存/算力消耗间取得最佳平衡）。

> 💡 **其他尺寸与量化**: 如需更小体积更快的速度（如 `base`、`small`）或极高精度的 `large-v3-turbo`，可直接前往 Hugging Face 的 [OpenVINO 官方组织主页](https://huggingface.co/OpenVINO) 检索替换模型 ID。

在当前工作目录下执行以下命令创建 `models` 目录并在前台自动下载：

```bash
mkdir -p models
docker run --rm -v ./models:/models docker.io/library/python:3.11 bash -c '
pip install -U huggingface_hub
hf download OpenVINO/whisper-medium-fp16-ov --local-dir /models/whisper-medium-fp16-ov
'
```

---

## 2. 编写 `docker-compose.yml`

在同一目录下创建 `docker-compose.yml` 文件：

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
      - LANG=zh_CN.UTF-8
    command:
      - --model-dir
      - /models/whisper-medium-fp16-ov
      - --device
      - GPU
      - --port
      - "10020"
      - --to-simplified
    restart: unless-stopped
```

### 关键细节说明与踩坑提醒
- **端口**: STT 默认标准端口为 `10020`（TTS 默认端口为 `10021`）。
- **硬件加速 (`devices`)**:
  - `/dev/dri/renderD128:/dev/dri/renderD128`：用于挂载 Intel 核显 (iGPU) 或 Arc 独立显卡进行 GPU 加速。
  - 如果仅在纯 CPU 环境运行，可删除 `devices` 映射并将 command 中的 `--device` 改为 `CPU`。
- **中文繁简转换 (`--to-simplified`)**: Whisper 默认中文经常输出繁体字，添加 `--to-simplified` 参数可强制转换为标准简体中文输出。
- **语言环境 (`LANG=zh_CN.UTF-8`)**: 控制控制台日志与 Home Assistant 元数据以中文显示。

---

## 3. 启动服务

使用 Docker Compose 一键在后台启动服务：

```bash
docker compose up -d
```

查看运行日志，确认模型是否加载成功：

```bash
docker compose logs -f wyoming-stt
```

---

## 4. 接入 Home Assistant

1. 打开 Home Assistant，依次进入 **配置 (Settings)** -> **设备与服务 (Devices & Services)** -> **添加集成 (Add Integration)**。
2. 搜索并选择 **Wyoming Protocol**。
3. 填入运行 STT 服务的机器 IP 地址，端口填写 `10020`，点击提交。
4. 前往 **语音助手 (Voice Assistants)** -> 选择流水线，在 **语音转文本 (Speech-to-text)** 下拉列表中选择 `wyoming-openvino-stt`。
