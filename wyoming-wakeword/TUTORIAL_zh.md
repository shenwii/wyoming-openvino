# Wyoming Wake Word (唤醒词) 极速抄作业教程

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

本教程提供基于 Intel OpenVINO 加速的 **Wyoming Wake Word (openWakeWord)** 容器化无脑部署流程，支持 Intel NPU、GPU 与 CPU 硬件加速及智能多模型并发监听。

## 1. 准备模型与目录

在工作目录下创建模型文件夹 `models`：

```bash
mkdir -p models
```

- 将训练好的自定义唤醒词模型（由 `wakeword_trainer` 生成的 `.onnx` 文件）放入 `./models` 目录中。
- 若使用官方内置模型（如 `hey_jarvis_v0.1.onnx`、`alexa_v0.1.onnx` 等），系统会在启动时自动下载基础声学模型（`melspectrogram.onnx` 和 `embedding_model.onnx`）。

---

## 2. 获取并配置 `wakeword_config.yaml`

下载带中文注释的配置文件：

```bash
wget https://raw.githubusercontent.com/shenwii/wyoming-openvino/refs/heads/main/wyoming-wakeword/wakeword_config_zh.yaml -O wakeword_config.yaml
```

使用编辑器打开 `wakeword_config.yaml`，按需在 `models` 列表添加您的唤醒词模型：

```yaml
models:
  - name: "nihaoxiaozhi"
    phrase: "你好小智"
    model_path: "models/nihaoxiaozhi.onnx"
    cutoff: 0.70
    trigger_level: 2
    sliding_window_size: 3
    trailing_fall: true
    refractory: 2.0
    device: "NPU"
```

---

## 3. 编写 `docker-compose.yml`

在同一目录下创建 `docker-compose.yml` 文件：

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
      - LANG=zh_CN.UTF-8
    command:
      - --config
      - wakeword_config.yaml
    restart: unless-stopped
```

### 关键细节说明与踩坑提醒
- **容器工作目录 (`WORKDIR`)**: 容器默认工作目录为 `/app`，因此挂载到 `/app/models` 和 `/app/wakeword_config.yaml` 后，`command` 与配置文件中均可直接使用相对路径。
- **硬件设备挂载 (`devices`)**:
  - `/dev/accel/accel0:/dev/accel/accel0`: 用于 Intel Core Ultra 原生 NPU 加速（超低功耗常驻监听）。若主板/CPU 无 NPU，请注释或删除此行。
  - `/dev/dri/renderD128:/dev/dri/renderD128`: 用于 Intel 核显或 Arc 独显加速。
  - 服务内置硬件自动回退机制：`NPU` -> `GPU` -> `CPU`，若硬件不可用会自动平滑回退。

---

## 4. 配置文件核心参数深度解析

| 参数名 | 默认值 | 作用说明与调优建议 |
| :--- | :--- | :--- |
| `cutoff` | `0.70` | **唤醒匹配分数阈值** (0.0 ~ 1.0)。通常设置在 0.60 ~ 0.80 之间，过低容易误唤醒，过高则需要发音非常用力。 |
| `trigger_level` | `2` | **连续达标帧数**。某些突发杂音（如电视背景声）偶尔会跳出 0.9 以上高分，设置连续达到 N 帧才算命中（每帧 80ms），大幅降低单帧毛刺误触发。 |
| `sliding_window_size` | `3` | **滑动窗口平均大小**。维护最近 N 帧得分的移动平均值（1帧=80ms，3帧约 240ms）。有效平滑分数毛刺。 |
| `trailing_fall` | `true` | **过峰回落释放机制 (核心防早截断机制)**。详见下方原理解释。 |
| `max_trailing_frames` | `8` | **过峰回落最大等待帧数**（8 帧约 640ms）。当处于等待回落阶段时，防止用户持续拖长音或环境高噪导致分数不降而卡死的保底超时机制。 |
| `trailing_raw_cutoff` | `0.20` | **单帧骤降秒退门槛**。当滑动窗口设置较大时平均分存在滞后。若用户发音结束突然闭嘴，当前单帧原始分一旦跌破此值立即触发释放。 |
| `refractory` | `2.0` | **冷却防重时间 (秒)**。触发后指定时间内忽略该模型，防止同一声唤醒被连续二次触发。 |

### 💡 为什么需要 `trailing_fall` (过峰回落机制)？
如果唤醒词比较长（发音约 1 秒左右），当滑动窗口设置不够长时，可能在第 500ms（只念了一半词时）分数就已经达到 `cutoff`。
- **若没有回落机制**：程序立即判定成功返回给 Home Assistant，HA 随即截断音频并启动录音，导致唤醒词后半截（例如“小智”后面的“智”字）被当作语音指令开头的脏数据录入 STT，造成 STT 识别错乱（如识别成“智把灯打开”）。
- **启用 `trailing_fall: true`**：当分数达标后，程序不会立刻截断返回，而是进入挂起状态继续监听。通常声音发音过程的分数走势类似：`0.1 -> 0.5 -> 0.7 -> 0.9 -> 0.9 -> 0.9 -> 0.7 -> 0.3 -> 0.1`。系统会在分数达到波峰并开始下跌回落时才释放触发，正好切中用户念完唤醒词后 0.3s 的自然停顿窗口，送入 STT 的音频干净且完整！

---

## 5. 启动服务与 Home Assistant 接入

启动容器：
```bash
docker compose up -d
```

查看日志：
```bash
docker compose logs -f wyoming-wakeword
```

### 接入 Home Assistant
1. 打开 Home Assistant，进入 **配置 (Settings)** -> **设备与服务 (Devices & Services)** -> **添加集成 (Add Integration)**。
2. 搜索 **Wyoming Protocol**，填入主机 IP 与端口 `10400`。
3. 前往 **语音助手 (Voice Assistants)** -> 选择流水线，在 **唤醒词 (Wake word)** 中即可选用已配置的模型。
