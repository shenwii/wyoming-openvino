# 唤醒词模型训练流水线 (wakeword_trainer) 极速抄作业教程

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

本教程提供基于 Docker 容器的 **自定义唤醒词全自动端到端训练** 无脑操作流程。
借助 Kokoro OpenVINO 与微软 Edge-TTS 双引擎自动批量合成语音样本、模拟真实房间混响反射与环境噪声混叠，一键训练并自动导出 **Intel NPU 优化 ONNX 模型** 与 **ESPHome / microWakeWord 兼容 TFLite 模型**。

---

## 1. 克隆代码仓库

```bash
git clone --depth=1 'https://github.com/shenwii/wyoming-openvino'
cd wyoming-openvino
```

---

## 2. 复制并编辑配置文件

仓库内预置了不同语种的示例配置文件，直接复制中文模板：

```bash
cp wakeword_trainer/config.sample_zh.yaml ./config.yaml
```

使用编辑器打开 `./config.yaml`，仅需修改顶部的 `wakeword` 区域：

```yaml
wakeword:
  phrase: "你好小智"            # 目标唤醒词
  friend_name: "nihaoxiaozhi_v1" # 模型输出标识名 (最终导出 {friend_name}.onnx 和 {friend_name}.tflite)
  candidate_phrases:             # 相似变体/候补词 (提高不同口音与连读兼容性)
    - "小智小智"
    - "嗨小智"
  hard_negatives:                # 强负样本 / 近音干扰词 (防日常误触发的核心防线)
    - "你好小夏"
    - "你好小李"
    - "开灯"
    - "关灯"
    - "今天天气怎么样"
  language: "zh"                 # 语言代码 (zh, en, ja, ko 等)
```

> 💡 **训练设备建议**: `training.device` 建议保持 `"cpu"`（速度飞快且算子兼容性极佳）。

---

## 3. 一键启动 Docker 全自动训练

执行以下命令，即可在 Docker 隔离环境中自动安装依赖、下载 Kokoro 本地离线语音合成模型并启动全流程流水线：

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

> 💡 **说明**:
> - 若机器没有 Intel 显卡，可直接去掉 `--device /dev/dri/renderD128:/dev/dri/renderD128` 行。
> - 容器将在后台执行，可随时查看实时训练日志。

---

## 4. 实时查看训练进度

```bash
docker logs -f wakeword-trainer
```

流水线将依次全自动执行：
1. **下载基础声学模型与环境数据**（melspectrogram、embedding_model、MIT 真实房间反射脉冲响应 RIR、环境噪音等）。
2. **双引擎语音批量合成**（本地 Kokoro TTS + 微软 Edge-TTS 自动生成数千条正负样本并加权扰动）。
3. **声学环境仿真与混音增强**（模拟房间混响与日常背景噪音）。
4. **神经网络训练与全格式模型导出**。

---

## 5. 产物使用说明

训练完成后，模型文件保存在 `./wakeword_trainer/output/` 目录下：

- **`{friend_name}.onnx`**:
  Intel NPU / GPU / CPU 优化的纯静态形状 ONNX 模型。
  👉 **用法**: 直接复制到 `wyoming-wakeword/models/` 目录中，在 `wakeword_config.yaml` 中配置即可在 Home Assistant 中常驻监听。
- **`{friend_name}.tflite`**:
  标准 32 位浮点 TensorFlow Lite 模型。
  👉 **用法**: 适用于 ESP32-S3 等智能硬件的 **ESPHome microWakeWord** 语音终端。
- **`{friend_name}.json`**:
  模型配置元数据，包含建议的测试阈值 `cutoff`。
- **`{friend_name}.pt`**:
  PyTorch 原始权重文件，后续如需微调或秒级快速重导出，可直接使用 `python train.py --export-only`。
