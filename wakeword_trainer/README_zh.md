# Wyoming / openWakeWord NPU 唤醒词全流程训练套件

基于 **[openWakeWord](https://github.com/dscripka/openWakeWord)** 与 **[MicroWakeWordV2Trainer](https://github.com/JohnnyPrimus/MicroWakeWordV2Trainer)** 设计，专为 **Intel NPU (OpenVINO VPUX / Core Ultra)** 与硬件加速深度定制的自定义唤醒词端到端训练套件。

---

## 🌟 核心特性与设计亮点

1. **结构化统一配置 (`config.yaml`)**:
   - 核心参数均收敛在 `wakeword` 区域：目标唤醒词、`friend_name`（不带空格的友好模型名）、候补/变体词、NG 负向干扰词（Hard Negatives / 近音词）。
   - 支持多语种自由切换（中文 `zh`、英文 `en`、日文 `ja` 等）。
2. **双引擎协同 TTS 生成 (`generate_tts.py`)**:
   - **本地引擎 (Kokoro OpenVINO TTS)**: 本地极速离线合成，支持音色加权融合与语速变化。
   - **云端引擎 (微软 Edge-TTS)**: 高保真神经发音人，支持语速、音调与音量扰动。
3. **真实声学仿真与环境混音 (`augment_data.py`)**:
   - 卷积 MIT Environmental RIR 实现真实房间反射混响。
   - 随机信噪比（SNR 5~25dB）混叠日常生活背景噪声。
   - 边界静音时间位移与音量扰动，大幅增强实际拾音鲁棒性。
4. **负样本深度融合 (Hard Negatives + 2000 小时 ACAV100M)**:
   - 自动合成近音词作为强对抗负样本，强力抑制日常误触发。
5. **针对 Intel NPU 与硬件加速的专项优化 (`train.py`)**:
   - **纯静态形状导出**: 输出严格静态的 `[1, 16, 96] -> [1, 1]` ONNX 模型，杜绝驱动编译崩溃。
   - **纯净算子图**: 使用纯粹的 `Gemm (Linear) + ReLU + Sigmoid`，确保 100% 在 NPU 上硬件加速。
   - **硬件加速支持**: 支持 CPU、CUDA 以及 Intel GPU（通过 PyTorch 2.4+ 原生 `xpu` 或 IPEX）进行训练，并支持 OpenVINO `GPU` 进行特征提取加速。

---

## 📁 目录结构

```text
wakeword_trainer/
├── config.sample.yaml     # 示例配置文件 (英文默认)
├── config.sample_zh.yaml  # 示例配置文件 (中文版)
├── config.sample_ja.yaml  # 示例配置文件 (日文版)
├── config.sample_ko.yaml  # 示例配置文件 (韩文版)
├── requirements.txt       # 依赖清单
├── i18n.py                # 国际化引擎
├── lang/                  # 多国语言包 (en, zh, ja, ko)
├── setup_libs.py          # 基础特征提取模型下载脚本
├── download_data.py       # 声学环境、RIR 与负样本数据下载器
├── generate_tts.py        # 双引擎合成数据生成器
├── augment_data.py        # 声学环境仿真与混音增强器
├── train.py               # 判别模型训练与 NPU 优化 ONNX 导出器
├── run_pipeline.py        # 一键全自动端到端调度器
├── README.md              # 英文主文档
├── README_zh.md           # 中文文档
├── data/                  # [运行时生成] 存放合成语音、增强音频、RIR与背景噪声
├── lib/                   # [运行时生成] 特征提取模型
└── output/                # [运行时生成] 导出的 ONNX 模型与元数据
```

---

## 🛠️ 环境准备与依赖安装

```bash
cd wakeword_trainer
pip install -r requirements.txt
```

> **提示**:
> - 音频处理依赖系统级 `ffmpeg` 工具（Ubuntu/Debian: `sudo apt install ffmpeg`）。
> - 若使用本地 Kokoro TTS，请确保主目录的 Kokoro OpenVINO 环境可用；若未部署，将自动使用微软 Edge-TTS。

---

## ⚙️ 配置文件定制 (`config.yaml`)

复制中文示例配置文件：
```bash
cp config.sample_zh.yaml config.yaml
```

用户只需修改 `wakeword` 区域：
```yaml
wakeword:
  phrase: "你好小智"            # 目标唤醒词
  friend_name: "nihaoxiaozhi_v1" # 友好模型名 (决定输出文件名: {friend_name}.onnx)
  candidate_phrases:             # 候补/变体词
    - "小智小智"
    - "嗨小智"
  hard_negatives:                # NG 负向干扰词 / 近音词
    - "你好小夏"
    - "你好小李"
    - "开灯"
  language: "zh"                 # 语言代码
```

硬件设备配置（在 `training:` 区域）：
```yaml
training:
  # "auto", "cpu", "cuda", 或 "xpu" (Intel Arc / Core Ultra GPU，需 PyTorch 2.4+ XPU 或 IPEX)
  device: "auto"

  # "auto", "CPU", 或 "GPU" (OpenVINO Intel GPU)
  feature_device: "auto"
```

---

## 🚀 训练操作指南

### 方式一: 全自动一键端到端训练 (推荐)

```bash
python run_pipeline.py --config config.yaml
```

快速测试轻量模式：
```bash
python run_pipeline.py --quick-mode
```

---

### 方式二: 分步骤单独执行

1. **准备特征提取基础模型**:
   ```bash
   python setup_libs.py --models-only
   ```
2. **下载声学噪声与 RIR 冲激响应**:
   ```bash
   python download_data.py --rirs --noise
   ```
3. **TTS 批量合成正向词与 NG 词**:
   ```bash
   python generate_tts.py --config config.yaml
   ```
4. **执行房间混响与背景噪声增强**:
   ```bash
   python augment_data.py --config config.yaml
   ```
5. **训练神经网络并导出 NPU 优化 ONNX 模型**:
   ```bash
   python train.py --config config.yaml --steps 20000 --batch-size 128
   ```

最终模型将输出至 `output/{friend_name}.onnx`，并附带 `{friend_name}.json` 元信息文件。

---

## 🎯 部署到 Wyoming 唤醒词服务

```bash
cd ../wyoming-wakeword
python wyoming-wakeword.py \
    --model ../wakeword_trainer/output/nihaoxiaozhi_v1.onnx \
    --device NPU \
    --cutoff 0.55 \
    --refractory 2.0
```
