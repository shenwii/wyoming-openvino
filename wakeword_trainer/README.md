# Wyoming / openWakeWord NPU 唤醒词全流程训练套件

基于 **[openWakeWord](https://github.com/dscripka/openWakeWord)** 与 **[MicroWakeWordV2Trainer](https://github.com/JohnnyPrimus/MicroWakeWordV2Trainer)** 设计，专为 **Intel NPU (OpenVINO VPUX / Core Ultra)** 深度定制的自定义唤醒词端到端训练套件。

---

## 🌟 核心特性与设计亮点

1. **统一与结构化配置 (`config.yaml`)**:
   - 集中定义目标唤醒词（主词）、候补/变体词（提高鲁棒性与口音容忍）、NG 负向干扰词（Hard Negatives / 近音词），强力抑制误触发。
   - 语种自由切换（中文 `zh`、英文 `en`、日文 `ja` 等）。
2. **双引擎协同 TTS 生成 (`generate_tts.py`)**:
   - **引擎 A (本地离线加速)**: 深度复用本项目内置的 `Kokoro OpenVINO TTS`，支持多发音人加权融合（Voice Blending）与丰富语速变化。
   - **引擎 B (云端高质量)**: 整合微软免费的 `Edge-TTS`，覆盖海量高保真神经发音人，并扰动语速（Rate）、音调（Pitch）与音量（Volume）。
3. **真实声学仿真与环境混音 (`augment_data.py`)**:
   - 参考 MicroWakeWordV2Trainer，下载并应用 **MIT Environmental RIR**（房间冲激响应）实现真实房间反射混响卷积。
   - 随机以不同信噪比（SNR 5~25dB）混叠环境背景噪声（如风扇、电视、生活家电）。
   - 边界静音时间位移与音量扰动，大幅增强实际麦克风拾音鲁棒性。
4. **负样本深度融合 (Hard Negatives + 2000 小时 ACAV100M)**:
   - 自动生成近音词（NG 词）作为强对抗负样本。
   - 支持自动下载与挂载 openWakeWord 官方 ACAV100M 2000 小时预提取语音特征，有效防止日常交谈与电视播放时的误触发。
5. **针对 Intel NPU 的极致优化 (`train.py`)**:
   - **纯静态形状导出**: 输出严格静态的 `[1, 16, 96] -> [1, 1]` ONNX 模型，杜绝 Intel NPU 驱动因动态 Shape 导致的编译崩溃或延迟飙升。
   - **纯净算子图**: 使用纯粹的 `Gemm (Linear) + ReLU + Sigmoid`，零复杂非标算子，确保 **100% 在 NPU 上由硬件执行**，零 CPU 回退。
   - **开箱即用**: 生成的 ONNX 模型与主项目的 `wyoming-wakeword.py` 无缝对接。

---

## 📁 目录结构

```text
wakeword_trainer/
├── config.yaml          # 主配置文件 (详细注释：词汇/TTS/数据下载/训练超参/NPU优化)
├── requirements.txt     # 训练套件独立依赖清单
├── setup_libs.py        # 第三方开源库与基础特征提取模型配置脚本
├── download_data.py     # 声学环境、RIR混响与官方负样本特征下载器
├── generate_tts.py      # 双引擎 (Kokoro + Edge-TTS) 正样本与 NG 词生成器
├── augment_data.py      # 声学环境仿真、混响卷积与环境噪声混音增强器
├── train.py             # 特征抽取、判别模型训练与 NPU 优化 ONNX 导出器
├── run_pipeline.py      # 一键全自动端到端调度器 (CLI)
├── README.md            # 本使用说明文档
├── data/                # [运行时生成] 存放合成语音、增强音频、RIR与背景噪声
├── lib/                 # [运行时生成] 第三方特征提取模型与 upstream 源码
└── output/              # [运行时生成] 导出的 ONNX 模型与元数据配置文件
```

---

## 🛠️ 环境准备与依赖安装

在进行训练前，请先安装训练套件所需的 Python 依赖：

```bash
cd wakeword_trainer
pip install -r requirements.txt
```

> **提示**:
> - 音频处理依赖系统级 `ffmpeg` 工具，若系统未安装请通过包管理器安装（Ubuntu/Debian: `sudo apt install ffmpeg`）。
> - 若使用本地 Kokoro TTS，请确保主目录的 Kokoro OpenVINO 环境可用；若未部署 Kokoro，脚本将自动全量使用微软 Edge-TTS 进行合成。

---

## ⚙️ 配置文件定制 (`config.yaml`)

修改 `config.yaml` 即可完成全部定制。关键字段说明：

### 1. 唤醒词与干扰词定义
```yaml
wakeword:
  # 目标唤醒词 (标记为 1.0)
  phrase: "你好小智"

  # 候补/变体词 (标记为 1.0，增加发音容忍度)
  candidate_phrases:
    - "小智小智"
    - "嗨小智"
    - "小智同学"

  # NG 负向干扰词 / 近音词 (标记为 0.0，最核心的防误唤醒手段！)
  hard_negatives:
    - "你好小夏"
    - "你好小李"
    - "你好"
    - "小智"
    - "下雪了"
    - "天气预报"

  # 语言代码: zh (中文), en (英语), ja (日语) 等
  language: "zh"
```

### 2. 双引擎 TTS 生成参数
```yaml
tts:
  target_positive_samples: 2000  # 正样本目标数量
  target_negative_samples: 1000  # NG 干扰词目标数量

  kokoro:
    enabled: true
    speed_range: [0.75, 1.25]    # 语速扰动区间
    enable_voice_blending: true  # 启用多音色融合扩充

  edge_tts:
    enabled: true
    rate_range: ["-20%", "+20%"] # 语速变化范围
    pitch_range: ["-15Hz", "+15Hz"] # 音调变化范围
```

### 3. NPU 专属优化设置
```yaml
training:
  model_name: "nihaoxiaozhi_v1"
  steps: 20000                   # 训练迭代步数
  batch_size: 128
  
  npu_optimization:
    force_static_shape: true     # 强制固定纯静态形状 [1, 16, 96]
    batch_size: 1
    onnx_opset: 17
    pure_gemm_graph: true        # 纯净 Gemm + ReLU 算子图
```

---

## 🚀 训练操作指南

### 方式一: 全自动一键端到端训练 (推荐)

直接运行流水线主入口，它将按顺序自动完成：环境初始化 -> 数据准备 -> TTS合成 -> 混音增强 -> 特征抽取与模型训练 -> NPU ONNX 导出。

```bash
python run_pipeline.py --config config.yaml
```

若希望先以少量样本快速验证全流程，可开启快速模式：
```bash
python run_pipeline.py --quick-mode
```

---

### 方式二: 分步骤单独执行与调试

你可以单独执行流水线中的任意步骤进行精细化调整：

#### 步骤 1: 准备 openWakeWord 官方特征提取基础模型
```bash
python setup_libs.py --models-only
```
该命令会自动下载 `melspectrogram.onnx` 与 `embedding_model.onnx` 到 `lib/models/` 目录。

#### 步骤 2: 下载声学噪声与 RIR 冲激响应数据
```bash
# 下载 MIT RIR 混响数据与环境噪声
python download_data.py --rirs --noise

# (可选) 同时下载官方 2000 小时 ACAV100M 负样本特征集 (~2.5GB)
python download_data.py --negatives
```

#### 步骤 3: 运行 TTS 批量合成唤醒词与 NG 词
```bash
python generate_tts.py --config config.yaml
```
生成的 16kHz 16-bit 单声道音频与索引文件将保存在 `data/synthetic_speech/` 中。

#### 步骤 4: 执行真实房间混响与背景噪声增强
```bash
python augment_data.py --config config.yaml
```
增强后的仿真音频将保存在 `data/augmented/` 中。

#### 步骤 5: 训练神经网络并导出 NPU 优化 ONNX 模型
```bash
python train.py --config config.yaml --steps 20000 --batch-size 128
```
训练完成后，最终模型将输出至 `output/{model_name}.onnx`，并附带 `{model_name}.json` 元信息文件。

---

## 🎯 部署到 Wyoming 唤醒词服务

训练完成后导出的 ONNX 模型已针对 Intel NPU 的静态 Shape 及算子要求完成了端到端对齐，可以直接在主项目的 `wyoming-wakeword.py` 中加载并运行：

### 命令行方式启动:
```bash
python wyoming-wakeword.py \
    --model wakeword_trainer/output/nihaoxiaozhi_v1.onnx \
    --device NPU \
    --cutoff 0.55 \
    --refractory 2.0
```

### YAML 配置文件方式启动:
在 `wyoming-wakeword/wakeword_config.yaml` 的 `models` 列表中添加：
```yaml
models:
  - name: "nihaoxiaozhi"
    model_path: "../wakeword_trainer/output/nihaoxiaozhi_v1.onnx"
    phrase: "你好小智"
    cutoff: 0.55
    refractory: 2.0
    device: "NPU"
    languages: ["zh"]
```
然后进入 `wyoming-wakeword` 目录启动服务：
```bash
cd wyoming-wakeword
python wyoming-wakeword.py --config wakeword_config.yaml
```

---

## 💡 NPU 性能与调优建议

1. **置信度阈值 (Cutoff)**:
   - 默认推荐值为 `0.55`。
   - 若日常使用中出现偶然误触，建议微调提升至 `0.60` ~ `0.70`；若发现较远距离拾音困难，可适当降低至 `0.45` ~ `0.50`。
2. **NG 负样本的威力**:
   - 如果发现模型对某个特定的词（例如电视里的某个广告词或日常口头禅）容易误唤醒，只需在 `config.yaml` 的 `hard_negatives` 列表中追加该词，重新运行 `generate_tts.py`、`augment_data.py` 和 `train.py`，模型即可学会精准区分。
3. **NPU 延迟优势**:
   - 纯静态 Shape 的 ONNX 模型在 Intel NPU 上单帧判定耗时通常小于 **0.1 毫秒**，且功耗极低，非常适合 7x24 小时常驻监听。
