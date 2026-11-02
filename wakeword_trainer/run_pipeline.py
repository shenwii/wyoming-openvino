#!/usr/bin/env python3
"""
一键全自动唤醒词模型训练流水线 (run_pipeline.py)
==============================================
一键贯通:
  Step 1: 环境与基础特征模型初始化 (setup_libs.py)
  Step 2: 声学环境、RIR 冲激响应与负样本数据准备 (download_data.py)
  Step 3: 双引擎多参数 TTS 唤醒词与 NG 词生成 (generate_tts.py)
  Step 4: 真实环境声学仿真与混音增强 (augment_data.py)
  Step 5: openWakeWord 判别模型训练与 NPU 优化 ONNX 导出 (train.py)

使用示例:
    # 完整全自动端到端运行 (推荐)
    python run_pipeline.py --config config.yaml

    # 仅执行语音生成与训练
    python run_pipeline.py --step tts
    python run_pipeline.py --step augment
    python run_pipeline.py --step train

    # 快速测试轻量模式
    python run_pipeline.py --quick-mode
"""

import argparse
import logging
from pathlib import Path
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


def run_command(cmd: list, desc: str) -> bool:
    """执行单个流水线步骤并捕获状态。"""
    print(f"\n▶️  正在执行: {desc}")
    print(f"   命令: {' '.join(cmd)}")
    start = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        logger.info("✅ 步骤完成: %s (耗时: %.1f 秒)", desc, elapsed)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("❌ 步骤失败: %s (退出码: %d)", desc, e.returncode)
        return False
    except Exception as exc:
        logger.error("❌ 执行异常: %s (%s)", desc, exc)
        return False


def main():
    parser = argparse.ArgumentParser(description="唤醒词全自动训练流水线调度器")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径 (默认: config.yaml)")
    parser.add_argument(
        "--step",
        choices=["all", "setup", "download", "tts", "augment", "train"],
        default="all",
        help="指定执行的单个阶段 (默认: all 执行全流程)",
    )
    parser.add_argument("--quick-mode", action="store_true", help="启用快速轻量模式 (减少生成样本与训练步数以快速跑通测试)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    cfg_file = (base_dir / args.config).resolve()

    if not cfg_file.exists():
        logger.error("配置文件不存在: %s", cfg_file)
        sys.exit(1)

    py_exe = sys.executable

    print("======================================================================")
    print("🚀 Wyoming / openWakeWord NPU 唤醒词全流程自动化流水线")
    print(f"   - 配置文件: {cfg_file}")
    print(f"   - 执行模式: {args.step.upper()}")
    print(f"   - 快速模式: {'开启' if args.quick_mode else '关闭'}")
    print("======================================================================")

    steps_to_run = []
    if args.step == "all":
        steps_to_run = ["setup", "download", "tts", "augment", "train"]
    else:
        steps_to_run = [args.step]

    # 1. 基础模型与库
    if "setup" in steps_to_run:
        cmd = [py_exe, str(base_dir / "setup_libs.py"), "--models-only"]
        ok = run_command(cmd, "Step 1: 准备 openWakeWord 官方特征提取基础模型")
        if not ok and args.step == "all":
            logger.warning("特征模型准备遇到网络问题，若本地已存在模型可继续...")

    # 2. 下载数据
    if "download" in steps_to_run:
        cmd = [py_exe, str(base_dir / "download_data.py"), "--config", str(cfg_file)]
        if args.quick_mode:
            cmd.append("--quick-mode")
        ok = run_command(cmd, "Step 2: 下载环境噪声、RIR 冲激响应与负样本数据")
        if not ok and args.step == "all":
            logger.warning("数据集下载未完全成功，后续增强将利用本地已有音频进行...")

    # 3. TTS 数据合成
    if "tts" in steps_to_run:
        cmd = [py_exe, str(base_dir / "generate_tts.py"), "--config", str(cfg_file)]
        if args.quick_mode:
            cmd.extend(["--count-scale", "0.05"])
        ok = run_command(cmd, "Step 3: 双引擎 (Kokoro + Edge-TTS) 正样本与 NG 干扰词合成")
        if not ok:
            logger.error("❌ TTS 音频生成失败，流水线中止。")
            sys.exit(1)

    # 4. 数据增强与混音
    if "augment" in steps_to_run:
        cmd = [py_exe, str(base_dir / "augment_data.py"), "--config", str(cfg_file)]
        ok = run_command(cmd, "Step 4: 声学环境混音、RIR 混响与时间扰动增强")
        if not ok:
            logger.error("❌ 数据增强失败，流水线中止。")
            sys.exit(1)

    # 5. 模型训练与 NPU 优化 ONNX 导出
    if "train" in steps_to_run:
        cmd = [py_exe, str(base_dir / "train.py"), "--config", str(cfg_file)]
        if args.quick_mode:
            cmd.extend(["--steps", "500", "--batch-size", "32"])
        ok = run_command(cmd, "Step 5: 判别网络训练与 Intel NPU 优化 ONNX 模型导出")
        if not ok:
            logger.error("❌ 模型训练失败。")
            sys.exit(1)

    print("\n======================================================================")
    print("🎉 流水线执行完毕！")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
