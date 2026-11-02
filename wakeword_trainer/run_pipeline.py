#!/usr/bin/env python3
"""
Automated wake word model training pipeline runner (run_pipeline.py).
===================================================================
End-to-end pipeline steps:
  Step 1: Environment & base feature extraction model setup (setup_libs.py)
  Step 2: Acoustic environment, RIR, and negative feature download (download_data.py)
  Step 3: Dual-engine multi-voice TTS synthesis for positive & negative audio (generate_tts.py)
  Step 4: Acoustic simulation, reverberation convolution & noise mixing (augment_data.py)
  Step 5: openWakeWord classifier training & NPU-optimized ONNX export (train.py)

Usage examples:
    # 1. Complete end-to-end pipeline run (recommended)
    python run_pipeline.py --config config.yaml

    # 2. Run a specific step
    python run_pipeline.py --step tts
    python run_pipeline.py --step augment
    python run_pipeline.py --step train

    # 3. Quick test mode (fewer samples and steps for fast verification)
    python run_pipeline.py --quick-mode
"""

import argparse
import logging
from pathlib import Path
import subprocess
import sys
import time

from i18n import t

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


def run_command(cmd: list, desc: str) -> bool:
    """Execute a single pipeline step and report status."""
    print(f"\n{t('pipeline.step_exec', desc=desc)}")
    print(t("pipeline.step_cmd", cmd=" ".join(cmd)))
    start = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        logger.info(t("pipeline.step_complete", desc=desc, elapsed=elapsed))
        return True
    except subprocess.CalledProcessError as e:
        logger.error(t("pipeline.step_failed", desc=desc, code=e.returncode))
        return False
    except Exception as exc:
        logger.error(t("pipeline.step_error", desc=desc, error=str(exc)))
        return False


def main():
    parser = argparse.ArgumentParser(description=t("cli.pipeline_help"))
    parser.add_argument("--config", default="config.yaml", help="Path to config file [default: config.yaml]")
    parser.add_argument(
        "--step",
        choices=["all", "setup", "download", "tts", "augment", "train"],
        default="all",
        help="Specify single pipeline stage to execute [default: all]",
    )
    parser.add_argument("--quick-mode", action="store_true", help="Enable quick testing mode")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    cfg_file = (base_dir / args.config).resolve()

    if not cfg_file.exists():
        candidate_sample = base_dir / "config.sample.yaml"
        if candidate_sample.exists():
            cfg_file = candidate_sample
        else:
            logger.error(f"Configuration file not found: {cfg_file}")
            sys.exit(1)

    py_exe = sys.executable

    print("======================================================================")
    print("🚀 Wyoming / openWakeWord NPU Pipeline Runner")
    print(f"   - Config file: {cfg_file}")
    print(f"   - Mode: {args.step.upper()}")
    print(f"   - Quick mode: {'ON' if args.quick_mode else 'OFF'}")
    print("======================================================================")

    steps_to_run = []
    if args.step == "all":
        steps_to_run = ["setup", "download", "tts", "augment", "train"]
    else:
        steps_to_run = [args.step]

    # 1. Base models and libraries
    if "setup" in steps_to_run:
        cmd = [py_exe, str(base_dir / "setup_libs.py"), "--models-only"]
        ok = run_command(cmd, "Step 1: Setup openWakeWord base feature models")
        if not ok and args.step == "all":
            logger.warning("Feature model setup encountered issues, continuing if models exist locally...")

    # 2. Download datasets
    if "download" in steps_to_run:
        cmd = [py_exe, str(base_dir / "download_data.py"), "--config", str(cfg_file)]
        if args.quick_mode:
            cmd.append("--quick-mode")
        ok = run_command(cmd, "Step 2: Download environmental noise, RIRs, and negative features")
        if not ok and args.step == "all":
            logger.warning("Dataset download partially completed, continuing with local data...")

    # 3. TTS speech synthesis
    if "tts" in steps_to_run:
        cmd = [py_exe, str(base_dir / "generate_tts.py"), "--config", str(cfg_file)]
        if args.quick_mode:
            cmd.extend(["--count-scale", "0.05"])
        ok = run_command(cmd, "Step 3: Dual-engine (Kokoro + Edge-TTS) speech synthesis")
        if not ok:
            logger.error("TTS generation failed. Aborting pipeline.")
            sys.exit(1)

    # 4. Audio augmentation and mixing
    if "augment" in steps_to_run:
        cmd = [py_exe, str(base_dir / "augment_data.py"), "--config", str(cfg_file)]
        ok = run_command(cmd, "Step 4: Acoustic simulation, RIR reverberation, and noise mixing")
        if not ok:
            logger.error("Audio augmentation failed. Aborting pipeline.")
            sys.exit(1)

    # 5. Model training & NPU-optimized ONNX export
    if "train" in steps_to_run:
        cmd = [py_exe, str(base_dir / "train.py"), "--config", str(cfg_file)]
        if args.quick_mode:
            cmd.extend(["--steps", "500", "--batch-size", "32"])
        ok = run_command(cmd, "Step 5: Classifier training & Intel NPU-optimized ONNX export")
        if not ok:
            logger.error("Model training failed.")
            sys.exit(1)

    print("\n======================================================================")
    print(t("pipeline.all_complete"))
    print("======================================================================\n")


if __name__ == "__main__":
    main()
