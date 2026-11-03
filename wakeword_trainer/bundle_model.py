#!/usr/bin/env python3
"""
Stand-alone Wake Word Model Bundler & Re-exporter (bundle_model.py).
=============================================================================
Quickly fixes and bundles wake word models into single, self-contained ONNX and
TensorFlow Lite (.tflite) files compatible with Wyoming, Intel NPU, and microWakeWord.

Usage Examples:
1. Re-export both ONNX and TFLite from PyTorch checkpoint (.pt):
   python3 bundle_model.py output/xiongmaojingling.pt

2. Export only TFLite (microWakeWord / ESPHome compatible):
   python3 bundle_model.py output/xiongmaojingling.pt --tflite
   # or
   python3 bundle_model.py output/xiongmaojingling.pt --output my_model.tflite

3. Convert an existing ONNX model to TFLite:
   python3 bundle_model.py xiongmaojingling.onnx --tflite

4. Bundle separated .onnx and .data files into a self-contained single .onnx file:
   python3 bundle_model.py xiongmaojingling.onnx [--data xiongmaojingling.tmp.onnx.data]
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bundle_model")


def bundle_onnx_external_data(
    onnx_path: Path,
    output_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
) -> bool:
    """Combines an ONNX file and its external .data weights into a single file."""
    try:
        import onnx
        from onnx.external_data_helper import convert_model_from_external_data, load_external_data_for_model
    except ImportError:
        logger.error("The 'onnx' Python package is required. Install via: pip install onnx")
        return False

    onnx_path = Path(onnx_path).resolve()
    if not onnx_path.exists():
        logger.error("ONNX file not found: %s", onnx_path)
        return False

    out_file = Path(output_path).resolve() if output_path else onnx_path
    temp_out = out_file.with_suffix(".bundle_tmp.onnx")

    try:
        logger.info("Inspecting ONNX model: %s", onnx_path)
        model = onnx.load(str(onnx_path), load_external_data=False)

        # Check for external data references
        external_entries = []
        for init in model.graph.initializer:
            if getattr(init, "data_location", 0) == onnx.TensorProto.EXTERNAL or bool(getattr(init, "external_data", [])):
                external_entries.append(init)

        if not external_entries:
            logger.info("✅ Model '%s' is already self-contained (no external weights found).", onnx_path.name)
            return True

        logger.info("Found %d external data tensor(s). Resolving external files...", len(external_entries))
        for init in external_entries:
            for entry in init.external_data:
                if entry.key == "location":
                    loc_file = onnx_path.parent / entry.value
                    if not loc_file.exists():
                        if data_path and Path(data_path).exists():
                            logger.info("Mapping external location '%s' -> '%s'", entry.value, Path(data_path).name)
                            entry.value = Path(data_path).name
                        else:
                            candidates = (
                                list(onnx_path.parent.glob("*.data"))
                                + list(onnx_path.parent.glob("*.tmp.onnx.data"))
                            )
                            if candidates:
                                logger.info("Auto-discovered external data file: '%s'", candidates[0].name)
                                entry.value = candidates[0].name
                            else:
                                raise FileNotFoundError(
                                    f"External data file '{entry.value}' not found in {onnx_path.parent}. "
                                    f"Please provide the external data file path with --data."
                                )

        load_external_data_for_model(model, base_dir=str(onnx_path.parent))
        convert_model_from_external_data(model)
        onnx.save_model(model, str(temp_out), save_as_external_data=False)

        # Verification
        verified = onnx.load(str(temp_out), load_external_data=False)
        onnx.checker.check_model(verified)
        for init in verified.graph.initializer:
            if getattr(init, "data_location", 0) == onnx.TensorProto.EXTERNAL or bool(getattr(init, "external_data", [])):
                raise RuntimeError(f"Initializer '{init.name}' still references external data after conversion!")

        if out_file.exists():
            out_file.unlink()
        temp_out.rename(out_file)

        final_size_kb = out_file.stat().st_size / 1024
        logger.info("🎉 Successfully created self-contained ONNX model: %s (%.1f KB)", out_file, final_size_kb)
        return True

    except Exception as exc:
        logger.error("Failed to bundle external data: %s", exc)
        if temp_out.exists():
            temp_out.unlink()
        return False


def _build_pytorch_model(
    context_length: int = 16,
    embedding_dim: int = 96,
    layer_size: int = 64,
):
    import torch
    import torch.nn as nn

    class WakeWordDNN(nn.Module):
        def __init__(self, ctx_len: int = 16, emb_dim: int = 96, l_size: int = 64):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(ctx_len * emb_dim, l_size)
            self.relu1 = nn.ReLU()
            self.drop1 = nn.Dropout(0.0)
            self.fc2 = nn.Linear(l_size, l_size)
            self.relu2 = nn.ReLU()
            self.drop2 = nn.Dropout(0.0)
            self.out_fc = nn.Linear(l_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.flatten(x)
            h = self.drop1(self.relu1(self.fc1(h)))
            h = self.drop2(self.relu2(self.fc2(h)))
            return self.sigmoid(self.out_fc(h))

    return WakeWordDNN(context_length, embedding_dim, layer_size)


def reexport_onnx_from_checkpoint(
    pt_path: Path,
    output_path: Path,
    context_length: int = 16,
    embedding_dim: int = 96,
    layer_size: int = 64,
    opset: int = 17,
) -> bool:
    """Exports a self-contained static ONNX model directly from PyTorch weights (.pt)."""
    try:
        import torch
    except ImportError:
        logger.error("PyTorch is required. Install via: pip install torch")
        return False

    pt_path = Path(pt_path).resolve()
    if not pt_path.exists():
        logger.error("PyTorch checkpoint file not found: %s", pt_path)
        return False

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_onnx = output_path.with_suffix(".tmp.onnx")

    logger.info("Loading PyTorch weights from %s ...", pt_path)
    model = _build_pytorch_model(context_length, embedding_dim, layer_size)
    state = torch.load(str(pt_path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    logger.info("Exporting self-contained static ONNX to %s ...", output_path)
    dummy_input = torch.zeros(1, context_length, embedding_dim, dtype=torch.float32)

    import inspect
    sig = inspect.signature(torch.onnx.export)
    export_kwargs = {
        "input_names": ["input"],
        "output_names": ["output"],
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamic_axes": None,
    }
    if "export_params" in sig.parameters:
        export_kwargs["export_params"] = True
    if "dynamo" in sig.parameters:
        export_kwargs["dynamo"] = False
    if "use_external_data_format" in sig.parameters:
        export_kwargs["use_external_data_format"] = False

    try:
        torch.onnx.export(model, dummy_input, str(temp_onnx), **export_kwargs)
    except TypeError:
        torch.onnx.export(
            model,
            dummy_input,
            str(temp_onnx),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )

    # Post-process with onnx if available
    try:
        import onnx
        from onnx.external_data_helper import convert_model_from_external_data
        onnx_model = onnx.load(str(temp_onnx), load_external_data=True)
        convert_model_from_external_data(onnx_model)
        onnx.save_model(onnx_model, str(temp_onnx), save_as_external_data=False)
    except Exception:
        pass

    for ext_f in temp_onnx.parent.glob(f"{temp_onnx.name}*.data*"):
        try:
            ext_f.unlink()
        except OSError:
            pass

    if output_path.exists():
        output_path.unlink()
    temp_onnx.rename(output_path)

    final_size_kb = output_path.stat().st_size / 1024
    logger.info("🎉 Successfully exported single ONNX model: %s (%.1f KB)", output_path, final_size_kb)
    return True


def export_tflite_from_model(
    model,
    output_tflite_path: Path,
    context_length: int = 16,
    embedding_dim: int = 96,
    quantize: bool = False,
) -> bool:
    """Exports a trained model into TFLite format (.tflite) for microWakeWord / ESPHome."""
    import numpy as np

    output_tflite_path = Path(output_tflite_path).resolve()
    output_tflite_path.parent.mkdir(parents=True, exist_ok=True)
    temp_tflite = output_tflite_path.with_suffix(".tmp.tflite")

    logger.info("Exporting TFLite (microWakeWord compatible) model to %s ...", output_tflite_path)

    # 1. Try TensorFlow backend
    try:
        import tensorflow as tf
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.get_logger().setLevel("ERROR")

        layer_size = model.fc1.out_features

        keras_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(context_length, embedding_dim), batch_size=1, name="input"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(layer_size, activation="relu", name="fc1"),
            tf.keras.layers.Dense(layer_size, activation="relu", name="fc2"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="out_fc"),
        ])

        fc1_w = model.fc1.weight.detach().cpu().numpy().T
        fc1_b = model.fc1.bias.detach().cpu().numpy()
        fc2_w = model.fc2.weight.detach().cpu().numpy().T
        fc2_b = model.fc2.bias.detach().cpu().numpy()
        out_w = model.out_fc.weight.detach().cpu().numpy().T
        out_b = model.out_fc.bias.detach().cpu().numpy()

        keras_model.get_layer("fc1").set_weights([fc1_w, fc1_b])
        keras_model.get_layer("fc2").set_weights([fc2_w, fc2_b])
        keras_model.get_layer("out_fc").set_weights([out_w, out_b])

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_bytes = converter.convert()
        with open(temp_tflite, "wb") as f:
            f.write(tflite_bytes)

        if output_tflite_path.exists():
            output_tflite_path.unlink()
        temp_tflite.rename(output_tflite_path)

        size_kb = output_tflite_path.stat().st_size / 1024
        logger.info("🎉 Successfully exported TFLite model: %s (%.1f KB)", output_tflite_path, size_kb)
        return True

    except ImportError:
        pass
    except Exception as exc:
        logger.error("TensorFlow export failed: %s", exc)
        if temp_tflite.exists():
            temp_tflite.unlink()

    # 2. Try LiteRT backend
    try:
        try:
            import litert_torch as edge_torch
        except ImportError:
            import ai_edge_torch as edge_torch
        import torch

        dummy_input = torch.zeros(1, context_length, embedding_dim, dtype=torch.float32)
        edge_model = edge_torch.convert(model, (dummy_input,))
        edge_model.export(str(temp_tflite))

        if output_tflite_path.exists():
            output_tflite_path.unlink()
        temp_tflite.rename(output_tflite_path)

        size_kb = output_tflite_path.stat().st_size / 1024
        logger.info("🎉 Successfully exported TFLite model (LiteRT): %s (%.1f KB)", output_tflite_path, size_kb)
        return True

    except ImportError:
        pass
    except Exception as exc:
        logger.error("LiteRT export failed: %s", exc)
        if temp_tflite.exists():
            temp_tflite.unlink()

    logger.warning(
        "⚠️ TensorFlow or litert-torch is required for TFLite export. Please install: pip install tensorflow"
    )
    return False


def reexport_tflite_from_checkpoint(
    pt_path: Path,
    output_path: Path,
    context_length: int = 16,
    embedding_dim: int = 96,
    layer_size: int = 64,
    quantize: bool = False,
) -> bool:
    """Exports TFLite model directly from PyTorch weights (.pt)."""
    try:
        import torch
    except ImportError:
        logger.error("PyTorch is required. Install via: pip install torch")
        return False

    pt_path = Path(pt_path).resolve()
    if not pt_path.exists():
        logger.error("PyTorch checkpoint file not found: %s", pt_path)
        return False

    model = _build_pytorch_model(context_length, embedding_dim, layer_size)
    state = torch.load(str(pt_path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return export_tflite_from_model(
        model=model,
        output_tflite_path=output_path,
        context_length=context_length,
        embedding_dim=embedding_dim,
        quantize=quantize,
    )


def convert_onnx_to_tflite(
    onnx_path: Path,
    output_tflite_path: Path,
    quantize: bool = False,
) -> bool:
    """Converts a self-contained ONNX model into TFLite format."""
    try:
        import onnx
        from onnx import numpy_helper
        import numpy as np
    except ImportError:
        logger.error("onnx and numpy are required. Install via: pip install onnx numpy")
        return False

    onnx_path = Path(onnx_path).resolve()
    if not onnx_path.exists():
        logger.error("ONNX file not found: %s", onnx_path)
        return False

    output_tflite_path = Path(output_tflite_path).resolve()
    output_tflite_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import tensorflow as tf
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.get_logger().setLevel("ERROR")

        model_proto = onnx.load(str(onnx_path), load_external_data=True)
        initializers = {init.name: numpy_helper.to_array(init) for init in model_proto.graph.initializer}

        fc1_w, fc1_b, fc2_w, fc2_b, out_w, out_b = None, None, None, None, None, None
        for name, arr in initializers.items():
            if "fc1.weight" in name or name == "fc1.weight":
                fc1_w = arr
            elif "fc1.bias" in name or name == "fc1.bias":
                fc1_b = arr
            elif "fc2.weight" in name or name == "fc2.weight":
                fc2_w = arr
            elif "fc2.bias" in name or name == "fc2.bias":
                fc2_b = arr
            elif "out_fc.weight" in name or name == "out_fc.weight":
                out_w = arr
            elif "out_fc.bias" in name or name == "out_fc.bias":
                out_b = arr

        if fc1_w is not None and fc2_w is not None and out_w is not None:
            layer_size = fc1_w.shape[0]
            in_dim = fc1_w.shape[1]
            context_length = 16
            embedding_dim = in_dim // context_length

            keras_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(context_length, embedding_dim), batch_size=1, name="input"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(layer_size, activation="relu", name="fc1"),
                tf.keras.layers.Dense(layer_size, activation="relu", name="fc2"),
                tf.keras.layers.Dense(1, activation="sigmoid", name="out_fc"),
            ])

            keras_model.get_layer("fc1").set_weights([fc1_w.T, fc1_b if fc1_b is not None else np.zeros(layer_size, dtype=np.float32)])
            keras_model.get_layer("fc2").set_weights([fc2_w.T, fc2_b if fc2_b is not None else np.zeros(layer_size, dtype=np.float32)])
            keras_model.get_layer("out_fc").set_weights([out_w.T, out_b if out_b is not None else np.zeros(1, dtype=np.float32)])

            converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            tflite_bytes = converter.convert()
            with open(output_tflite_path, "wb") as f:
                f.write(tflite_bytes)

            size_kb = output_tflite_path.stat().st_size / 1024
            logger.info("🎉 Successfully converted ONNX to TFLite: %s (%.1f KB)", output_tflite_path, size_kb)
            return True

    except ImportError:
        pass
    except Exception as exc:
        logger.error("Failed to convert ONNX to TFLite via TensorFlow: %s", exc)

    logger.warning("TensorFlow is required to convert ONNX to TFLite. Install via: pip install tensorflow")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Bundle ONNX external weights or re-export PyTorch weights into ONNX and TFLite models."
    )
    parser.add_argument("input_file", nargs="?", default=None, help="Input model file (.pt or .onnx)")
    parser.add_argument("--onnx", type=str, default=None, help="Path to input ONNX model file")
    parser.add_argument("--data", type=str, default=None, help="Path to external .data file (optional)")
    parser.add_argument("--pt", type=str, default=None, help="Path to PyTorch checkpoint (.pt) for fast re-export")
    parser.add_argument("--output", type=str, default=None, help="Destination output path (.onnx or .tflite)")
    parser.add_argument("--format", type=str, choices=["all", "onnx", "tflite"], default="all", help="Output format [default: all]")
    parser.add_argument("--tflite", action="store_true", help="Shortcut to export TFLite format (microWakeWord / ESPHome)")
    parser.add_argument("--quantize", action="store_true", help="Enable dynamic range INT8 quantization for TFLite")
    parser.add_argument("--context-length", type=int, default=16, help="Context length [default: 16]")
    parser.add_argument("--embedding-dim", type=int, default=96, help="Embedding dimension [default: 96]")
    parser.add_argument("--layer-size", type=int, default=64, help="Hidden layer size [default: 64]")
    args = parser.parse_args()

    # Automatically resolve positional argument if provided
    if args.input_file:
        if args.input_file.endswith(".pt") and not args.pt:
            args.pt = args.input_file
        elif (args.input_file.endswith(".onnx") or args.input_file.endswith(".onnx.data")) and not args.onnx:
            args.onnx = args.input_file

    # Handle .pt input
    if args.pt:
        pt_path = Path(args.pt).resolve()
        # Resolve target format
        do_onnx = True
        do_tflite = True

        if args.tflite:
            do_onnx = False
            do_tflite = True
        elif args.output and args.output.endswith(".tflite"):
            do_onnx = False
            do_tflite = True
        elif args.output and args.output.endswith(".onnx"):
            do_onnx = True
            do_tflite = False
        elif args.format == "onnx":
            do_onnx = True
            do_tflite = False
        elif args.format == "tflite":
            do_onnx = False
            do_tflite = True

        success = True
        if do_onnx:
            out_onnx = Path(args.output) if (args.output and args.output.endswith(".onnx")) else pt_path.with_suffix(".onnx")
            ok = reexport_onnx_from_checkpoint(
                pt_path=pt_path,
                output_path=out_onnx,
                context_length=args.context_length,
                embedding_dim=args.embedding_dim,
                layer_size=args.layer_size,
            )
            success = success and ok

        if do_tflite:
            out_tflite = Path(args.output) if (args.output and args.output.endswith(".tflite")) else pt_path.with_suffix(".tflite")
            ok = reexport_tflite_from_checkpoint(
                pt_path=pt_path,
                output_path=out_tflite,
                context_length=args.context_length,
                embedding_dim=args.embedding_dim,
                layer_size=args.layer_size,
                quantize=args.quantize,
            )
            success = success and ok

        sys.exit(0 if success else 1)

    # Handle .onnx input
    if args.onnx:
        onnx_path = Path(args.onnx).resolve()
        if args.tflite or (args.output and args.output.endswith(".tflite")) or args.format == "tflite":
            out_tflite = Path(args.output) if (args.output and args.output.endswith(".tflite")) else onnx_path.with_suffix(".tflite")
            ok = convert_onnx_to_tflite(
                onnx_path=onnx_path,
                output_tflite_path=out_tflite,
                quantize=args.quantize,
            )
            sys.exit(0 if ok else 1)
        else:
            out_path = Path(args.output) if args.output else None
            ok = bundle_onnx_external_data(
                onnx_path=onnx_path,
                output_path=out_path,
                data_path=Path(args.data) if args.data else None,
            )
            sys.exit(0 if ok else 1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
