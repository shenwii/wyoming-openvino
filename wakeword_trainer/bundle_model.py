#!/usr/bin/env python3
"""
Stand-alone Wake Word Model Bundler & Re-exporter (bundle_model.py).
=============================================================================
Quickly fixes and bundles wake word models into single, self-contained ONNX files
compatible with Wyoming and Intel NPU.

Features:
1. Bundle separated .onnx and .data files into a single, self-contained .onnx file:
   python3 bundle_model.py --onnx xiongmaojingling.onnx [--data xiongmaojingling.tmp.onnx.data]

2. Fast re-export from PyTorch checkpoint (.pt) without retraining:
   python3 bundle_model.py --pt output/xiongmaojingling.pt --output xiongmaojingling.onnx
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
            if init.has_field("data_location") and init.data_location == onnx.TensorProto.EXTERNAL:
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
            if init.has_field("data_location") and init.data_location == onnx.TensorProto.EXTERNAL:
                raise RuntimeError(f"Initializer '{init.name}' still references external data after conversion!")

        if out_file.exists():
            out_file.unlink()
        temp_out.rename(out_file)

        final_size_kb = out_file.stat().st_size / 1024
        logger.info("🎉 Successfully created self-contained model: %s (%.1f KB)", out_file, final_size_kb)
        return True

    except Exception as exc:
        logger.error("Failed to bundle external data: %s", exc)
        if temp_out.exists():
            temp_out.unlink()
        return False


def reexport_from_checkpoint(
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
        import torch.nn as nn
    except ImportError:
        logger.error("PyTorch is required. Install via: pip install torch")
        return False

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

    pt_path = Path(pt_path).resolve()
    if not pt_path.exists():
        logger.error("PyTorch checkpoint file not found: %s", pt_path)
        return False

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_onnx = output_path.with_suffix(".tmp.onnx")

    logger.info("Loading PyTorch weights from %s ...", pt_path)
    model = WakeWordDNN(context_length, embedding_dim, layer_size)
    state = torch.load(str(pt_path), map_location="cpu", weights_only=True)
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


def main():
    parser = argparse.ArgumentParser(
        description="Bundle ONNX external weights or re-export PyTorch weights into a single self-contained ONNX model."
    )
    parser.add_argument("--onnx", type=str, default=None, help="Path to input ONNX model file")
    parser.add_argument("--data", type=str, default=None, help="Path to external .data file (optional)")
    parser.add_argument("--pt", type=str, default=None, help="Path to PyTorch checkpoint (.pt) for fast re-export")
    parser.add_argument("--output", type=str, default=None, help="Destination output ONNX path")
    parser.add_argument("--context-length", type=int, default=16, help="Context length [default: 16]")
    parser.add_argument("--embedding-dim", type=int, default=96, help="Embedding dimension [default: 96]")
    parser.add_argument("--layer-size", type=int, default=64, help="Hidden layer size [default: 64]")
    args = parser.parse_args()

    if args.pt:
        out_path = Path(args.output or Path(args.pt).with_suffix(".onnx"))
        ok = reexport_from_checkpoint(
            pt_path=Path(args.pt),
            output_path=out_path,
            context_length=args.context_length,
            embedding_dim=args.embedding_dim,
            layer_size=args.layer_size,
        )
        sys.exit(0 if ok else 1)

    if args.onnx:
        out_path = Path(args.output) if args.output else None
        ok = bundle_onnx_external_data(
            onnx_path=Path(args.onnx),
            output_path=out_path,
            data_path=Path(args.data) if args.data else None,
        )
        sys.exit(0 if ok else 1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
