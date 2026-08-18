"""
Build a self-contained pre-quantized cache from a single-file checkpoint.

The output file replaces the original checkpoint for training: point
model.quantized_cache_path at it and the trainer restores the quantized
transformer directly — no full precision read, no quantize-at-load pass. See
toolkit/util/quantized_cache.py for the format.

Example (Krea 2, ~26GB bf16 in, ~7GB uint4 out):

    python scripts/make_quantized_cache.py \
        --arch krea2 \
        --checkpoint /path/to/krea2_raw_bf16.safetensors \
        --output /path/to/krea2_raw_uint4_cache.safetensors \
        --qtype uint4

Runs with just torch, safetensors and einops: the model class is loaded from
its file and the model constants (config, quantization exclude patterns) are
read out of the extension source by AST, so neither the extensions package nor
the full training environment has to be importable. Quantization streams one
layer at a time (peak VRAM well under 1GB), but the quantized result
accumulates in host RAM and safetensors saving copies it once more — budget
roughly 2x the output size in free RAM.
"""

import argparse
import ast
import importlib.util
import json
import os
import sys
import time

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from toolkit.util.quantized_cache import build_quantized_cache  # noqa: E402

KREA2_DIR = os.path.join(ROOT, "extensions_built_in", "diffusion_models", "krea2")


def load_module_from_file(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_krea2_source_constants():
    """Extract KREA2_MMDIT_CONFIG and the quantization exclude patterns from
    krea2.py without importing it (its import chain needs the full training
    environment; this script runs on slim conversion machines too)."""
    with open(os.path.join(KREA2_DIR, "krea2.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())

    config = None
    exclude = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "KREA2_MMDIT_CONFIG"
            for t in node.targets
        ):
            call = node.value
            assert (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "dict"
            ), "KREA2_MMDIT_CONFIG is no longer a dict(...) literal"
            config = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "get_quantization_exclude_modules"
        ):
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.List):
                    exclude = [ast.literal_eval(elt) for elt in stmt.value.elts]
    assert config is not None, "KREA2_MMDIT_CONFIG not found in krea2.py"
    assert exclude is not None, (
        "get_quantization_exclude_modules' pattern list not found in krea2.py"
    )
    return config, exclude


def build_krea2(mmdit_config_overrides: dict):
    mmdit = load_module_from_file(
        "krea2_mmdit", os.path.join(KREA2_DIR, "src", "mmdit.py")
    )
    mmdit_kwargs, exclude = read_krea2_source_constants()
    mmdit_kwargs.update(mmdit_config_overrides)
    with torch.device("meta"):
        model = mmdit.SingleStreamDiT(mmdit.SingleMMDiTConfig(**mmdit_kwargs))
    return model, exclude


ARCHS = {
    "krea2": build_krea2,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--arch", required=True, choices=sorted(ARCHS.keys()))
    parser.add_argument(
        "--checkpoint", required=True, help="single-file checkpoint (.safetensors)"
    )
    parser.add_argument("--output", required=True, help="cache file to write")
    parser.add_argument(
        "--qtype",
        default="uint4",
        help="ostris backend qtype (e.g. uint4, uint6); default uint4",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to run quantization math on (default: cuda if available)",
    )
    parser.add_argument(
        "--model-config",
        default="{}",
        help="JSON overrides for the model config (matches model_kwargs.mmdit_config)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        parser.error(f"checkpoint not found: {args.checkpoint}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    model, exclude = ARCHS[args.arch](json.loads(args.model_config))
    print(f"arch={args.arch} qtype={args.qtype} device={args.device}")
    print(f"exclude patterns: {exclude}")

    start = time.time()
    n_quantized, n_extra = build_quantized_cache(
        model,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        qtype=args.qtype,
        exclude_patterns=exclude,
        device=torch.device(args.device),
        info={"arch": args.arch},
        status_fn=lambda msg: print(f"  {msg}", flush=True),
    )
    elapsed = time.time() - start

    in_size = os.path.getsize(args.checkpoint) / 1024**3
    out_size = os.path.getsize(args.output) / 1024**3
    print(
        f"done in {elapsed:.0f}s: {n_quantized} layers quantized, "
        f"{n_extra} tensors passed through"
    )
    print(f"{in_size:.1f}GB -> {out_size:.1f}GB ({args.output})")


if __name__ == "__main__":
    main()
