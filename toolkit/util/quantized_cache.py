"""
Self-contained pre-quantized model cache.

A cache file holds a whole single-file checkpoint with its eligible linears
already quantized by an ostris backend (toolkit/util/ostris_quant.py): the
backend buffers of every quantized linear, plus every remaining tensor
(excluded linears, norms, embedders, biases) verbatim in the checkpoint dtype.
Loading one onto a meta-built model skips both the full-precision weight read
and the quantize-at-load pass, which together dominate startup for short
training runs; the file is also several times smaller than the bf16
checkpoint, so it can replace it on capacity-constrained volumes.

build_quantized_cache streams the checkpoint one tensor at a time (peak memory
is one layer plus the accumulated quantized buffers, never the full model) and
writes the file via save_quantized_layers, so the quantized-layer layout and
metadata stay shared with every other consumer of that format.

Only ostris backends (uintX, convrot, orbit) can round-trip: quanto and
torchao qtypes store their state in tensor subclasses this format does not
serialize. Quantization runs through convert_linear_to_ostris, so results are
bit-identical to quantizing the same checkpoint at load time.
"""

import json
import os
from fnmatch import fnmatch
from typing import Callable, Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from toolkit.util.ostris_quant import (
    QUANT_LAYERS_METADATA_KEY,
    OstrisLinear,
    convert_linear_to_ostris,
    get_ostris_quantizer,
    load_quantized_layers,
    save_quantized_layers,
)

# provenance metadata (arch, qtype, source file, exclude patterns); informational
CACHE_INFO_METADATA_KEY = "aitk_quantized_cache"


def _matches_any(name: str, patterns: List[str]) -> bool:
    return any(fnmatch(name, p) for p in patterns)


@torch.no_grad()
def build_quantized_cache(
    model: torch.nn.Module,
    checkpoint_path: str,
    output_path: str,
    qtype: str,
    exclude_patterns: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    info: Optional[Dict[str, str]] = None,
    status_fn: Callable[[str], None] = lambda msg: None,
) -> Tuple[int, int]:
    """Quantize a single-file checkpoint into a self-contained cache file.

    ``model`` must be a meta-built instance whose state dict matches the
    checkpoint keys exactly (the same parity the strict load enforces on the
    normal path). Its linears are converted in place, so pass a throwaway
    instance. Returns (quantized modules, passthrough tensors).
    """
    exclude_patterns = exclude_patterns or []
    quantizer = get_ostris_quantizer(qtype)
    if quantizer is None:
        raise ValueError(
            f"qtype '{qtype}' is not an ostris backend; only ostris backends "
            "(uintX, convrot, orbit) can be serialized into a quantized cache."
        )
    device = device or torch.device("cpu")

    # key parity with the checkpoint, checked before any conversion because
    # OstrisLinear materializes full weights in state_dict()
    expected_keys = set(model.state_dict().keys())

    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        file_keys = set(f.keys())
        if file_keys != expected_keys:
            missing = sorted(expected_keys - file_keys)[:5]
            unexpected = sorted(file_keys - expected_keys)[:5]
            raise ValueError(
                f"Checkpoint does not match the model: missing {missing}, "
                f"unexpected {unexpected}"
            )

        modules: Dict[str, OstrisLinear] = {}
        consumed = set()
        for name, module in model.named_modules():
            if type(module) is not torch.nn.Linear:
                continue
            if _matches_any(name, exclude_patterns):
                continue
            # shape gates (e.g. uintX group divisibility) work on meta params
            if not quantizer.can_quantize(module):
                continue
            weight = f.get_tensor(f"{name}.weight").to(device)
            module._parameters["weight"] = torch.nn.Parameter(
                weight, requires_grad=False
            )
            consumed.add(f"{name}.weight")
            if module.bias is not None:
                bias = f.get_tensor(f"{name}.bias").to(device)
                module._parameters["bias"] = torch.nn.Parameter(
                    bias, requires_grad=False
                )
                consumed.add(f"{name}.bias")
            if not convert_linear_to_ostris(module, quantizer):
                raise RuntimeError(f"Failed to quantize eligible module '{name}'")
            module.to("cpu")
            modules[name] = module
            if len(modules) % 50 == 0:
                status_fn(f"quantized {len(modules)} layers")

        # everything else (excluded/refused linears, norms, embedders,
        # modulation params) passes through in the checkpoint dtype
        extras = {
            key: f.get_tensor(key) for key in sorted(file_keys - consumed)
        }

    status_fn(
        f"quantized {len(modules)} layers, passing {len(extras)} tensors through"
    )
    meta = {
        CACHE_INFO_METADATA_KEY: json.dumps(
            {
                "qtype": qtype,
                "source_file": os.path.basename(checkpoint_path),
                "exclude": exclude_patterns,
                **(info or {}),
            }
        )
    }
    save_quantized_layers(modules, output_path, metadata=meta, extra_tensors=extras)
    return len(modules), len(extras)


@torch.no_grad()
def apply_quantized_cache(
    model: torch.nn.Module,
    cache_path: str,
    cast_dtype: Optional[torch.dtype] = None,
    expected_qtype: Optional[str] = None,
) -> int:
    """Restore a cache file onto a meta-built model. Returns the number of
    quantized layers restored.

    Passthrough float tensors are cast to ``cast_dtype`` (matching what the
    normal load path does to the checkpoint); quantized backend buffers are
    installed exactly as saved. Fails loudly if the cache does not cover the
    model completely.

    A cache carries the qtype it was built with, and restoring uses that —
    the config's qtype is not consulted. Pass ``expected_qtype`` to assert the
    two agree, so switching precision by editing one of them and not the other
    fails loudly instead of training at a precision nobody asked for.
    """
    with safe_open(cache_path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
    if QUANT_LAYERS_METADATA_KEY not in meta:
        raise ValueError(f"{cache_path} is not a quantized cache file")
    quant_map = json.loads(meta[QUANT_LAYERS_METADATA_KEY])["modules"]

    if expected_qtype is not None:
        cache_qtypes = {entry["qtype"] for entry in quant_map.values()}
        if cache_qtypes != {expected_qtype}:
            raise ValueError(
                f"quantized cache precision mismatch: config asks for "
                f"qtype '{expected_qtype}' but {os.path.basename(cache_path)} "
                f"holds {sorted(cache_qtypes)}. Point quantized_cache_path at "
                f"the matching cache, or set qtype to what the cache holds."
            )

    state_dict = load_file(cache_path)
    buffer_keys = {
        f"{name}.{buf}" for name, entry in quant_map.items() for buf in entry["buffers"]
    }
    plain = {
        k: (v.to(cast_dtype) if cast_dtype is not None and v.is_floating_point() else v)
        for k, v in state_dict.items()
        if k not in buffer_keys
    }
    missing, unexpected = model.load_state_dict(plain, strict=False, assign=True)
    # the only keys the passthrough load may leave behind are the weights the
    # quantized buffers replace
    expected_missing = {f"{name}.weight" for name in quant_map}
    if unexpected or set(missing) != expected_missing:
        raise ValueError(
            f"{cache_path} does not match the model: unexpected keys "
            f"{sorted(unexpected)[:5]}, missing {sorted(set(missing) - expected_missing)[:5]}"
        )

    restored = load_quantized_layers(model, cache_path, state_dict=state_dict)

    leftover_meta = [
        n
        for n, t in list(model.named_parameters()) + list(model.named_buffers())
        if t.is_meta
    ]
    if leftover_meta:
        raise ValueError(
            f"{cache_path} left {len(leftover_meta)} tensors unmaterialized: "
            f"{leftover_meta[:5]}"
        )
    return restored


def read_cache_info(cache_path: str) -> Optional[dict]:
    """Provenance metadata stamped by build_quantized_cache, if present."""
    with safe_open(cache_path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
    raw = meta.get(CACHE_INFO_METADATA_KEY)
    return json.loads(raw) if raw else None
