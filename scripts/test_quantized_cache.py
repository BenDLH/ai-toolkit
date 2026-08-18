"""
Round-trip test for the pre-quantized cache (toolkit/util/quantized_cache.py).

Builds a tiny SingleStreamDiT, quantizes it two ways — directly in memory (the
same convert_linear_to_ostris call quantize-at-load uses) and through a
build_quantized_cache -> apply_quantized_cache round trip — and asserts the
results are bitwise identical: backend buffers, passthrough tensors, module
classes, and per-module forward outputs.

The krea2 modules are loaded through the converter's standalone helpers (file
loading + AST constant extraction), so the test both avoids the extensions
package import chain (which needs dependencies like k_diffusion) and covers
the loading logic the converter actually ships with.

Run directly:  python scripts/test_quantized_cache.py
"""

import os
import sys
import tempfile
from fnmatch import fnmatch

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from safetensors.torch import save_file  # noqa: E402

from toolkit.util.ostris_quant import (  # noqa: E402
    OstrisLinear,
    convert_linear_to_ostris,
    get_ostris_quantizer,
)
from toolkit.util.quantized_cache import (  # noqa: E402
    apply_quantized_cache,
    build_quantized_cache,
    read_cache_info,
)

_converter_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "make_quantized_cache.py")
# import the converter as a module to reuse its standalone loading helpers
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("make_quantized_cache", _converter_path)
CONVERTER = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CONVERTER)

MMDIT = CONVERTER.load_module_from_file(
    "krea2_mmdit", os.path.join(CONVERTER.KREA2_DIR, "src", "mmdit.py")
)
_real_config, EXCLUDE = CONVERTER.read_krea2_source_constants()
assert _real_config.get("features"), "converter returned an empty mmdit config"

# smallest config that keeps every quantized in_features divisible by the
# uintX group size (64) and the rope axes even: headdim 64 -> axes [16, 24, 24]
TINY_CONFIG = dict(
    features=128,
    tdim=64,
    txtdim=128,
    heads=2,
    kvheads=2,
    multiplier=4,
    layers=2,
    patch=2,
    channels=16,
    txtlayers=2,
    txtheads=2,
    txtkvheads=2,
)


def tiny_model(device="cpu", dtype=torch.bfloat16):
    torch.manual_seed(42)
    model = MMDIT.SingleStreamDiT(MMDIT.SingleMMDiTConfig(**TINY_CONFIG))
    return model.to(device, dtype)


def tiny_meta_model():
    with torch.device("meta"):
        return MMDIT.SingleStreamDiT(MMDIT.SingleMMDiTConfig(**TINY_CONFIG))


def quantize_reference(model, qtype, device):
    """Quantize in place exactly the way quantize-at-load does."""
    quantizer = get_ostris_quantizer(qtype)
    names = []
    for name, module in model.named_modules():
        if type(module) is not torch.nn.Linear:
            continue
        if any(fnmatch(name, p) for p in EXCLUDE):
            continue
        module.to(device)
        if convert_linear_to_ostris(module, quantizer):
            names.append(name)
        module.to("cpu")
    return names


def run_round_trip(qtype, device):
    workdir = tempfile.mkdtemp(prefix="quantized_cache_test_")
    checkpoint = os.path.join(workdir, "checkpoint.safetensors")
    cache = os.path.join(workdir, "cache.safetensors")

    source = tiny_model()
    save_file(
        {k: v.contiguous() for k, v in source.state_dict().items()}, checkpoint
    )

    # reference: quantize the source weights in memory
    reference = tiny_model()
    reference.load_state_dict(source.state_dict())
    ref_names = quantize_reference(reference, qtype, device)
    assert ref_names, "reference quantized nothing; config too small?"

    # cache: converter -> file -> restore onto a meta model
    n_quantized, n_extra = build_quantized_cache(
        tiny_meta_model(),
        checkpoint_path=checkpoint,
        output_path=cache,
        qtype=qtype,
        exclude_patterns=EXCLUDE,
        device=torch.device(device),
        info={"arch": "krea2-tiny"},
    )
    assert n_quantized == len(ref_names), (n_quantized, len(ref_names))
    info = read_cache_info(cache)
    assert info["qtype"] == qtype and info["arch"] == "krea2-tiny", info

    restored = tiny_meta_model()
    n_restored = apply_quantized_cache(restored, cache, cast_dtype=torch.bfloat16)
    assert n_restored == n_quantized, (n_restored, n_quantized)

    # 1. same modules quantized, excluded modules left as plain linears
    ref_modules = dict(reference.named_modules())
    for name, module in restored.named_modules():
        ref = ref_modules[name]
        if isinstance(module, OstrisLinear) or isinstance(ref, OstrisLinear):
            assert type(module) is type(ref), f"{name}: {type(module)} vs {type(ref)}"
            assert not any(
                fnmatch(name, p) for p in EXCLUDE
            ), f"excluded module {name} was quantized"

    # 2. backend buffers bitwise identical
    for name in ref_names:
        ref = ref_modules[name]
        mod = restored.get_submodule(name)
        assert set(ref._buffers) == set(mod._buffers), name
        for buf_name, ref_buf in ref._buffers.items():
            buf = mod._buffers[buf_name]
            assert ref_buf.dtype == buf.dtype, (name, buf_name)
            assert torch.equal(ref_buf, buf), f"{name}.{buf_name} differs"

    # 3. every state dict tensor identical (passthrough tensors verbatim,
    # quantized weights via their dequantized emission)
    ref_sd = reference.state_dict()
    got_sd = restored.state_dict()
    assert set(ref_sd) == set(got_sd)
    for key in ref_sd:
        assert ref_sd[key].dtype == got_sd[key].dtype, key
        assert torch.equal(ref_sd[key], got_sd[key]), f"{key} differs"

    # 4. per-module forwards agree bitwise
    torch.manual_seed(7)
    for name in ref_names:
        ref = ref_modules[name]
        mod = restored.get_submodule(name)
        x = torch.randn(2, ref.in_features, dtype=torch.bfloat16)
        assert torch.equal(ref(x), mod(x)), f"{name} forward differs"

    print(f"PASS round trip qtype={qtype} device={device} "
          f"({n_quantized} quantized, {n_extra} passthrough)")


def run_rejects_qtype_mismatch(qtype="uint6", claimed="uint4"):
    """A cache restores at its own precision, so a config claiming a different
    qtype must fail rather than silently train at the cache's."""
    workdir = tempfile.mkdtemp(prefix="quantized_cache_mismatch_")
    checkpoint = os.path.join(workdir, "checkpoint.safetensors")
    cache = os.path.join(workdir, "cache.safetensors")
    save_file(
        {k: v.contiguous() for k, v in tiny_model().state_dict().items()}, checkpoint
    )
    build_quantized_cache(
        tiny_meta_model(), checkpoint_path=checkpoint, output_path=cache,
        qtype=qtype, exclude_patterns=EXCLUDE, device=torch.device("cpu"),
    )
    try:
        apply_quantized_cache(tiny_meta_model(), cache, expected_qtype=claimed)
    except ValueError as e:
        assert "precision mismatch" in str(e), e
        print(f"PASS mismatch rejected ({claimed} config vs {qtype} cache)")
        return
    raise AssertionError("mismatched qtype should have been rejected")


def run_rejects_non_ostris_qtype():
    try:
        build_quantized_cache(
            tiny_meta_model(),
            checkpoint_path="unused",
            output_path="unused",
            qtype="qfloat8",
        )
    except ValueError as e:
        assert "ostris backend" in str(e)
        print("PASS non-ostris qtype rejected")
        return
    raise AssertionError("qfloat8 should have been rejected")


if __name__ == "__main__":
    run_rejects_non_ostris_qtype()
    run_rejects_qtype_mismatch()
    run_round_trip("uint4", "cpu")
    run_round_trip("uint6", "cpu")
    if torch.cuda.is_available():
        run_round_trip("uint6", "cuda")
    print("all tests passed")
