"""
Distribution test for the 'logit_normal_shift' timestep type on Krea 2.

Samples the scheduler the way training does, with Krea 2's own scheduler config
and patch size, and checks the draws against the closed form: a standard
logit-normal t, then the exponential shift t' = e^mu t / (1 + (e^mu - 1) t),
with mu linear in image tokens. Also checks that the Krea 2 config reports its
patch size, without which the token count (and so the shift) is 4x off.

The krea2 sources are read by file path and AST, so the test avoids the
extensions package import chain (which needs the full training environment).

Run directly:  python scripts/test_timestep_distribution.py
"""

import ast
import importlib.util
import math
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from toolkit.samplers.custom_flowmatch_sampler import (  # noqa: E402
    CustomFlowMatchEulerDiscreteScheduler,
    calculate_shift,
)

KREA2_DIR = os.path.join(ROOT, "extensions_built_in", "diffusion_models", "krea2")
VAE_SCALE = 8
LATENT_CHANNELS = 16
DRAWS_PER_CALL = 1000
CALLS = 200


def read_krea2_constants():
    with open(os.path.join(KREA2_DIR, "krea2.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())

    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "scheduler_config":
                found["scheduler"] = ast.literal_eval(node.value)
            if isinstance(target, ast.Name) and target.id == "KREA2_MMDIT_CONFIG":
                found["mmdit"] = {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords}
    assert "scheduler" in found, "scheduler_config not found in krea2.py"
    assert "mmdit" in found, "KREA2_MMDIT_CONFIG not found in krea2.py"
    return found["scheduler"], found["mmdit"]


def load_mmdit_config_class():
    spec = importlib.util.spec_from_file_location(
        "krea2_mmdit", os.path.join(KREA2_DIR, "src", "mmdit.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SingleMMDiTConfig


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def share_at_or_above(threshold, mu):
    """P(t' >= threshold) for a standard logit-normal t under shift mu."""
    shift = math.exp(mu)
    t = threshold / (shift - (shift - 1) * threshold)
    return 1 - normal_cdf(math.log(t / (1 - t)))


def sample(scheduler, resolution, patch_size):
    side = resolution // VAE_SCALE
    latents = torch.zeros((1, LATENT_CHANNELS, side, side))
    draws = []
    for _ in range(CALLS):
        timesteps = scheduler.set_train_timesteps(
            DRAWS_PER_CALL,
            device="cpu",
            timestep_type="logit_normal_shift",
            latents=latents,
            patch_size=patch_size,
        )
        assert torch.all(timesteps[:-1] >= timesteps[1:]), "timesteps are not descending"
        assert torch.allclose(scheduler.sigmas[:-1] * 1000, timesteps), "sigmas do not match timesteps"
        assert scheduler.sigmas[-1].item() == 0, "sigmas do not end at zero"
        draws.append(timesteps / 1000)
    return torch.cat(draws)


def run_patch_size():
    _, mmdit = read_krea2_constants()
    config = load_mmdit_config_class()(**mmdit)
    assert config.patch_size == mmdit["patch"] == 2, f"patch_size {config.patch_size}"
    print("PASS Krea 2 config reports patch_size 2")
    return config.patch_size


def run_distribution(resolution, patch_size, expected_noisy_share):
    scheduler_config, _ = read_krea2_constants()
    scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    tokens = (resolution // VAE_SCALE // patch_size) ** 2
    mu = calculate_shift(
        tokens,
        scheduler_config["base_image_seq_len"],
        scheduler_config["max_image_seq_len"],
        scheduler_config["base_shift"],
        scheduler_config["max_shift"],
    )
    t = sample(scheduler, resolution, patch_size)

    for threshold in (0.3, 0.5, 0.9):
        expected = share_at_or_above(threshold, mu)
        observed = (t >= threshold).float().mean().item()
        assert abs(observed - expected) < 0.005, (
            f"{resolution}px t>={threshold}: observed {observed:.4f}, expected {expected:.4f}"
        )

    noisy_share = share_at_or_above(0.9, mu)
    assert abs(noisy_share - expected_noisy_share) < 0.002, (
        f"{resolution}px closed-form t>=0.9 share {noisy_share:.4f}, expected {expected_noisy_share}"
    )
    print(f"PASS {resolution}px: {tokens} tokens, mu {mu:.3f}, "
          f"t>=0.9 {(t >= 0.9).float().mean().item():.3f}, median {t.median().item():.3f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    patch_size = run_patch_size()
    run_distribution(512, patch_size, expected_noisy_share=0.053)
    run_distribution(768, patch_size, expected_noisy_share=0.069)
    print("all tests passed")
