"""Original scale resolver plus original Torch custom autograd, own CPU inputs.

No model adapter, Transformers model, Triton/Liger kernel, or original test bodies.
"""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace
from typing import Optional

import torch

HEAD = "1ecd1a9e7e2b9b772cf1d03b3eb5a090e358fdef"
BASE = "2ab35292039a51175a4b3d3b3b2d76e951a0d3e4"
DENSE = "verl/models/transformers/dense_common.py"
MATH = "verl/utils/experimental/torch_functional.py"
READING = ("tests/models/test_fused_kernels_ulysses_sp_on_cpu.py",
           "verl/models/transformers/monkey_patch.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources, hashes = {}, {}
    for ref, paths in ((BASE, (DENSE, MATH)), (HEAD, (DENSE, MATH, *READING))):
        for path in paths:
            raw = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=args.repo,
                                          env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
            compile(ast.parse(raw), path, "exec")
            sources[ref, path] = raw
            hashes[f"{ref}:{path}"] = hashlib.sha256(raw).hexdigest()
    assert sources[BASE, MATH] == sources[HEAD, MATH]
    ns = {"torch": torch, "Optional": Optional, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE": False}
    for path, names in ((DENSE, {"_GRANITE_MODEL_TYPES", "_COHERE_MODEL_TYPES", "_require_nonzero_scale",
                                "_logit_scale_factor"}),
                        (MATH, {"_fused_linear_for_ppo_fwd", "_fused_linear_for_ppo_bwd", "FusedLinearForPPOFunction"})):
        nodes = [n for n in ast.parse(sources[HEAD, path]).body
                 if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
                 or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)

    def model(model_type, location, attribute, value):
        obj = SimpleNamespace(config=SimpleNamespace(model_type=model_type), lm_head=SimpleNamespace(bias=None),
                              model=SimpleNamespace())
        if model_type == "cohere":
            obj.config.logit_scale = 0.25
        setattr(obj if location == "self" else getattr(obj, location), attribute, value)
        return obj

    g = torch.Generator().manual_seed(37)
    hidden = torch.randn(5, 4, generator=g)
    weight = torch.randn(7, 4, generator=g)
    labels = torch.tensor([0, 1, 3, 6, 2])
    dlp = torch.randn(5, generator=g)
    dent = torch.randn(5, generator=g) * 0.3
    temperature = 0.7

    def evaluate(scale, folded=None):
        h, w = hidden.clone().requires_grad_(), weight.clone().requires_grad_()
        if folded is None:
            logits = ((h @ w.t()) * scale) / temperature
            logprobs = logits.log_softmax(-1)
            lp = logprobs.gather(-1, labels[:, None]).squeeze(-1)
            entropy = -(logprobs.exp() * logprobs).sum(-1)
        else:
            lp, entropy = ns["FusedLinearForPPOFunction"].apply(h, w, labels, folded, 2)
        dh, dw = torch.autograd.grad((lp * dlp + entropy * dent).sum(), (h, w))
        return lp, entropy, dh, dw

    cases = []
    configurations = [("granite", "config", "logits_scaling", 8.0, 0.125),
                      ("cohere", "self", "logit_scale", 0.0625, 0.0625),
                      ("cohere", "config", "logit_scale", 0.0625, 0.0625),
                      ("falcon_h1", "model", "lm_head_multiplier", 0.0390625, 0.0390625),
                      ("hyperclovax", "config", "logits_scaling", 0.125, 0.125),
                      ("unknown_family", "config", "logits_scaling", 8.0, 1.0)]
    for family, location, attribute, value, expected in configurations:
        scale = ns["_logit_scale_factor"](model(family, location, attribute, value))
        assert scale == expected
        ref = evaluate(scale)
        output = evaluate(scale, temperature / scale)
        old = evaluate(scale, temperature)
        errors = {name: float((a - b).abs().max().detach())
                  for name, a, b in zip(("logp", "entropy", "hidden_grad", "weight_grad"), output, ref)}
        assert max(errors.values()) < 3e-6, errors
        old_error = float((old[0] - ref[0]).abs().max().detach())
        assert (old_error > 0.1) if scale != 1 else old_error < 3e-6
        cases.append({"family": family, "scale_location": location, "scale": scale,
                      "effective_temperature": temperature / scale, "new_max_abs_errors": errors,
                      "omitted_scale_max_logp_error": old_error})

    guards = {}
    for key, obj in (("zero_scale", model("granite", "config", "logits_scaling", 0.0)),
                     ("zero_bias_tensor", model("unknown_family", "config", "logits_scaling", 8.0))):
        if key == "zero_bias_tensor":
            obj.lm_head.bias = torch.zeros(7)
        try:
            ns["_logit_scale_factor"](obj)
        except ValueError as error:
            guards[key] = str(error)
        else:
            raise AssertionError(f"Expected rejection: {key}")
    negative = ns["_logit_scale_factor"](model("cohere", "config", "logit_scale", -0.5))
    nonfinite = ns["_logit_scale_factor"](model("cohere", "config", "logit_scale", float("nan")))
    assert negative == -0.5 and math.isnan(nonfinite)
    guards["negative_scale_returned"] = negative
    guards["nan_scale_returned"] = "nan"
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version,
                      "head": HEAD, "base": BASE, "source_sha256": hashes,
                      "math_module_identical_to_base": True,
                      "own_numeric_cases": cases, "own_resolver_guard_cases": guards,
                      "inputs": {"seed": 37, "hidden_shape": [5, 4], "weight_shape": [7, 4],
                                 "dtype": "float32", "temperature": temperature, "chunk_size": 2,
                                 "objective": "weighted selected logp plus weighted entropy; own fixed upstream gradients"},
                      "scope": "Six complete inputs compile. Exact original resolver/constants and original Torch "
                      "fwd/bwd/custom autograd class extracted by AST; FlashAttention path explicitly unavailable. "
                      "Six own positive-scale/default numerical cases compare against explicit dense scaling with "
                      "autograd, plus four own resolver-only guard probes. Old math is original unchanged custom "
                      "autograd at unadjusted temperature, not execution of old adapter. No original test bodies, "
                      "dense_common adapter forward or real HF model, monkey patch application, VLM wrappers, "
                      "Triton/Liger, model checkpoint, SP/TP, rollout, GPU/NPU, training or performance."}, indent=2,
                     allow_nan=False))


if __name__ == "__main__":
    main()
