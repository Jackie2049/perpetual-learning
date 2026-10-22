#!/usr/bin/env python3
"""Pinned original torch output head vs raw/softcapped logit references on CPU.

Python >=3.10 and torch; pass a local verl Git checkout containing HEAD/BASE.
No checkout, lazy fetch, framework package shell, fake model/config or devices.
Full original torch_functional.py is executed; optional Liger/FlashAttention
must be absent so its normal torch fallback is used. The PR's HF guard tests
are only compiled, not executed. This is not fused softcap implementation.
"""

import argparse
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

sys.dont_write_bytecode = True
import torch

HEAD = "b145ec9b405ca93092c45b5f9e7995fb4604dbd2"
BASE = "dc709afb7a74fd2645640a4e5f2fdf40166b4fbb"
IMPL = "verl/utils/experimental/torch_functional.py"
PRODUCTION = ("verl/models/transformers/monkey_patch.py", IMPL,
              "verl/models/transformers/dense_common.py", "tests/special_sanity/check_license.py")
TEST = "tests/models/test_fused_softcap_on_cpu.py"


def blob(root, revision, path):
    data = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=root,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    print(revision[:8], path, "sha256=" + hashlib.sha256(data).hexdigest())
    compile(ast.parse(data, filename=path), path, "exec")
    return data


def reference(h, w, labels, temperature, cap=None):
    logits = h @ w.t()
    if cap is not None:
        logits = cap * torch.tanh(logits / cap)
    logits = logits / temperature
    log_probs = logits.log_softmax(-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1), entropy


def evaluate(operation, h_values, w_values, labels, objective):
    h, w = h_values.clone().requires_grad_(), w_values.clone().requires_grad_()
    lp, ent = operation(h, w, labels)
    if objective == "both":
        loss = -lp.sum() + 0.17 * ent.sum()
    elif objective == "logprob":
        loss = -lp.sum()
    else:
        loss = ent.sum()
    grads = torch.autograd.grad(loss, (h, w))
    return (lp.detach(), ent.detach(), *(grad.detach() for grad in grads))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verl", type=Path, help="read-only local verl Git checkout")
    args = parser.parse_args()
    # Do not import optional GPU kernels or substitute their interfaces.
    assert importlib.util.find_spec("liger_kernel") is None
    assert importlib.util.find_spec("flash_attn") is None
    sources = {}
    for revision in (BASE, HEAD):
        for path in PRODUCTION:
            sources[revision, path] = blob(args.verl, revision, path)
    blob(args.verl, HEAD, TEST)
    assert sources[BASE, IMPL] == sources[HEAD, IMPL]
    assert sources[BASE, PRODUCTION[2]] == sources[HEAD, PRODUCTION[2]]
    module = ModuleType("pinned_verl_torch_output_head")
    exec(compile(sources[HEAD, IMPL], f"{HEAD}:{IMPL}", "exec"), module.__dict__)
    assert module._LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY is None
    assert module._FLASH_ATTN_CROSS_ENTROPY_AVAILABLE is False
    print("Python", sys.version.split()[0], "torch", torch.__version__, "git", torch.version.git_version)

    h = torch.tensor([[1., 0.], [0., 1.], [1., -0.5]], dtype=torch.float32, device="cpu")
    w = torch.tensor([[0., 0.], [60., 30.], [120., 60.]], dtype=torch.float32, device="cpu")
    labels = torch.tensor([2, 1, 0], dtype=torch.int64, device="cpu")
    count, forward_gap, gradient_gap = 0, 0.0, 0.0
    raw_errors = [0.0] * 4
    tight_matches = 0
    min_semantic_gaps = [float("inf"), float("inf")]
    for layout in ("2d", "3d"):
        h_values = h if layout == "2d" else h.unsqueeze(0)
        ids = labels if layout == "2d" else labels.unsqueeze(0)
        for chunk in (1, 2):
            fused = module.FusedLinearForPPO(chunk_size=chunk, impl_backend="torch")
            for temperature in (0.7, 1., 2.):
                for objective in ("logprob", "entropy", "both"):
                    actual = evaluate(lambda x, v, y: fused(x, v, y, temperature),
                                      h_values, w, ids, objective)
                    raw = evaluate(lambda x, v, y: reference(x, v, y, temperature),
                                   h_values.double(), w.double(), ids, objective)
                    capped = evaluate(lambda x, v, y: reference(x, v, y, temperature, cap=30.),
                                      h_values.double(), w.double(), ids, objective)
                    errors = []
                    for index, (got, expected) in enumerate(zip(actual, raw)):
                        assert got.device.type == "cpu" and torch.isfinite(got).all().item()
                        error = (got.double() - expected).abs().max().item()
                        errors.append(error)
                        raw_errors[index] = max(raw_errors[index], error)
                    tight_matches += all(error <= 2e-5 for error in errors)
                    # Same real inputs/weights, but omitting the transform changes both quantities and VJPs.
                    fgap = max((a - b).abs().max().item() for a, b in zip(actual[:2], capped[:2]))
                    ggap = max((a - b).abs().max().item() for a, b in zip(actual[2:], capped[2:]))
                    assert fgap > 1e-3 and ggap > 1e-3, (layout, chunk, temperature, objective)
                    assert fgap > 100 * max(errors[:2]) and ggap > 100 * max(errors[2:])
                    min_semantic_gaps = [min(min_semantic_gaps[0], fgap), min(min_semantic_gaps[1], ggap)]
                    forward_gap, gradient_gap = max(forward_gap, fgap), max(gradient_gap, ggap)
                    count += 1
    original_lp, original_ent = module.FusedLinearForPPO(impl_backend="torch")(h[:1], w, labels[:1], 1.)
    raw_lp, raw_ent = reference(h[:1].double(), w.double(), labels[:1], 1.)
    cap_lp, cap_ent = reference(h[:1].double(), w.double(), labels[:1], 1., cap=30.)
    print("original torch logp/entropy", original_lp.item(), original_ent.item())
    print("z=[0,60,120], label=2, T=1: raw logp/entropy", raw_lp.item(), raw_ent.item())
    print("softcap=30 logp/entropy", cap_lp.item(), cap_ent.item())
    print("raw FP64 errors (logp, entropy, dh, dw)", raw_errors)
    print("raw controls within absolute 2e-5:", tight_matches, "/", count, "; exceed:", count - tight_matches)
    assert count == 36
    print("PASS", count, "semantic separation controls, min capped gaps", min_semantic_gaps)
    print("max capped gaps", forward_gap, gradient_gap, "; all >100x corresponding raw FP64 error")
    print("9 full-module syntax checks passed; original 24 HF guard tests NOT run")


if __name__ == "__main__":
    main()
