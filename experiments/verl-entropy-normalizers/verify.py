"""CPU math only: pinned original entropy helpers, not CANN operators or devices."""

import argparse
import ast
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess

import torch

BASE = "8df5baeed0f5ecdd7f870d83bc3d79194816e2a9"
HEAD = "c261bd4ab2efa740805d5b7c956996b175eebc38"
SOURCE = "verl/utils/kernel/npu/cann_linear_ce.py"
TESTS = "tests/utils/test_cann_linear_ce_on_cpu.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    hashes, sources = {}, {}
    for ref, path in [(BASE, SOURCE), (HEAD, SOURCE), (HEAD, TESTS)]:
        raw = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=args.repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        compile(raw, path, "exec")
        sources[ref, path] = raw
        hashes[f"{ref}:{path}"] = hashlib.sha256(raw).hexdigest()
    modules = {}
    for ref in (BASE, HEAD):
        namespace = {"__name__": "cpu_math_only"}
        # Complete source imports torch, not torch_npu; its lazy resolver is never called.
        exec(compile(sources[ref, SOURCE], SOURCE, "exec"), namespace)
        modules[ref] = namespace

    cases = {
        "test_entropy_stats_fp64_finite_difference": list(itertools.product([0.5, 1.0, 1.5], [0.0, 10000.0])),
        "test_entropy_stats_preserve_fp32_under_autocast": [()],
        "test_entropy_stats_extreme_logits": [()],
        "test_chunked_ce_entropy_backward_matches_torch": list(itertools.product([1.0, 1.5], [32, 128, 2048], [7, 256])),
    }
    original_bodies = {}
    for node in ast.parse(sources[HEAD, TESTS]).body:
        if isinstance(node, ast.FunctionDef) and node.name in cases:
            function = copy.deepcopy(node)
            function.decorator_list = []  # Explicit cases above replace pytest parametrization only.
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), TESTS, "exec"),
                 modules[HEAD])
            for parameters in cases[node.name]:
                modules[HEAD][node.name](*parameters)
            original_bodies[node.name] = {"passed": len(cases[node.name]), "parameters": cases[node.name]}
    assert sum(value["passed"] for value in original_bodies.values()) == 20

    # Own negative control: deliberately BF16-round logits before deriving supplied CE stats.
    # This models a normalization mismatch; it is not a claim about any actual CANN build.
    mismatch = []
    for offset in (0.0, 32.0, 128.0):
        torch.manual_seed(42)
        h = (torch.randn(5, 8) * 0.2).bfloat16()
        w = (torch.randn(11, 8) * 0.2).bfloat16()
        h[:, 0] = 1
        w[:, 0] = offset
        labels = torch.tensor([0, 10, 3, 1, 7])
        g = torch.linspace(-0.2, 0.3, 5)
        z = h.float() @ w.float().T
        rounded = z.bfloat16().float()
        m = rounded.max(-1).values
        s = (rounded - m[:, None]).exp().sum(-1)
        reused_mass = ((z - m[:, None]).exp() / s[:, None]).sum(-1)
        hr, wr = h.double().requires_grad_(), w.double().requires_grad_()
        logp = (hr @ wr.T).log_softmax(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        expected = torch.autograd.grad(entropy, (hr, wr), g.double())
        entry = {"offset": offset, "reused_probability_mass_max_error": (reused_mass - 1).abs().max().item()}
        for ref, label in ((BASE, "base"), (HEAD, "head")):
            with torch.no_grad():
                actual = modules[ref]["_chunked_ce_entropy_backward"](
                    torch.zeros(5), g, h, w, labels, m, s, 1.0, "none", 0, chunk_size=4)
            entry[label] = {
                "hidden_grad_max_error": (actual[0].double() - expected[0]).abs().max().item(),
                "weight_grad_max_error": (actual[1].double() - expected[1]).abs().max().item(),
                "common_shift_hidden_grad_max": actual[0][:, 0].abs().max().item(),
            }
            if label == "head":
                for a, b in zip(actual, expected, strict=True):
                    torch.testing.assert_close(a.double(), b, atol=1e-4, rtol=1e-2)
        mismatch.append(entry)
    assert mismatch[-1]["base"]["hidden_grad_max_error"] > 1
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "torch_git": torch.version.git_version, "threads": torch.get_num_threads(),
                      "base": BASE, "head": HEAD, "source_sha256": hashes,
                      "original_head_test_bodies": original_bodies, "own_rounded_stats_controls": mismatch,
                      "scope": "20 unchanged original CPU math parameter bodies with decorators replaced by "
                      "explicit calls, not native pytest collection or the complete CPU suite. Three own "
                      "entropy-only base/head helper controls with deliberately rounded supplied statistics. "
                      "Real CPU matmul, autograd and finite differences. No CannLinearCrossEntropy.apply, "
                      "CANN mocks/operators, torch_npu, NPU/GPU, TP/HCCL, actor, training, performance or memory measurement."},
                     indent=2))


if __name__ == "__main__":
    main()
