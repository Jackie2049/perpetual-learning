"""Run #8386's 20 new CPU cases using isolated original functions, not training.

Requires torch and pytest. No framework imports, device discovery or network.
python -B verify.py --repo /path/to/verl --output result.json
"""

import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch

BASE = "0bd668c1aa7b63c5f174f7dcb1c5de12e0c74a0b"
HEAD = "c854acbbec2248950d562dde49e4d831641daf1f"
CORE = "verl/trainer/ppo/core_algos.py"
CORR = "verl/trainer/ppo/rollout_corr_helper.py"
FUNC = "verl/utils/torch_functional.py"
TEST = "tests/trainer/ppo/test_core_algos_on_cpu.py"
TEST_NAMES = {
    "test_k3_penalty_small_log_ratios",
    "test_k3_rejection_agrees_with_reference_near_zero",
    "test_k3_rejection_accumulates_small_divergences_on_long_sequences",
    "test_offpolicy_k3_metric_preserves_small_divergences",
    "test_k3_penalty_preserves_gradients_and_clipping",
    "test_k3_penalty_preserves_values_away_from_zero",
}
CORR_NAMES = {
    "_parse_rollout_rs_thresholds", "compute_rollout_rejection_mask",
    "compute_rs_metrics", "compute_offpolicy_metrics",
    "SAFETY_BOUND", "SUPPORTED_ROLLOUT_RS_OPTIONS", "TOKEN_LEVEL_ROLLOUT_RS_OPTIONS",
}


def select(source, names, namespace, filename):
    """Keep complete named original definitions/constants; never rewrite bodies."""
    tree = ast.parse(source, filename=filename)
    compile(tree, filename, "exec")  # Syntax only; don't execute the full module.
    nodes = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
        elif isinstance(node, ast.AnnAssign):
            name = getattr(node.target, "id", None)
        if name in names:
            nodes.append(node)
    assert len(nodes) == len(names), (filename, names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_device("cpu")

    def source(ref, path):
        return subprocess.check_output(["git", "-C", str(args.repo), "show", f"{ref}:{path}"])

    original_test = source(HEAD, TEST)
    result = {"scope": __doc__, "base": BASE, "head": HEAD,
              "python": platform.python_version(), "torch": torch.__version__,
              "pytest": pytest.__version__, "platform": platform.platform(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "test_sha256": hashlib.sha256(original_test).hexdigest(), "runs": {}}
    for ref in (BASE, HEAD):
        sources = {path: source(ref, path) for path in (CORE, CORR, FUNC)}
        ns = {"torch": torch, "math": math, "pytest": pytest, "Optional": Optional, "Any": Any}
        functional = {"torch": torch}
        select(sources[FUNC], {"masked_sum", "masked_mean"}, functional, FUNC)
        ns["verl_F"] = SimpleNamespace(**{k: functional[k] for k in ("masked_sum", "masked_mean")})
        select(sources[CORE], {"kl_penalty", "kl_penalty_forward"}, ns, CORE)
        select(sources[CORR], CORR_NAMES, ns, CORR)
        select(original_test, TEST_NAMES | {"_small_k3_reference"}, ns, TEST)
        cases = []
        for name in sorted(TEST_NAMES):
            fn = ns[name]
            parameter_groups = []
            for mark in getattr(fn, "pytestmark", []):
                assert mark.name == "parametrize"
                keys = [key.strip() for key in mark.args[0].split(",")]
                parameter_groups.append([
                    dict(zip(keys, values if len(keys) > 1 else (values,), strict=True))
                    for values in mark.args[1]
                ])
            for combination in itertools.product(*parameter_groups):
                kwargs = {key: value for group in combination for key, value in group.items()}
                try:
                    fn(**kwargs)
                    outcome, detail = "passed", None
                except AssertionError as exc:
                    outcome, detail = "assertion_failed", str(exc)[:600]
                cases.append({"name": name, "parameters": {k: str(v) for k, v in kwargs.items()},
                              "outcome": outcome, "detail": detail})
        assert len(cases) == 20
        failed = [case for case in cases if case["outcome"] != "passed"]
        assert len(failed) == (8 if ref == BASE else 0), failed
        # Observe the same long-sequence input used in the original regression.
        x = torch.full((1, 32768), -1e-4, dtype=torch.float32)
        keep, metrics = ns["compute_rollout_rejection_mask"](x, torch.ones_like(x), "seq_sum_k3", 1e-4)
        result["runs"][ref] = {
            "source_sha256": {p: hashlib.sha256(s).hexdigest() for p, s in sources.items()},
            "passed": 20 - len(failed), "assertion_failed": len(failed), "cases": cases,
            "long_sequence": {"actual_fp32_x": x[0, 0].item(),
                              "oracle_sum": ns["_small_k3_reference"](x).sum().item(),
                              "observed_sum": metrics["rollout_rs_seq_sum_k3_mean"],
                              "kept_tokens": int(keep.sum().item())},
        }
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(json.dumps({ref: {k: v for k, v in run.items() if k in
                            {"passed", "assertion_failed", "long_sequence"}}
                      for ref, run in result["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
