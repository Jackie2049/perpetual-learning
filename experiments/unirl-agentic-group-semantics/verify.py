"""Original normalization/loss controls for UniRL RFC #635, not a diagnostic.

Fixed local Git blobs, AST-extracted complete definitions, deferred annotations.
Group members are cardinality-only placeholders, not real Sample trajectories.
No frequencies, trainer, reward service, optimizer, model or device discovery.
Original definitions: Tencent-Hunyuan/UniRL (Apache-2.0).
"""

import __future__
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import torch


REVISION = "5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    namespace = {"torch": torch}
    hashes = {}
    for path, names in [("unirl/types/advantages.py", {"finite_mean_std"}),
                        ("unirl/trainer/agentic.py", {"_group_advantages"}),
                        ("unirl/algorithms/base.py", {"grpo_clip_loss", "select_active_tokens"}),
                        ("unirl/algorithms/grpo.py", set())]:  # static loss/backward audit only
        raw = subprocess.check_output(["git", "show", REVISION + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[path] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw)
        nodes = [copy.deepcopy(n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes} == names
        for node in nodes:
            node.decorator_list = []
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), path, "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
    torch.set_num_threads(1)
    results = {}
    for label, values, expected in [("no_finite", [None, None], [0, 0]),
                                    ("one_finite", [2, None], [0, 0]),
                                    ("equal_pair", [2, 2], [0, 0]),
                                    ("different_pair_with_failure", [0, 2, None], [-1, 1, 0])]:
        rewards = torch.tensor([float("nan") if v is None else v for v in values], dtype=torch.float32, device="cpu")
        mean, std = namespace["finite_mean_std"](rewards)
        advantages = namespace["_group_advantages"](None, [[None] * len(values)], rewards)
        assert torch.equal(advantages, torch.tensor(expected, dtype=torch.float32))
        results[label] = {"rewards_null_is_nan_sentinel": values, "normalizer_mean": mean.item(),
                          "normalizer_std": std.item(), "advantages": advantages.tolist()}
    new_logp = torch.tensor([-0.3, -0.2], device="cpu", requires_grad=True)
    loss, metrics = namespace["grpo_clip_loss"](new_logp=new_logp,
        old_logp=torch.full((2,), -0.4), advantages=torch.zeros(2), clip_range=0.2)
    loss.sum().backward()
    assert torch.equal(new_logp.grad, torch.zeros(2)) and metrics["approx_kl"].item() > 0
    print(json.dumps({"revision": REVISION, "source_sha256": hashes,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "cpu", "threads": 1}, "normalization_controls": results,
        "zero_advantage_clip_surrogate": {"loss_per_token": loss.detach().tolist(),
            "gradient_wrt_new_logp": new_logp.grad.tolist(),
            "approx_kl_metric": metrics["approx_kl"].item()},
        "scope": "Original function controls only; no diagnostic implementation, real Sample, trainer, service, optimizer, rates or training quality"},
        ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
