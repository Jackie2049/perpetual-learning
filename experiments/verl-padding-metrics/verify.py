#!/usr/bin/env python3
"""Run one unchanged upstream regression body against pinned original CPU functions."""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch

REVISIONS = {
    "ancestor": "d040717b21af2e23e8e789a3e354cff2394ae2de",
    "candidate": "7ffe2fa4a4f5581c27e09b8f2e438e3b61d8e7ee",
    "main": "3c6d9b43b67c74ec343d77874cd8fa6e841523da",
}
HELPER = "verl/trainer/ppo/rollout_corr_helper.py"
UTILS = "verl/utils/torch_functional.py"
TEST = "tests/trainer/ppo/test_rollout_corr.py"
CASE = "test_offpolicy_metrics_ignore_all_padding_rows"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verl", type=Path)
    args = parser.parse_args()
    hashes = {}

    def source(revision, path):
        raw = subprocess.check_output(["git", "show", revision + ":" + path], cwd=args.verl,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw, filename=path)
        compile(tree, path, "exec")  # Full module syntax only; no package import.
        return tree

    def functions(tree, names, namespace, filename):
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes} == set(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
        return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)

    test_tree = source(REVISIONS["candidate"], TEST)
    output, function_asts = {}, {}
    for label, revision in REVISIONS.items():
        utility_namespace = {"torch": torch}
        functions(source(revision, UTILS), ["masked_mean", "masked_sum"], utility_namespace, UTILS)
        helper_tree = source(revision, HELPER)
        safety = next(ast.literal_eval(n.value) for n in helper_tree.body
                      if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SAFETY_BOUND"
                                                          for t in n.targets))
        namespace = {"torch": torch, "Any": Any, "Optional": Optional, "SAFETY_BOUND": safety,
                     "verl_F": SimpleNamespace(**{n: utility_namespace[n] for n in ("masked_mean", "masked_sum")})}
        function_asts[label] = functions(helper_tree, ["compute_offpolicy_metrics"], namespace, HELPER)
        namespace["pytest"] = pytest
        functions(test_tree, [CASE], namespace, TEST)
        try:
            namespace[CASE]()
            status = "passed"
        except AssertionError as error:
            status = "assertion_failed"
            assert label != "candidate"
            failure = str(error)
        assert status == ("passed" if label == "candidate" else "assertion_failed")
        fn = namespace["compute_offpolicy_metrics"]
        real = fn(torch.full((1, 6), -2.0), torch.full((1, 6), -1.0), torch.ones(1, 6))
        mask = torch.cat([torch.ones(1, 6), torch.zeros(3, 6)])
        padded = fn(torch.full((4, 6), -2.0), torch.full((4, 6), -1.0), mask)
        assert all(real[key] == pytest.approx(padded[key]) for key in ("kl", "k3_kl", "chi2_token"))
        output[label] = {"original_test": status, "real": real, "with_three_padding_rows": padded}
        if status != "passed":
            output[label]["original_assertion"] = failure
        if label == "candidate":
            # One unequal-length algebra illustration, not another upstream regression.
            mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            old = torch.tensor([[-1.0] * 3, [-3.0] * 3])
            rollout = torch.tensor([[-0.5] * 3, [-2.0] * 3])
            metrics = fn(old, rollout, mask)
            output["aggregation_illustration"] = {
                "original_metrics": metrics,
                "exp_mean_sequence_nll": math.exp(metrics["training_log_ppl"]),
                "corpus_token_ppl": math.exp(float(-(old * mask).sum() / mask.sum())),
                "exp_mean_log_ppl_diff": math.exp(metrics["log_ppl_diff"]),
                "ratio_of_mean_ppls": metrics["training_ppl"] / metrics["rollout_ppl"],
            }
    assert function_asts["ancestor"] == function_asts["main"]
    print(json.dumps({"revisions": REVISIONS, "python": sys.version.split()[0],
                      "torch": torch.__version__, "pytest": pytest.__version__, "results": output,
                      "ancestor_main_metric_ast_equal": True, "source_sha256": hashes,
                      "scope": "one original test body; original metric/masked functions only; no trainer/TQ/IS/loss/model"},
                     indent=2))


if __name__ == "__main__":
    main()
