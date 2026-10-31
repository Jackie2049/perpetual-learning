"""Pinned original validation metric functions and four test bodies, CPU only."""

import argparse
import ast
from collections import defaultdict
import copy
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable

import numpy as np
import pytest

BASE = "8e48d72de4766228c36623c90f9f5c49ee5b45f3"
HEAD = "9b6ebeb4ca3828cc17ff110943f873f5f8739b14"
SOURCE = "verl/trainer/ppo/metric_utils.py"
TESTS = "tests/trainer/ppo/test_metric_utils_on_cpu.py"
READING = ("verl/trainer/ppo/v1/trainer_base.py", "verl/workers/rollout/llm_server.py")
NAMES = ("bootstrap_metric", "calc_maj_val", "process_validation_metrics")
TEST_NAMES = (
    "test_real_run_partial_failure_scores_use_original_five_prompts",
    "test_reward_only_source_does_not_acquire_accuracy_or_extra_score",
    "test_extra_score_is_padded_only_for_sources_that_define_it",
    "test_all_failed_validation_still_has_zero_canonical_reward",
)


def load_nodes(tree, names, namespace):
    nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "original-functions", "exec"),
         namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    hashes, trees = {}, {}
    for ref, path in [(BASE, SOURCE), (HEAD, SOURCE), (HEAD, TESTS)] + [(HEAD, path) for path in READING]:
        raw = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=args.repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        tree = ast.parse(raw)
        compile(tree, path, "exec")
        hashes[f"{ref}:{path}"] = hashlib.sha256(raw).hexdigest()
        trees[ref, path] = tree
    results, functions = {}, {}
    for ref in (BASE, HEAD):
        namespace = {"np": np, "pytest": pytest, "defaultdict": defaultdict, "partial": partial,
                     "Any": Any, "Callable": Callable}
        load_nodes(trees[ref, SOURCE], NAMES, namespace)
        load_nodes(trees[HEAD, TESTS], TEST_NAMES, namespace)
        statuses = {}
        for name in TEST_NAMES:
            try:
                namespace[name]()
            except AssertionError:
                statuses[name] = "assertion_failed"
            else:
                statuses[name] = "passed"
        results[ref] = statuses
        functions[ref] = namespace["process_validation_metrics"]
    assert set(results[BASE].values()) == {"assertion_failed"}
    assert set(results[HEAD].values()) == {"passed"}
    process = functions[HEAD]
    unequal = process(["s", "s"], ["a", "b"], {"reward": [1.0, 4.0]},
                      expected_acc_counts={("s", "a"): 2, ("s", "b"): 4})["s"]
    unequal_summary = {key: float(value) for key, value in unequal["reward"].items() if key.startswith("mean@")}
    assert unequal_summary == {"mean@2": 0.5, "mean@4": 1.0}
    negative = process(["s"], ["a"], {"reward": [-1.0]}, expected_acc_counts={("s", "a"): 2})["s"]
    assert negative["reward"]["mean@2"] == -0.5
    assert negative["reward_success_only"]["mean@1"] == -1.0
    print(json.dumps({"python": platform.python_version(), "numpy": np.__version__, "pytest": pytest.__version__,
                      "base": BASE, "head": HEAD, "source_sha256": hashes, "original_test_bodies": results,
                      "own_controls": {
                          "unequal_expected_counts": {"canonical_mean_keys": unequal_summary,
                                                      "completed_mean_at_1": float(unequal["reward_success_only"]["mean@1"]),
                                                      "arithmetic_attempt_weighted_mean_not_returned": 5 / 6},
                          "negative_reward": {"canonical_mean_at_2": float(negative["reward"]["mean@2"]),
                                              "completed_mean_at_1": float(negative["reward_success_only"]["mean@1"])}},
                      "scope": "Four unchanged original test bodies with pytest.approx, not native pytest "
                      "collection or the full suite. Three complete original metric functions per version; "
                      "original bootstrap executes normally with 1000 resamples where invoked. Two own "
                      "aggregation controls. Full source files compile only outside selected functions. "
                      "No trainer/validation lifecycle, model samples, grading, TQ, Ray cancellation, "
                      "uni-agent, training or GPU/NPU execution."}, indent=2))


if __name__ == "__main__":
    main()
