"""CPU checks of verl #8444: nonnegative weights are not accurate weights.

Uses fixed, locally cached Git blobs; never fetches or edits the source checkout.
Original selected definitions/tests are from verl-project/verl (Apache-2.0).
Only registration/pytest decorators are removed, with annotations deferred.
The metrics check uses a plain batch-dict adapter, not real DataProto or its test.
No device discovery, model, trainer, native pytest collection, or GPU execution.
Requires Python 3.12, torch and numpy. Prints JSON; does not write result files.
"""

import __future__
import argparse
import ast
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


BASE = "59f9a42c749a49793ee38298f3081beab43d79bc"
HEAD = "615aa6111ce0c5f3ef8ed3bb0614444d9a9223f3"
CORE = "verl/trainer/ppo/core_algos.py"
METRIC = "verl/trainer/ppo/metric_utils.py"
UTIL = "verl/utils/torch_functional.py"
TEST = "tests/trainer/ppo/test_core_algos_on_cpu.py"
ESTIMATORS = ["compute_optimal_token_baseline_advantage",
              "compute_multi_turn_optimal_token_baseline_advantage"]


def load_definitions(tree, names, namespace):
    nodes = [copy.deepcopy(n) for n in tree.body if getattr(n, "name", None) in names]
    assert {n.name for n in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, "selected_original_definitions", "exec",
                 flags=__future__.annotations.compiler_flag), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    hashes = {}

    def source(revision, path):
        key = revision + ":" + path
        raw = subprocess.check_output(["git", "show", key], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[key] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw, filename=key)
        compile(tree, key, "exec")
        return tree

    parent = subprocess.check_output(["git", "show", "-s", "--format=%P", HEAD],
                                     cwd=args.source_repo, text=True).strip()
    assert parent == BASE
    torch.set_num_threads(1)
    helpers = {"torch": torch, "F": F}
    load_definitions(source(BASE, UTIL), ["masked_sum", "masked_mean",
                     "logprobs_from_logits_v2", "calculate_sum_pi_squared_from_logits"], helpers)
    # The PR does not modify these helpers; verify rather than mix versions.
    source(HEAD, UTIL)
    assert hashes[BASE + ":" + UTIL] == hashes[HEAD + ":" + UTIL]
    test_tree = source(HEAD, TEST)
    versions = {}
    for label, revision in [("base", BASE), ("candidate", HEAD)]:
        namespace = {"torch": torch, "np": np, "defaultdict": defaultdict,
                     "verl_F": SimpleNamespace(**helpers)}
        load_definitions(source(revision, CORE), ESTIMATORS, namespace)
        load_definitions(source(revision, METRIC), ["compute_variance_proxy_metrics"], namespace)
        load_definitions(test_tree, ["test_otb_nonnegative_path_variance_on_confident_tokens"], namespace)
        versions[label] = namespace

    original_cases = {}
    witness = {}
    same_logits = {}
    logits = torch.zeros((4, 1, 128), dtype=torch.float32, device="cpu")
    logits[:, :, 0] = 14.0
    labels = torch.zeros((4, 1), dtype=torch.long, device="cpu")
    lp = helpers["logprobs_from_logits_v2"](logits, labels)
    sps = helpers["calculate_sum_pi_squared_from_logits"](logits)
    p64 = logits.double().softmax(dim=-1)
    # Direct squared norm avoids the expanded formula's cancellation.
    reference_w = (1 - p64[:, :, 0]).square() + p64[:, :, 1:].square().sum(dim=-1)
    reference_b = reference_w.sum(dim=0) / (reference_w.sum(dim=0) + 1e-8)
    raw_w = 1 - 2 * lp.exp() + sps
    assert torch.equal(raw_w, torch.zeros_like(raw_w))
    assert reference_w.min().item() > 0 and reference_b.item() > 0.8

    for label, namespace in versions.items():
        original_cases[label] = {}
        witness[label] = {}
        same_logits[label] = {}
        for name in ESTIMATORS:
            fn = namespace[name]
            try:
                namespace["test_otb_nonnegative_path_variance_on_confident_tokens"](fn)
            except AssertionError as error:
                original_cases[label][name] = {"passed": False, "error": str(error)}
            else:
                original_cases[label][name] = {"passed": True}
            assert original_cases[label][name]["passed"] == (label == "candidate")

            kwargs = dict(token_level_rewards=torch.tensor([[0.0, 1.0]] * 4),
                          response_mask=torch.ones((4, 2)),
                          index=np.array(["g0"] * 4, dtype=object),
                          old_log_probs=torch.zeros((4, 2)),
                          sum_pi_squared=torch.full((4, 2), float.fromhex("0x1.ffff00p-1")))
            adv, ret = fn(**kwargs)
            witness[label][name] = {"baseline": (ret - adv).tolist()}
            assert torch.isfinite(adv).all() and not adv.requires_grad

            adv, ret = fn(token_level_rewards=torch.ones((4, 1)),
                          response_mask=torch.ones((4, 1)),
                          index=np.array(["g0"] * 4, dtype=object),
                          old_log_probs=lp, sum_pi_squared=sps)
            baseline = ret - adv
            assert torch.equal(baseline, torch.zeros_like(baseline))
            same_logits[label][name] = {"baseline": baseline.tolist(),
                                      "max_abs_error_vs_fp64_reference":
                                      (baseline.double() - reference_b).abs().max().item()}

        # Same tensor values as the new upstream metrics test, but the adapter
        # replaces DataProto construction. Do not count this as that test passing.
        batch = SimpleNamespace(batch={"old_log_probs": torch.zeros((2, 1)),
                "sum_pi_squared": torch.full((2, 1), float.fromhex("0x1.ffff00p-1")),
                "advantages": torch.ones((2, 1)), "response_mask": torch.ones((2, 1))})
        metrics = namespace["compute_variance_proxy_metrics"](batch)
        witness[label]["metrics_dict_adapter"] = metrics
        assert (metrics["variance_proxy/expected_w"] >= 0) == (label == "candidate")
        assert metrics["variance_proxy/proxy2_total_power"] == 0.0

    print(json.dumps({"base_first_parent": BASE, "head": HEAD,
          "environment": {"python": platform.python_version(), "torch": torch.__version__,
                          "numpy": np.__version__, "device": "cpu", "threads": 1},
          "source_sha256": hashes,
          "original_test_parameter_bodies": original_cases,
          "constructed_negative_weight_witness": {"raw_w": -(2 ** -17), "results": witness},
          "same_logits_precision_control": {"logits_shape": list(logits.shape),
              "selected_logit": 14.0, "other_logits": 0.0, "selected_label": 0,
              "old_log_probs": lp.tolist(), "sum_pi_squared": sps.tolist(),
              "fp32_raw_w": raw_w.tolist(), "fp64_direct_w": reference_w.tolist(),
              "epsilon": 1e-8, "returns": 1.0, "fp64_reference_baseline": reference_b.tolist(),
              "results": same_logits},
          "scope": {"original_estimator_test_parameters": 2,
              "metrics_input_adapter": "SimpleNamespace(batch=dict), not DataProto or original metrics test",
              "logprob_path": "original logprobs_from_logits_v2, float32 branch; no fused dispatch",
              "not_run": ["native pytest collection", "full suite", "DataProto integration",
                          "trainer", "models", "GPU/NPU", "performance", "training quality"]}},
          ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
