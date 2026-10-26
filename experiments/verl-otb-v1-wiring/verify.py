"""Exercise #8364's seven original parameter cases with extracted definitions.

Original definitions/tests: verl-project/verl, Apache-2.0, ByteDance contributors.
No implementation or fixture bodies are rewritten. The upstream FakeTQ and
MagicMock workers remain mocks; this is not native pytest, Ray, or training.
Dependencies are read from fixed base Git blobs; the PR changes only trainer
and test. Candidate/test copies must match GitHub's fixed Git blob identities.
"""

import __future__
import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import numpy as np
from omegaconf import OmegaConf
import pytest
from tensordict import TensorDict
import tensordict
import torch

BASE = "3c6d9b43b67c74ec343d77874cd8fa6e841523da"
HEAD = "d17219079c48e3f580922c59702a5ee03002d067"
TRAINER = "verl/trainer/ppo/v1/trainer_base.py"


def execute(nodes, ns):
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, "original_definitions", "exec", flags=__future__.annotations.compiler_flag), ns)


def definitions(tree, names, ns):
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    assert len(nodes) == len(names), names
    execute(nodes, ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("candidate_sources", type=Path,
                        help="candidate.py and test.py downloaded from fixed HEAD")
    args = parser.parse_args()
    hashes = {}

    def parse(raw, key):
        hashes[key] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw, filename=key)
        compile(tree, key, "exec")
        return tree

    def source(path):
        raw = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        return parse(raw, BASE + ":" + path)

    trees = {}
    for filename, oid in [("candidate.py", "e8cfe65b4e1907cb35601e55a787f8df28494569"),
                          ("test.py", "0346a1fd629d9fbbcabc92a4facf1cc63453a225")]:
        raw = (args.candidate_sources / filename).read_bytes()
        assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == oid
        trees[filename] = parse(raw, HEAD + ":" + filename)

    dep_trees = {p: source(p) for p in ["verl/protocol.py", "verl/utils/torch_functional.py",
                 "verl/workers/utils/padding.py", "verl/trainer/ppo/core_algos.py",
                 "verl/trainer/ppo/ray_trainer.py", "verl/trainer/ppo/v1/utils.py"]}
    results = {}
    for label, trainer_tree in [("base", source(TRAINER)), ("candidate", trees["candidate.py"])]:
        module = ModuleType("_original_otb_" + label)
        sys.modules[module.__name__] = module
        ns = module.__dict__
        ns.update(torch=torch, np=np, TensorDict=TensorDict, dataclass=dataclass, field=field,
                  Any=Any, Optional=Optional, Callable=Callable, defaultdict=defaultdict,
                  SimpleNamespace=SimpleNamespace, MagicMock=MagicMock, pytest=pytest,
                  OmegaConf=OmegaConf, tq=None, trainer_base=module)
        definitions(dep_trees["verl/protocol.py"], {"DataProto", "DataProtoItem"}, ns)
        functional = ModuleType("_original_torch_helpers")
        functional.__dict__.update(torch=torch)
        definitions(dep_trees["verl/utils/torch_functional.py"], {"masked_sum"}, functional.__dict__)
        core = ModuleType("_original_otb_algos")
        core.__dict__.update(torch=torch, np=np, Enum=Enum, Any=Any, defaultdict=defaultdict,
                             verl_F=functional)
        algo_tree = dep_trees["verl/trainer/ppo/core_algos.py"]
        declarations = [n for n in algo_tree.body if isinstance(n, ast.AnnAssign)
                        and isinstance(n.target, ast.Name) and n.target.id == "ADV_ESTIMATOR_REGISTRY"]
        assert len(declarations) == 1
        execute(declarations, core.__dict__)
        definitions(algo_tree, {"AdvantageEstimator", "register_adv_est", "get_adv_estimator_fn",
                    "compute_optimal_token_baseline_advantage",
                    "compute_multi_turn_optimal_token_baseline_advantage", "agg_loss"}, core.__dict__)
        ns.update(core_algos=core, AdvantageEstimator=core.AdvantageEstimator, agg_loss=core.agg_loss)
        definitions(dep_trees["verl/workers/utils/padding.py"],
                    {"response_from_nested", "response_to_nested"}, ns)
        definitions(dep_trees["verl/trainer/ppo/ray_trainer.py"], {"compute_advantage"}, ns)
        definitions(dep_trees["verl/trainer/ppo/v1/utils.py"], {"compute_advantage_for_multi_trajectories"}, ns)
        original_class = next(n for n in trainer_tree.body if isinstance(n, ast.ClassDef) and n.name == "PPOTrainer")
        methods = [n for n in original_class.body if getattr(n, "name", None) in
                   {"_compute_old_log_prob", "_compute_ref_log_prob", "_compute_advantage"}]
        assert len(methods) == 3
        execute([ast.ClassDef(name="PPOTrainer", bases=[], keywords=[], body=methods, decorator_list=[])], ns)
        # Keep all original fixtures/test bodies and pytest decorators, omit only
        # imports (injected above) and use the extracted three-method container.
        execute([n for n in trees["test.py"].body if not isinstance(n, (ast.Import, ast.ImportFrom))], ns)
        cases = [("test_old_log_prob_writes_response_aligned_sum_pi_squared", (flag,)) for flag in (True, False)]
        cases += [("test_optimal_token_baseline_receives_sum_pi_squared", (adv,))
                  for adv in ("optimal_token_baseline", "tir_optimal_token_baseline")]
        cases += [("test_ref_log_prob_keeps_actor_sum_pi_squared", (flag,)) for flag in (True, False)]
        cases += [("test_bypass_mode_rejects_sum_pi_squared", ())]
        outcomes = []
        for name, values in cases:
            with pytest.MonkeyPatch.context() as patch:
                try:
                    ns[name](patch, *values)
                except Exception as exc:
                    outcome = {"status": "failed", "exception": type(exc).__name__, "message": str(exc)}
                else:
                    outcome = {"status": "passed"}
            outcomes.append({"name": name, "parameters": list(values), **outcome})
        results[label] = outcomes

    assert all(r["status"] == "passed" for r in results["candidate"]), results
    assert [r.get("exception") for r in results["base"]] == [
        "KeyError", "KeyError", "AssertionError", "AssertionError", "KeyError", "KeyError", "KeyError"], results
    assert all("requires sum_pi_squared" in r["message"] for r in results["base"][2:4]), results
    assert all(r["message"] == "'calculate_sum_pi_squared'" for r in
               results["base"][:2] + results["base"][4:6]), results
    assert "rollout_log_probs" in results["base"][6]["message"], results
    assert "verl" not in sys.modules and "ray" not in sys.modules and "transfer_queue" not in sys.modules
    print(json.dumps({"base": BASE, "head": HEAD, "source_sha256": hashes, "results": results,
                      "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                      "numpy": np.__version__, "tensordict": tensordict.__version__,
                                      "pytest": pytest.__version__, "system": platform.system()},
                      "scope": "original seven parameter cases; extracted complete methods/functions/classes; "
                      "real CPU jagged/padded TensorDict/DataProto/OTB; original FakeTQ and mock workers; "
                      "no native pytest/trainer constructor/TransferQueue/Ray/engine/reference writes/GPU/training"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
