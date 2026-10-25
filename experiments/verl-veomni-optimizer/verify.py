"""Run #8370's five original config-forwarding cases without VeOmni/Ray.

Upstream source: verl-project/verl, Apache-2.0, ByteDance contributors.
Original config classes and method/test bodies are AST-loaded; only the engine
class shell bypasses distributed construction. The builder is the original
test fixture's recorder. A separate tiny CPU AdamW step is not a benchmark.
"""

import __future__
import argparse
import ast
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Literal, Optional
import warnings

from omegaconf import MISSING
import pytest
import torch

VERSIONS = {"base": "3c6d9b43b67c74ec343d77874cd8fa6e841523da",
            "head": "44c97bcd4bd167b93462e3bc380bee6947a90e5f"}
ENGINE = "verl/workers/engine/veomni/transformer_impl.py"
TEST = "tests/workers/test_veomni_optimizer_overrides_on_cpu.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    hashes, results = {}, {}

    def source(revision, path):
        raw = subprocess.check_output(["git", "show", revision + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw, filename=path)
        compile(tree, path, "exec")
        return tree

    def execute(nodes, ns):
        tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(tree, "original_definitions", "exec",
                     flags=__future__.annotations.compiler_flag), ns)

    def definitions(tree, names, ns):
        nodes = [node for node in tree.body if getattr(node, "name", None) in names]
        assert len(nodes) == len(names)
        execute(nodes, ns)

    test_tree = source(VERSIONS["head"], TEST)
    for label, revision in VERSIONS.items():
        module = ModuleType("_original_optimizer_" + label)
        sys.modules[module.__name__] = module
        ns = module.__dict__
        ns.update(dataclass=dataclass, field=field, Any=Any, Callable=Callable, Literal=Literal,
                  Optional=Optional, logging=logging, math=math, warnings=warnings, MISSING=MISSING,
                  logger=logging.getLogger("isolated_veomni_config"), SimpleNamespace=SimpleNamespace,
                  pytest=pytest, build_optimizer=None)
        base_tree = source(revision, "verl/base_config.py")
        execute(base_tree.body, ns)
        definitions(source(revision, "verl/workers/config/engine.py"),
                    {"EngineRouterReplayConfig", "EngineConfig", "VeOmniEngineConfig"}, ns)
        definitions(source(revision, "verl/workers/config/optimizer.py"),
                    {"OptimizerConfig", "VeOmniOptimizerConfig"}, ns)
        engine_tree = source(revision, ENGINE)
        engine = next(n for n in engine_tree.body if getattr(n, "name", None) == "VeOmniEngine")
        method = next(n for n in engine.body if getattr(n, "name", None) == "_build_optimizer")
        execute([ast.ClassDef(name="VeOmniEngine", bases=[], keywords=[], body=[method],
                              decorator_list=[], type_params=[])], ns)
        ns["transformer_impl"] = module
        definitions(test_tree, {"_engine", "build_optimizer_calls", "test_optimizer_overrides_reach_build_optimizer",
                                "test_override_keys_other_than_fused_are_forwarded"}, ns)
        records = {}
        cases = [("default", False, None, None), ("override", False, {"fused": True}, True),
                 ("offload-default", True, None, True), ("offload-override", True, {"fused": False}, False)]
        for name, offload, override, fused in cases + [("other-key", None, None, None)]:
            with pytest.MonkeyPatch.context() as monkeypatch:
                calls = inspect.unwrap(ns["build_optimizer_calls"])(monkeypatch)
                try:
                    if name == "other-key":
                        ns["test_override_keys_other_than_fused_are_forwarded"](calls)
                    else:
                        ns["test_optimizer_overrides_reach_build_optimizer"](calls, offload, override, fused)
                except (AssertionError, KeyError) as error:
                    assert label == "base" and name != "default"
                    status = "expected_" + type(error).__name__
                else:
                    status = "passed"
            assert len(calls) == 1
            should_pass = label == "head" or name == "default"
            assert (status == "passed") == should_pass
            records[name] = {"status": status, "builder_kwargs": calls[0]}
        results[label] = records

    # Direct PyTorch control, deliberately not the VeOmni builder or offloaded DTensor engine.
    parameters, states = {}, {}
    for label, options in {"foreach": {"foreach": True, "fused": False},
                           "fused": {"foreach": False, "fused": True}}.items():
        parameter = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 257, dtype=torch.float32))
        parameter.grad = torch.linspace(-0.5, 0.5, 257, dtype=torch.float32)
        optimizer = torch.optim.AdamW([parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=0.01, **options)
        optimizer.step()
        parameters[label] = parameter.detach().clone()
        states[label] = {k: v.clone() for k, v in optimizer.state[parameter].items()}
    max_abs = (parameters["fused"] - parameters["foreach"]).abs().max().item()
    torch.testing.assert_close(parameters["fused"], parameters["foreach"], atol=1e-7, rtol=1e-6)
    state_diffs = {k: (states["fused"][k] - states["foreach"][k]).abs().max().item()
                   for k in states["fused"]}
    assert "veomni" not in sys.modules and "ray" not in sys.modules
    print(json.dumps({"versions": VERSIONS, "source_sha256": hashes, "results": results,
                      "tiny_adamw_cpu_control": {"numel": 257, "steps": 1, "dtype": "float32",
                                                "parameter_max_abs_diff": max_abs,
                                                "parameter_bitwise_equal": torch.equal(parameters["fused"], parameters["foreach"]),
                                                "state_max_abs_diff": state_diffs,
                                                "timing_or_memory_measured": False},
                      "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                      "pytest": pytest.__version__, "system": platform.system()},
                      "scope": "five unchanged original test cases; original config classes/method; "
                               "original recorder builder, not original pytest collection; "
                               "separate direct CPU AdamW; no VeOmni/DTensor/FSDP/ROCm/checkpoint ingestion"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
