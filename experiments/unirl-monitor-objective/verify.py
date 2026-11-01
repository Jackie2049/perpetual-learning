"""#641 original spec/backend with real Hydra containers, own response fixtures.

PYTHONPATH=/isolated/hydra-deps python verify.py /cached/UniRL-git
HTTP session is a constructor/close recorder; no HTTP, Ray, scorer or trainer.
"""

import __future__
import ast
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from types import ModuleType, SimpleNamespace

from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

BASE = "b8e9740e981a68a5556fbeb8225558040a4d47a4"
HEAD = "ec026cd73c73ec4b9ee9cc3feb23e834e90357d6"
REMOTE = "unirl/reward/remote.py"
METHODS = ("weighted_sum", "mean", "min", "max")
EXPECTED = {"weighted_sum": (0.8 * 2 + 0.6) / 3, "mean": 0.7, "min": 0.6, "max": 0.8}


class SessionRecorder:
    """Own no-network seam for original backend constructor and dispose only."""
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def main(cache):
    hashes = {}
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")

    def read(ref, path):
        raw = subprocess.check_output(["git", "-C", cache, "show", f"{ref}:{path}"], env=env)
        hashes[f"{ref}:{path}"] = hashlib.sha256(raw).hexdigest()
        compile(raw, path, "exec")
        return ast.parse(raw)

    def named(tree, name):
        return next(node for node in tree.body if getattr(node, "name", None) == name)

    def execute(nodes, module):
        tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(tree, "original_ast", "exec", flags=__future__.annotations.compiler_flag), module.__dict__)

    def module(name):
        result = ModuleType(name)
        sys.modules[name] = result  # Real Hydra target lookup / dataclass annotation lookup.
        return result

    ancestor = subprocess.check_output(["git", "-C", cache, "merge-base", BASE, HEAD], env=env).decode().strip()
    assert ancestor == BASE
    batch = module("cpu_monitor_original_batch")
    execute(read(HEAD, "unirl/distributed/tensor/batch.py").body, batch)
    reward = module("cpu_monitor_original_response")
    reward.__dict__.update(Batch=batch.Batch, concat_field=batch.concat_field,
                           max_field=batch.max_field, dataclass=dataclass)
    execute([named(read(HEAD, "unirl/types/reward.py"), "RewardResponse")], reward)
    require_mod = module("cpu_monitor_original_require")
    execute(read(HEAD, "unirl/config/require.py").body, require_mod)
    read(HEAD, "unirl/reward/service.py")  # Static fail-fast scope only, not executed.
    originals = {}
    remote_trees = {}
    for label, ref in (("base", BASE), ("head", HEAD)):
        ns = module("cpu_monitor_" + label)
        ns.__dict__.update(DEFAULT_PROMPT_SOURCE="generation", PROMPT_SOURCES=("original", "generation"),
                           RewardResponse=reward.RewardResponse, require=require_mod.require,
                           math=math, http_requests=SimpleNamespace(Session=SessionRecorder))
        base_tree = read(ref, "unirl/reward/base.py")
        execute([node for node in base_tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == "unirl.types.reward")], ns)
        tree = read(ref, REMOTE)
        remote_trees[label] = tree
        execute([named(tree, "RemoteRewardBackend"), named(tree, "RemoteRewardSpec")], ns)
        originals[label] = ns

    def config(label="head", method="weighted_sum", monitor=True, **overrides):
        values = {"base_url": "http://offline.invalid/", "required_rewards": ["main_a", "main_b", "probe"],
                  "reward_weights": {"main_a": 2.0}, "aggregation_method": method}
        if label == "head" and monitor:
            values["monitor_rewards"] = ["probe"]
        values.update(overrides)
        return values

    def backend(label="head", method="weighted_sum", monitor=True, **overrides):
        ns = originals[label]
        spec = ns.RemoteRewardSpec(**config(label, method, monitor, **overrides))
        result = ns.RemoteRewardBackend(config=spec, base_device="cpu")
        assert result._session.trust_env is False
        result.dispose()
        assert result._session.closed
        return result

    def parse(parser, rows):
        result = parser._parse_score_response({"results": rows, "errors": [{} for _ in rows]}, len(rows), 0.01)
        assert isinstance(result, reward.RewardResponse)
        assert all(len(values) == len(rows) for values in result.component_rewards.values())
        return asdict(result)

    def row(probe):
        return {"main_a": {"score": 0.8}, "main_b": {"score": 0.6}, "probe": {"score": probe}}

    default_compat = []
    invariance = []
    failures = []
    for method in METHODS:
        for condition in ("normal", "missing", "nan"):
            sample = row(0.1)
            if condition == "missing":
                del sample["main_b"]
            elif condition == "nan":
                sample["main_b"]["score"] = float("nan")
            lhs = parse(backend("base", method), [sample])
            rhs = parse(backend("head", method, monitor=False), [sample])
            assert lhs == rhs
            default_compat.append({"aggregation": method, "condition": condition, "exact": True})
        parser = backend(method=method)
        assert parser.required_rewards == ["main_a", "main_b", "probe"]
        for probe in (0.0, 0.1, 99.0, -5.0):
            result = parse(parser, [row(probe), row(1.0 - probe)])
            assert all(math.isclose(value, EXPECTED[method], abs_tol=1e-15) for value in result["rewards"])
            assert result["component_rewards"]["monitor_probe"] == [probe, 1.0 - probe]
            assert "probe" not in result["component_rewards"] and result["successes"] == [True, True]
            invariance.append({"aggregation": method, "probe": probe, "rewards": result["rewards"]})
        for condition, invalid in (("missing", None), ("nan", float("nan")),
                                   ("inf", float("inf")), ("bool", True)):
            bad = row(invalid)
            if condition == "missing":
                del bad["probe"]
            result = parse(parser, [row(0.1), bad])
            assert result["successes"] == [True, False] and "probe" in result["errors"][1]
            assert result["component_rewards"]["monitor_probe"] == [0.1, 0.0]
            assert all(math.isclose(value, EXPECTED[method], abs_tol=1e-15) for value in result["rewards"])
            failures.append({"aggregation": method, "condition": condition,
                             "successes": result["successes"], "error": result["errors"][1]})
    no_training = parse(backend(), [{"probe": {"score": 7.0}}])
    assert no_training["rewards"] == [0.0] and no_training["successes"] == [False]

    # Real Hydra instantiate against original dataclass in an explicit in-memory module.
    # Not recipe composition, config-store registration or a full UniRL import.
    hydra_results = []
    invalid_specs = {
        "unknown_monitor": {"monitor_rewards": ["absent"]},
        "all_monitors": {"monitor_rewards": ["main_a", "main_b", "probe"], "reward_weights": {}},
        "monitor_weight_even_zero": {"reward_weights": {"probe": 0.0}},
        "component_key_collision": {"required_rewards": ["main_a", "probe", "monitor_probe"]},
    }
    for conversion in ("none", "all"):
        cfg = OmegaConf.create({"_target_": "cpu_monitor_head.RemoteRewardSpec", "_convert_": conversion,
                                **config()})
        spec = instantiate(cfg)
        assert isinstance(spec, originals["head"].RemoteRewardSpec)
        if conversion == "none":
            assert isinstance(spec.required_rewards, ListConfig) and isinstance(spec.monitor_rewards, ListConfig)
            assert isinstance(spec.reward_weights, DictConfig)
        parsed = parse(originals["head"].RemoteRewardBackend(config=spec, base_device="cpu"), [row(99.0)])
        assert math.isclose(parsed["rewards"][0], EXPECTED["weighted_sum"], abs_tol=1e-15)
        hydra_results.append({"conversion": conversion, "condition": "valid", "status": "accepted",
                              "monitor_type": type(spec.monitor_rewards).__name__,
                              "weights_type": type(spec.reward_weights).__name__})
        for condition, overrides in invalid_specs.items():
            bad_cfg = OmegaConf.create({"_target_": "cpu_monitor_head.RemoteRewardSpec",
                                       "_convert_": conversion, **config(**overrides)})
            try:
                instantiate(bad_cfg)
            except InstantiationException as exc:
                assert isinstance(exc.__cause__, ValueError)
                hydra_results.append({"conversion": conversion, "condition": condition,
                                      "status": "rejected", "cause": str(exc.__cause__)})
            else:
                raise AssertionError(condition)

    # Precise legacy workaround contrast, no policy/advantage rerun.
    old_min = parse(backend("base", "min", reward_weights={"probe": 1e-6}), [row(0.1)])
    new_min = parse(backend(method="min"), [row(0.1)])
    assert old_min["rewards"] == [0.1] and new_min["rewards"] == [0.6]
    wire_same = ast.dump(named(remote_trees["base"], "_wire_identity")) == ast.dump(
        named(remote_trees["head"], "_wire_identity"))
    assert wire_same  # Static identity helper equality, no payload/server/cache execution.
    print(json.dumps({
        "base_and_merge_base": ancestor, "head": HEAD, "source_sha256": hashes,
        "environment": {"python": platform.python_version(),
                        **{name: importlib.metadata.version(name) for name in ("hydra-core", "omegaconf", "torch")}},
        "default_compatibility": default_compat, "monitor_objective_invariance": invariance,
        "required_monitor_failures": failures, "no_valid_training_component": no_training,
        "hydra_original_spec": hydra_results,
        "legacy_tiny_weight_min": old_min["rewards"], "monitor_min": new_min["rewards"],
        "wire_identity_helper_ast_unchanged": wire_same,
        "scope": "Own 12 compatibility, 16 invariance, 16 bad-monitor, 10 Hydra configuration controls, "
                 "one missing-training control and legacy-min contrast. Original classes/spec/parser and Batch/response; "
                 "HTTP session constructor recorder. No author 37-test harness, full UniRL/Hydra recipe, "
                 "HTTP/server, RewardService, metrics dashboard, advantages, training, Ray or GPU/NPU.",
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main(sys.argv[1])
