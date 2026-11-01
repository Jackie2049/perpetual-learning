#!/usr/bin/env python3
"""Offline #8203 control-plane slice; not Ray, TQ, dataloader or engine validation.

Run with the lightweight sft-hydra-cpu-venv Python (pytest + OmegaConf):
  python verify.py --repo /path/to/verl-cache > result.json
Git lazy fetching is disabled. No candidate code is changed or installed.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

HEAD = "6dc7bd9ec57da84f7c36a8bc081f12104ba0ff96"
BASE = "a4b9354968705a441fb7da7367924b15b902e771"
SOURCE = "verl/trainer/ppo/v1/streaming_dataloader.py"
SCHEMA = "verl/trainer/ppo/v1/tq_schema.py"
TEST = "tests/trainer/ppo/v1/test_streaming_dataloader_actor_on_cpu.py"
METHODS = {
    "_default_rollout_n", "_init_refill_policy", "_refill_to_watermark",
    "outstanding_group_count", "unfinished_session_count",
    "_coordinator_step", "_serve_request",
}
TESTS = {
    "test_session_watermark_refills_complete_prompt_groups",
    "test_outstanding_cap_limits_watermark_refill",
    "test_session_watermark_rejects_positive_value_that_rounds_to_zero",
    "test_async_wait_strategy_requires_threshold_above_one",
    "test_outstanding_cap_resolves_to_prompt_groups",
    "test_colocate_async_has_no_outstanding_cap",
    "test_outstanding_cap_must_cover_watermark",
    "test_train_request_uses_one_poll_for_maintenance_and_fulfillment",
    "test_train_eviction_refills_without_fulfilling_unready_request",
    "test_validation_request_pauses_train_inventory_maintenance",
    "test_validation_before_load_does_not_maintain_train_inventory",
    "test_materialize_failure_completes_sample_future_with_exception",
    "test_non_consuming_count_request_uses_watermark_coordinator",
    "test_non_consuming_wait_request_refills_while_below_target",
}
HELPERS = {"_async_stub", "_stub_methods", "_advance_and_maintain_watermark", "_replay_state", "_refill_policy_stub"}


def fragment(source, node):
    """Keep original text, including parameter decorators and assertions."""
    lines = source.splitlines()
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    text = "\n".join(lines[start - 1:node.end_lineno]) + "\n"
    # Methods remain indented when inserted into our intentionally incomplete class.
    return text


def own_controls(ns):
    """Hand-driven metadata transitions and a dispatch recorder, not actual consume/poll."""
    from collections import defaultdict

    actor = ns["_async_stub"]()
    for attr in ("pending_keys", "running_keys", "finished_keys", "failure_keys", "finished_session_keys", "failure_session_keys"):
        setattr(actor, attr, defaultdict(set))
    actor.default_rollout_n = 4
    actor.watermark_sessions = 8
    actor.max_outstanding_groups = 4
    requested = []
    actor._refill_exact = lambda count: requested.append(count) or count
    trace = []

    def check(label, unfinished, groups, refill):
        requested.clear()
        actual = (actor.unfinished_session_count("train", 4), actor.outstanding_group_count("train"), actor._refill_to_watermark())
        assert actual == (unfinished, groups, refill), (label, actual)
        assert requested == ([refill] if refill else [])
        trace.append(dict(label=label, unfinished_sessions=unfinished, outstanding_groups=groups, requested_prompts=refill))

    actor.running_keys["train"] = {"a", "b"}
    check("two_full_running_groups", 8, 2, 0)
    actor.finished_session_keys["train"] = {"a_0", "a_1", "b_0"}
    check("three_slots_cannot_dispatch_four_session_prompt", 5, 2, 0)
    actor.failure_session_keys["train"] = {"b_1"}
    check("terminal_failure_also_releases_session_slot", 4, 2, 1)
    actor.pending_keys["train"] = {"c"}
    check("manually_register_dispatched_prompt", 8, 3, 0)
    actor.finished_session_keys["train"].update({"b_2", "b_3"})
    actor.running_keys["train"].remove("b")
    actor.finished_keys["train"].add("b")
    check("finished_group_still_occupies_inventory", 6, 3, 0)
    actor.finished_keys["train"].add("d")
    actor.finished_session_keys["train"].add("a_2")
    actor.pending_keys["train"].clear()
    actor.finished_keys["train"].add("c")
    check("inventory_full_despite_seven_free_session_slots", 1, 4, 0)
    actor.finished_keys["train"].remove("b")
    check("manually_remove_consumed_group_reopens_cap", 1, 3, 1)
    actor.failure_session_keys["train"].add("a_2")
    check("same_terminal_key_in_both_sets_counted_once", 1, 3, 1)
    actor.finished_session_keys["train"].add("orphan_0")
    check("orphan_marker_does_not_release_running_slot", 1, 3, 1)
    actor.pending_keys["train"].add("e")
    actor.finished_session_keys["train"].update({"e_0", "e_1", "e_2", "e_3"})
    check("pending_group_reserves_full_n_despite_markers", 5, 4, 0)
    actor.finished_session_keys["train"].update({"a_3", "a_4"})
    try:
        actor.unfinished_session_count("train", 4)
    except RuntimeError as error:
        assert "inconsistent session progress" in str(error)
    else:
        raise AssertionError("excess session progress accepted")
    for n in (0, -1):
        try:
            actor.unfinished_session_count("train", n)
        except RuntimeError as error:
            assert "Invalid rollout_n" in str(error)
        else:
            raise AssertionError("nonpositive rollout_n accepted")
    # The coordinator gate is tested with recorder methods, not an event-loop actor.
    actor._pending_sample = None
    actor._train_state_ready = True
    actor._validation_active = True
    actor.poll_once = lambda *_: (_ for _ in ()).throw(AssertionError("validation idle polled train"))
    actor._coordinator_step()
    return {"metadata_transitions": trace, "nonpositive_rollout_n_rejections": 2, "excess_progress_rejection": "passed", "validation_idle_gate": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    sources = {}
    hashes = {}
    for path in (SOURCE, SCHEMA, TEST):
        raw = subprocess.check_output(["git", "-C", args.repo, "show", f"{HEAD}:{path}"], env=env)
        sources[path] = raw.decode()
        compile(sources[path], path, "exec")
        hashes[path] = hashlib.sha256(raw).hexdigest()
    tree = ast.parse(sources[SOURCE])
    definitions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    async_cls = definitions["StreamingDataloaderAsync"]
    methods = {n.name: n for n in async_cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert METHODS <= methods.keys()
    setup = """from __future__ import annotations
import asyncio
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
import types
import pytest
from omegaconf import OmegaConf
logger = logging.getLogger(__name__)
DAPO_FILTERED_REWARD_COUNTS_KEY = '_dapo_filtered_reward_counts'
SkipManager = SimpleNamespace(skip_instances={})
tq_schema = types.ModuleType('fixed_tq_schema')
"""
    setup += "exec(" + repr(sources[SCHEMA]) + ", tq_schema.__dict__)\n"
    for name in ("_accumulate_eviction_metrics", "_RequestKind", "_SampleRequest"):
        setup += fragment(sources[SOURCE], definitions[name]) + "\n"
    setup += "class StreamingDataloader: pass\nclass StreamingDataloaderAsync(StreamingDataloader):\n"
    for node in async_cls.body:
        if getattr(node, "name", None) in METHODS:
            setup += fragment(sources[SOURCE], node) + "\n"
    setup += "streaming_module = SimpleNamespace(_SampleRequest=_SampleRequest, _RequestKind=_RequestKind)\n"
    test_nodes = {n.name: n for n in ast.parse(sources[TEST]).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert (TESTS | HELPERS) <= test_nodes.keys()
    for name, node in test_nodes.items():
        if name in TESTS | HELPERS:
            setup += fragment(sources[TEST], node) + "\n"
    # All selected production methods and original decorated tests retain their AST.
    generated_tree = ast.parse(setup)
    generated_defs = {getattr(n, "name", None): n for n in generated_tree.body}
    for name in TESTS | HELPERS:
        assert ast.dump(generated_defs[name]) == ast.dump(test_nodes[name])
    gen_methods = {getattr(n, "name", None): n for n in generated_defs["StreamingDataloaderAsync"].body}
    for name in METHODS:
        assert ast.dump(gen_methods[name]) == ast.dump(methods[name])
    ns = {"__name__": "watermark_controls"}
    # Dataclass needs its module registered. No verl, Ray, Torch or TQ import occurs.
    import types
    module = types.ModuleType(ns["__name__"])
    sys.modules[module.__name__] = module
    exec(compile(setup, "<original-control-slice>", "exec"), module.__dict__)
    controls = own_controls(module.__dict__)
    with tempfile.TemporaryDirectory(prefix="verl-watermark-") as tmp:
        path = Path(tmp) / "test_selected.py"
        path.write_text(setup)
        run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(path)], env=env, capture_output=True, text=True, timeout=45)
    report = {
        "head": HEAD, "base_and_merge_base": BASE,
        "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("pytest", "omegaconf")}},
        "source_sha256": hashes,
        "original_scope": {"methods": sorted(METHODS), "test_functions": sorted(TESTS), "tests_and_methods_ast_unchanged": True},
        "pytest": {"returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr},
        "own_controls": controls,
        "seams": ["empty base class; actor constructor not executed", "SkipManager.skip_instances empty", "original tests use SimpleNamespace/OmegaConf and recorder methods", "own transitions mutate in-memory sets; _refill_exact records only"],
        "not_executed": ["native verl module or full author suite", "Ray/TransferQueue/storage/dataloader", "metadata polling, dispatch or actual group consumption", "AgentLoop marker publication and trajectory write settlement", "checkpoint/reissue/off-policy filtering", "model, devices, performance or training quality"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if run.returncode:
        raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
