"""Offline CPU call-boundary controls for verl #7484, not an optimizer test.

Extract complete original wrapper/_build_optimizer methods from fixed Git blobs.
Bridge v0.6.2 source is inspected, not imported. Native optimizer, Bridge helper,
config constructor and training hooks are explicit recording/availability doubles.
No torch/Megatron/Bridge installation, model, parameter groups or device access.
Original verl and NVIDIA Bridge sources: Apache-2.0, linked by SHA in the output.
"""

import argparse
import ast
import builtins
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from types import ModuleType, SimpleNamespace


BASE = "a08ea02a784d7d4982b1d81df36d188a476192bb"
HEAD = "cf96a3a64978e6bb16f19ed6d55602d96a34fe14"
BRIDGE = "c0e164ed2aedac4ad1c877780e2564a19d5d54ec"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("bridge_lora_source", type=Path)
    args = parser.parse_args()
    hashes = {}

    def source(revision, path):
        raw = subprocess.check_output(["git", "show", revision + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    def function(raw, name, namespace, class_name=None):
        tree = ast.parse(raw)
        if class_name:
            tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        # Deferred annotations only; decorators/bodies of these two functions stay original.
        unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
                          type_ignores=[])
        ast.fix_missing_locations(unit)
        exec(compile(unit, "fixed-original.py", "exec"), namespace)
        return namespace[name]

    wrappers = {label: source(rev, "verl/utils/megatron/optimizer.py")
                for label, rev in [("base", BASE), ("head", HEAD)]}
    engine_source = source(HEAD, "verl/workers/engine/megatron/transformer_impl.py")
    project = source(HEAD, "pyproject.toml")
    yaml = source(HEAD, "verl/trainer/config/model/hf_model.yaml")
    assert '"megatron-bridge==0.6.2"' in project and "lora_plus_ratio: 1.0" in yaml
    bridge_raw = args.bridge_lora_source.read_bytes()
    assert hashlib.sha256(bridge_raw).hexdigest() == "3085edc165526c5c54f46f39a57dca2717beaaf52a4b82f757a1e7a3caa6cdac"
    hashes[BRIDGE + ":src/megatron/bridge/peft/lora.py"] = hashlib.sha256(bridge_raw).hexdigest()
    bridge_tree = ast.parse(bridge_raw)
    assert b"get_lora_plus_config_overrides" not in bridge_raw
    assert not any(isinstance(node, ast.FunctionDef) and node.name == "__getattr__" for node in bridge_tree.body)
    assert not any(isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
                   for node in bridge_tree.body)
    cases = {}

    def wrapper_case(name, label, ratio, helper_available=False, legacy_native=False):
        events = []
        bridge_module = ModuleType("capability_fixture")

        def helper(config, value):
            events.append({"helper_ratio": value, "same_config": config is config_object})
            return override_marker

        if helper_available:
            bridge_module.get_lora_plus_config_overrides = helper

        def controlled_import(module, globals=None, locals=None, fromlist=(), level=0):
            if module == "__future__":
                return builtins.__import__(module, globals, locals, fromlist, level)
            events.append({"import": module, "fromlist": list(fromlist)})
            assert module == "megatron.bridge.peft.lora"
            return bridge_module

        def native(*, config, model_chunks, **kwargs):
            events.append({"native_keys": sorted(kwargs), "same_config": config is config_object,
                           "same_model": model_chunks is model_marker,
                           "same_overrides": kwargs.get("config_overrides") is override_marker})
            return optimizer_marker

        def legacy(*, config, model_chunks):
            return native(config=config, model_chunks=model_chunks)

        config_object = SimpleNamespace(lr=0.001)
        override_marker, model_marker, optimizer_marker = object(), object(), object()
        namespace = {"__builtins__": dict(vars(builtins), __import__=controlled_import),
                     "get_megatron_optimizer_native": legacy if legacy_native else native,
                     "print_rank_0": lambda message: events.append({"log": message})}
        fn = function(wrappers[label], "get_megatron_optimizer", namespace)
        kwargs = {} if ratio is None else {"lora_plus_ratio": ratio}
        try:
            returned = fn(model_marker, config_object, **kwargs)
            outcome = "returned_optimizer_marker"
            assert returned is optimizer_marker
        except (ImportError, TypeError) as error:
            outcome = type(error).__name__
        cases[name] = {"ratio": ratio, "helper_available_fixture": helper_available,
                       "legacy_native_signature_fixture": legacy_native, "outcome": outcome, "events": events}
        return outcome, events

    for name, label, ratio in [("base_default", "base", None), ("head_default", "head", None),
                               ("head_zero", "head", 0.0)]:
        outcome, events = wrapper_case(name, label, ratio, legacy_native=True)
        assert outcome == "returned_optimizer_marker" and len(events) == 1
        assert events[0]["native_keys"] == []
    outcome, events = wrapper_case("head_opt_in_missing_helper", "head", 16.0)
    assert outcome == "ImportError" and len(events) == 1
    outcome, events = wrapper_case("head_opt_in_forwarding", "head", 16.0, helper_available=True)
    assert outcome == "returned_optimizer_marker" and events[-1]["same_overrides"]
    outcome, events = wrapper_case("head_opt_in_legacy_native", "head", 16.0,
                                   helper_available=True, legacy_native=True)
    assert outcome == "TypeError" and not any("native_keys" in event for event in events)

    for name, lora, expected in [("engine_default", {}, 1.0), ("engine_string_ratio", {"lora_plus_ratio": "16"}, 16.0)]:
        events = []
        module = ModuleType("optimizer_fixture")
        config_marker, optimizer_marker = object(), object()

        def init_config(config, **kwargs):
            events.append({"config_kwargs": kwargs})
            return config_marker

        def get_optimizer(*, model, config, lora_plus_ratio):
            assert config is config_marker
            events.append({"forwarded_ratio": lora_plus_ratio, "ratio_type": type(lora_plus_ratio).__name__})
            return optimizer_marker

        def import_optimizer(module_name, globals=None, locals=None, fromlist=(), level=0):
            if module_name == "__future__":
                return builtins.__import__(module_name, globals, locals, fromlist, level)
            assert module_name == "verl.utils.megatron.optimizer"
            return module

        module.init_megatron_optim_config = init_config
        module.get_megatron_optimizer = get_optimizer
        torch_fixture = SimpleNamespace(float16="fp16", bfloat16="bf16")
        namespace = {"__builtins__": dict(vars(builtins), __import__=import_optimizer), "torch": torch_fixture,
                     "register_megatron_training_hooks": lambda model, opt: events.append({"hook_optimizer": opt is optimizer_marker})}
        fn = function(engine_source, "_build_optimizer", namespace, "MegatronEngine")
        engine = SimpleNamespace(model_config=SimpleNamespace(lora=lora), module=object(), optimizer_config=object(),
                                 engine_config=SimpleNamespace(use_distributed_optimizer=True), param_dtype="bf16")
        assert fn(engine) is optimizer_marker
        assert events[1]["forwarded_ratio"] == expected and events[1]["ratio_type"] == "float"
        assert events[-1]["hook_optimizer"]
        cases[name] = {"config_input": lora, "events": events}
    print(json.dumps({"base": BASE, "head": HEAD, "bridge_v0_6_2": BRIDGE,
                      "python": platform.python_version(), "source_sha256": hashes,
                      "bridge_helper_absent_in_fixed_source": True, "cases": cases,
                      "scope": "Original wrapper/method call boundaries with explicit doubles; not actual Bridge import, native optimizer, parameter groups, LR schedule, model, training or device validation"}, indent=2))


if __name__ == "__main__":
    main()
