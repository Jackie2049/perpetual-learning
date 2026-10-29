"""Fixed-source routing audit and two original sleep-resolver CPU controls.

No trainer, replica, Ray, engine or config conversion is executed. Inputs to the
original resolver are SimpleNamespace fixtures; NPU capability is explicitly
False, not discovered. Original RolloutMode enum and resolver body are unchanged.
Original verl sources are Apache-2.0, Bytedance Ltd. and/or affiliates.
"""

import argparse
import ast
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace


REV = "b7fca1bf72c9ef488595af2ade0b28f95ad00e5e"
PATHS = ["verl/trainer/ppo/v1/trainer_base.py", "verl/checkpoint_engine/base.py",
         "verl/workers/rollout/llm_server.py", "verl/workers/rollout/replica.py",
         "verl/workers/rollout/vllm_rollout/vllm_async_server.py", "verl/utils/config.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    sources, hashes, trees = {}, {}, {}
    for path in PATHS:
        raw = subprocess.check_output(["git", "show", REV + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        sources[path] = raw.decode()
        hashes[path] = hashlib.sha256(raw).hexdigest()
        trees[path] = ast.parse(raw)
        compile(trees[path], path, "exec")  # Syntax only, no module execution.

    def definition(path, owner, name):
        cls = next(node for node in trees[path].body if isinstance(node, ast.ClassDef) and node.name == owner)
        return next(node for node in cls.body if getattr(node, "name", None) == name)

    replica_path, server_path = PATHS[3:5]
    enum = next(node for node in trees[replica_path].body if isinstance(node, ast.ClassDef) and node.name == "RolloutMode")
    resolver = definition(server_path, "vLLMHttpServer", "_resolve_sleep_level")
    ns = {"Enum": Enum, "is_torch_npu_available": lambda *, check_device: False}
    exec(compile(ast.Module(body=[enum, resolver], type_ignores=[]), "original-resolver.py", "exec"), ns)
    observations = {}
    for name, expected in [("HYBRID", 2), ("COLOCATED", 1)]:
        server = SimpleNamespace(config=SimpleNamespace(checkpoint_engine=SimpleNamespace(backend="nccl"), mtp=None),
                                 model_config=SimpleNamespace(lora={"resync_base": True}), lora_as_adapter=True,
                                 rollout_mode=getattr(ns["RolloutMode"], name))
        level = ns["_resolve_sleep_level"](server)
        assert level == expected
        observations[name] = {"configured_backend": "nccl", "resync_base": True,
                              "mtp_enabled": False, "npu_capability_fixture": False, "level": level}

    setup = definition(PATHS[0], "PPOTrainer", "_setup")
    manager_init = definition(PATHS[2], "LLMServerManager", "_initialize_llm_servers")
    route = next(node for node in manager_init.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "self.worker_group and self.rollout_config.name != 'trtllm'")
    assert "init_hybrid(self.worker_group)" in ast.unparse(route.body[0])
    init_hybrid = definition(replica_path, "RolloutReplica", "init_hybrid")
    assert "self.rollout_mode = RolloutMode.HYBRID" in ast.unparse(init_hybrid)
    manager_statements = [node for node in setup.body if "checkpoint_engine_config" in ast.unparse(node)]
    assert len(manager_statements) == 3
    manager_update = definition(PATHS[1], "CheckpointEngineManager", "update_weights")
    dispatch = next(node for node in manager_update.body if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "self.backend == 'naive'")
    assert "mode=self.backend" in ast.unparse(dispatch)
    assert "torch" not in sys.modules and "ray" not in sys.modules
    print(json.dumps({"revision": REV, "python": platform.python_version(), "source_sha256": hashes,
                      "static_source_facts_not_runtime": {
                          "manager_config_creation": "\n".join(ast.get_source_segment(sources[PATHS[0]], node) for node in manager_statements),
                          "replica_mode_selection": ast.get_source_segment(sources[PATHS[2]], route),
                          "manager_dispatch": ast.get_source_segment(sources[PATHS[1]], dispatch)},
                      "original_resolver_fixture_results": observations,
                      "scope": "Source wiring plus original Enum/resolver only; no native config conversion, trainer/actor/Ray construction, sleep, weight sync, GPU memory or performance validation"}, indent=2))


if __name__ == "__main__":
    main()
