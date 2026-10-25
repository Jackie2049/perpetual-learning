"""Compose fixed upstream YAMLs with real Hydra without importing the trainer.

Own integration controls, not #7426's original reflective pytest suite.
Upstream definitions: verl-project/verl, Apache-2.0, ByteDance contributors.
Package shells expose original BaseConfig, CheckpointConfig and historical
McoreCheckpointConfig only; other target classes are not imported.
"""

import argparse
import ast
from dataclasses import asdict, dataclass, field, fields
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, Optional


VERSIONS = {
    "ancestor_7426": "ea53291385ce764019a2b40733605f21d8317583",
    "actor_7388": "0e540bc35c234008e7c5e3b80a148582ae860c58",
    "all_7426": "51f8f311294b2732e04c736a3b9eb11d966a5eeb",
    "main": "3c6d9b43b67c74ec343d77874cd8fa6e841523da",
}
CONFIG = "verl/trainer/config"
ROOTS = {"ppo_trainer": ["actor_rollout_ref.actor.checkpoint", "critic.checkpoint"],
         "ppo_megatron_trainer": ["actor_rollout_ref.actor.checkpoint", "critic.checkpoint"],
         "sft_trainer_engine": ["checkpoint"]}
OVERRIDES = {"ppo_trainer": "actor_rollout_ref.actor.checkpoint.save_lora_only=True",
             "ppo_megatron_trainer": "critic.checkpoint.strict=False",
             "sft_trainer_engine": "checkpoint.async_save=True"}


class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "ray", "vllm", "sglang", "transformers"}:
            raise ImportError("Out-of-scope framework import: " + fullname)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, BlockFrameworks())
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.errors import ConfigCompositionException
    from omegaconf import DictConfig, ListConfig, OmegaConf
    import omegaconf

    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")
    hashes, results = {}, {}

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=args.source_repo, env=env)

    def source(revision, path):
        raw = git("show", revision + ":" + path)
        hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
        return raw

    def definition(revision, path, name, namespace):
        tree = ast.parse(source(revision, path), filename=path)
        compile(tree, path, "exec")
        nodes = [node for node in tree.body if getattr(node, "name", None) == name]
        assert len(nodes) == 1
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)

    with tempfile.TemporaryDirectory(prefix="perpetual-checkpoint-config-") as directory:
        for label, revision in VERSIONS.items():
            root = Path(directory) / label
            root.mkdir()
            archive = git("archive", revision, CONFIG)
            subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
            manifest = git("ls-tree", "-r", revision, "--", CONFIG)
            manifest_hash = hashlib.sha256(manifest).hexdigest()

            # No original package initializers or other target classes are loaded.
            for name in ("verl", "verl.trainer", "verl.trainer.config", "verl.base_config"):
                module = ModuleType(name)
                module.__path__ = []
                sys.modules[name] = module
                if "." in name:
                    parent, child = name.rsplit(".", 1)
                    setattr(sys.modules[parent], child, module)
            base = sys.modules["verl.base_config"]
            exec(compile(source(revision, "verl/base_config.py"), "base_config.py", "exec"), base.__dict__)
            module = sys.modules["verl.trainer.config"]
            module.__dict__.update(BaseConfig=base.BaseConfig, dataclass=dataclass, field=field)
            definition(revision, CONFIG + "/config.py", "CheckpointConfig", module.__dict__)
            converter = {"DictConfig": DictConfig, "ListConfig": ListConfig, "OmegaConf": OmegaConf,
                         "Any": Any, "Optional": Optional}
            definition(revision, "verl/utils/config.py", "omega_conf_to_dataclass", converter)
            workers = ModuleType("verl.workers")
            workers.__path__ = []
            sys.modules["verl.workers"] = workers
            sys.modules["verl"].workers = workers
            worker_config = ModuleType("verl.workers.config")
            sys.modules["verl.workers.config"] = worker_config
            workers.config = worker_config
            worker_config.__dict__.update(CheckpointConfig=module.CheckpointConfig, dataclass=dataclass,
                                          field=field, Any=Any)
            # Historical Megatron roots have a real backend-specific subclass.
            if label != "main":
                definition(revision, "verl/workers/config/checkpoint.py", "McoreCheckpointConfig",
                           worker_config.__dict__)
            records = {}
            for config_name, paths in ROOTS.items():
                with initialize_config_dir(config_dir=str(root / CONFIG), version_base=None):
                    config = compose(config_name=config_name)
                blocks = {}
                for path in paths:
                    node = OmegaConf.select(config, path)
                    assert node._target_ in {"verl.trainer.config.CheckpointConfig",
                                             "verl.workers.config.McoreCheckpointConfig"}
                    instance = converter["omega_conf_to_dataclass"](node)
                    assert isinstance(instance, module.CheckpointConfig)
                    expected_fields = {f.name for f in fields(type(instance))}
                    blocks[path] = {"target": node._target_, "missing": sorted(expected_fields - set(node)),
                                    "constructed": asdict(instance)}
                override = OVERRIDES[config_name]
                try:
                    with initialize_config_dir(config_dir=str(root / CONFIG), version_base=None):
                        changed = compose(config_name=config_name, overrides=[override])
                except ConfigCompositionException as error:
                    override_result = {"accepted": False, "error": type(error).__name__,
                                       "cause": type(error.__cause__).__name__}
                else:
                    key, value = override.split("=")
                    observed = OmegaConf.select(changed, key)
                    assert observed is (value == "True")
                    override_result = {"accepted": True, "value": observed}
                records[config_name] = {"blocks": blocks, "override": override,
                                        "override_result": override_result}
            results[label] = {"config_tree_manifest_sha256": manifest_hash,
                              "config_tree_files": len(manifest.splitlines()), "configs": records}

    for label, result in results.items():
        flags = [r["override_result"]["accepted"] for r in result["configs"].values()]
        expected = [True, True, True] if label == "all_7426" else (
            [True, False, False] if label == "actor_7388" else [False, False, False])
        assert flags == expected, (label, flags)
        for config_name, record in result["configs"].items():
            for path, block in record["blocks"].items():
                if label == "all_7426":
                    reference = results["ancestor_7426"]["configs"][config_name]["blocks"][path]
                    assert block["constructed"] == reference["constructed"]
                    assert not block["missing"]
    assert not any(name in sys.modules for name in ("torch", "ray", "vllm", "sglang", "transformers"))
    print(json.dumps({"versions": VERSIONS, "source_sha256": hashes, "results": results,
                      "default_constructed_values_equal_7426_and_parent": True,
                      "environment": {"python": platform.python_version(), "hydra": hydra.__version__,
                                      "omegaconf": omegaconf.__version__, "system": platform.system()},
                      "scope": "own real-Hydra composition controls, five explicit checkpoint nodes; "
                               "original classes/converter only; no original reflective pytest, "
                               "full resolve, all-subclass discovery, trainer, backend or checkpoint IO"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
