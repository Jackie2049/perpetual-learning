"""Six controls of the original resume method; fake workers, real trusted temporary data.pt files."""

import ast
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

import torch

REV = "c7e22a9e0ecbc08bc3600c05b0be717203fe7d1a"
PATHS = ["verl/trainer/ppo/v1/trainer_base.py", "verl/utils/dataset/dataloader.py",
         "tests/utils/dataset/test_custom_dataloader_on_cpu.py"]
trees, hashes = {}, {}
for path in PATHS:
    raw = subprocess.check_output(["git", "show", f"{REV}:{path}"], cwd=Path(sys.argv[1]),
                                  env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    hashes[path] = hashlib.sha256(raw).hexdigest()
    trees[path] = ast.parse(raw, filename=path)
    compile(trees[path], path, "exec")
cls = next(n for n in trees[PATHS[0]].body if isinstance(n, ast.ClassDef) and n.name == "PPOTrainer")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_load_checkpoint")
assert not method.decorator_list
find_latest = Mock()
scope = {"os": os, "torch": torch, "logger": logging.getLogger(__name__),
         "find_latest_ckpt_path": find_latest, "Role": SimpleNamespace(Critic="critic")}
exec(compile(ast.Module(body=[method], type_ignores=[]), "original-load-checkpoint", "exec"), scope)
results = []


class LoaderFixture:
    def __iter__(self):
        return iter(())


cases = [("nonstateful_auto_found", False, "auto", True, None, "ValueError", False),
         ("nonstateful_explicit", False, "resume_path", True, None, "ValueError", False),
         ("nonstateful_auto_fresh", False, "auto", False, None, None, False),
         ("stateful_missing_data", True, "auto", True, None, "FileNotFoundError", True),
         ("stateful_invalid_data", True, "auto", True, [3], "TypeError", True),
         ("stateful_valid_data", True, "auto", True, {"cursor": 3}, None, True)]
for label, stateful, mode, found, payload, expected_error, weights_called in cases:
    with tempfile.TemporaryDirectory(prefix="verl-loader-resume-") as temporary:
        root = Path(temporary)
        checkpoint = root / "global_step_7"
        checkpoint.mkdir()
        if payload is not None:
            # Self-generated primitive payload only; never load external pickle.
            torch.save(payload, checkpoint / "data.pt")
        find_latest.return_value = str(checkpoint) if found else None
        loader = LoaderFixture()
        if stateful:
            loader.load_state_dict = Mock()
            loader.state_dict = Mock(return_value={})
        trainer = SimpleNamespace(config=SimpleNamespace(
            trainer=SimpleNamespace(resume_mode=mode, default_local_dir=str(root), resume_from_path=str(checkpoint),
                                    del_local_ckpt_after_load=False), data={"custom_dataloader": {"_target_": "fixture"}}),
            train_dataloader=loader, actor_rollout_wg=Mock(), critic_wg=Mock(), use_critic=True, trainer_mode="sync")
        error = None
        try:
            scope["_load_checkpoint"](trainer)
        except (ValueError, FileNotFoundError, TypeError) as failure:
            error = type(failure).__name__
        assert error == expected_error
        assert trainer.actor_rollout_wg.load_checkpoint.called == weights_called
        assert trainer.critic_wg.load_checkpoint.called == weights_called
        assert trainer.global_steps == (7 if weights_called else 0)
        states = [call.args[0] for call in loader.load_state_dict.call_args_list] if stateful else []
        assert states == ([payload] if label == "stateful_valid_data" else [])
        results.append({"case": label, "error": error, "actor_load_called": weights_called,
                        "critic_load_called": weights_called, "global_steps": trainer.global_steps,
                        "loader_states": states})
assert "verl" not in sys.modules and "ray" not in sys.modules
print(json.dumps({"revision": REV, "python": platform.python_version(), "torch": torch.__version__,
                  "source_sha256": hashes, "local_controls_passed": len(results), "results": results,
                  "scope": "Original complete _load_checkpoint only; three full sources syntax-checked. Six locally designed controls, not original PR tests/native pytest. Config/loader/latest discovery/workers are explicit fixtures; self-written data.pt uses real torch.save/load. No model weights, optimizer, real stream, Hydra construction, saving-method execution, rollback, Ray/TransferQueue, GPU/NPU or training validation."}, indent=2))
