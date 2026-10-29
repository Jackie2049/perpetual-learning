"""Run #8032's unmodified eight CPU regression cases on head/direct parent.

Original test exports only the production async method with decorators removed;
its engine, rollout, torch and allocator are the author's mocks. No GPU/Ray,
backend installation, memory measurement or real weight transfer occurs.
Git checkout-index copies original inputs via a temporary, independent index.
Original verl files: Apache-2.0, Bytedance Ltd. and/or affiliates, 2026.
"""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import runpy
import subprocess
import sys
import tempfile

import pytest


HEAD = "b7fca1bf72c9ef488595af2ade0b28f95ad00e5e"
PARENT = "c0b1c7c3d9b4031fa1d8082d3afddfee43212164"
WORKER = "verl/workers/engine_workers.py"
TEST = "tests/workers/test_layered_lora_base_sync_on_cpu.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTEST_ADDOPTS="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    hashes = {}

    def git(*command, git_env=env):
        return subprocess.check_output(["git", *command], cwd=args.source_repo, env=git_env)

    for rev, path in [(HEAD, TEST), (HEAD, WORKER), (PARENT, WORKER)]:
        hashes[rev + ":" + path] = hashlib.sha256(git("show", rev + ":" + path)).hexdigest()
    assert git("rev-parse", HEAD + "^").decode().strip() == PARENT
    cases = {}
    probe = None
    for label, revision in [("parent", PARENT), ("head", HEAD)]:
        with tempfile.TemporaryDirectory(prefix="pl-layered-base-") as temporary:
            root = Path(temporary) / "snapshot"
            root.mkdir()
            git_env = dict(env, GIT_INDEX_FILE=str(Path(temporary) / "index"))
            for rev, path in [(HEAD, TEST), (revision, WORKER)]:
                git("read-tree", rev, git_env=git_env)
                git("checkout-index", "--prefix=" + str(root) + os.sep, "--", path, git_env=git_env)
                copied = (root / path).read_bytes()
                assert hashlib.sha256(copied).hexdigest() == hashes[rev + ":" + path]
            run = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "--tb=short", "-r", "f",
                                  "--color=no", "-p", "no:cacheprovider", "-c", "/dev/null",
                                  "--rootdir", str(root), TEST],
                                 cwd=root, env=env, capture_output=True, text=True, timeout=40)
            expected = 1 if label == "parent" else 0
            assert run.returncode == expected, run.stdout + run.stderr
            expected_summary = "3 failed, 5 passed" if label == "parent" else "8 passed"
            assert expected_summary in run.stdout, run.stdout
            if label == "parent":
                assert run.stdout.count("DID NOT RAISE") == 3, run.stdout
            # Keep native test names/status; normalize ephemeral path and elapsed time.
            output = re.sub(r" in [0-9.]+s", "", run.stdout.strip())
            output = output.replace(str(root), "<snapshot>")
            output = re.sub(r"(FAILED )[^:\n]+::", r"\1<snapshot>::", output)
            cases[label] = {"worker_revision": revision, "test_revision": HEAD,
                            "exit_code": run.returncode, "stdout": output, "stderr": run.stderr}
            if label == "head":
                # One additional control: the same author's fixture, with free-cache
                # enabled, exposes operations already done before the guard raises.
                module = runpy.run_path(str(root / TEST))
                update = module["update_weights"].__wrapped__()
                worker = module["_worker"]()
                worker.config.rollout.free_cache_engine = True
                try:
                    asyncio.run(update(worker))
                except ValueError as error:
                    message = str(error)
                else:
                    raise AssertionError("guard did not reject")
                allocator = update.__globals__["set_expandable_segments"]
                assert allocator.call_args_list == [module["call"](False)]
                assert worker.rollout.resume.await_args_list == [module["call"](tags=["weights"])]
                worker.rollout.update_weights.assert_not_awaited()
                assert worker.base_sync_done is True and worker.rollout.sleep_level == 2
                probe = {"input": "author _worker defaults, free_cache_engine=True",
                         "exception": message, "allocator_switch_calls": [False],
                         "resume_tags": [["weights"]], "weight_transfer_calls": 0,
                         "base_sync_done": worker.base_sync_done,
                         "rollout_sleep_level_after_rejection": worker.rollout.sleep_level}
    assert "torch" not in sys.modules and "ray" not in sys.modules
    print(json.dumps({"head": HEAD, "direct_parent": PARENT, "python": platform.python_version(),
                      "pytest": pytest.__version__, "source_sha256": hashes, "original_test_cases": cases,
                      "additional_guard_boundary_control": probe,
                      "scope": "Native pytest of original mock-based CPU cases; no production imports, real base export/transfer, allocator, GPU memory, recovery or performance validation"}, indent=2))


if __name__ == "__main__":
    main()
