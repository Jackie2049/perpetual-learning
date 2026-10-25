"""Check original #468 environment helpers with an isolated dictionary, not a process environment."""
import __future__
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from types import SimpleNamespace


VERSIONS = {"base": "f989c03825f187607bcbe6eaf6178909b798b43f",
            "head": "3dac11fbdf691dde142c74b0db0a6556a0d52966",
            "main": "6fd82cba13f52f394747832344039408699ea378"}
PREFIX = "unirl/rollout/engine/sglang/"
FILES = [PREFIX + suffix for suffix in ("backends/base.py", "backends/http.py", "backends/native.py", "engine.py")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    process_environment = dict(os.environ)
    hashes, trees = {}, {}
    for label, revision in VERSIONS.items():
        for path in FILES:
            raw = subprocess.check_output(["git", "show", revision + ":" + path], cwd=args.source_repo,
                                          env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
            hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
            tree = ast.parse(raw, filename=path)
            compile(tree, path, "exec")
            trees[label, path] = tree
    names = {"_split_preloads", "_scheduler_spawn_environment"}
    for label in ("base", "main"):
        assert not any(getattr(node, "name", None) in names for node in trees[label, FILES[0]].body)
    nodes = [node for node in trees["head", FILES[0]].body if getattr(node, "name", None) in names]
    assert len(nodes) == 2
    env = {"LD_LIBRARY_PATH": "/legacy:/active/lib", "CUDA_VISIBLE_DEVICES": "parent"}
    original = env.copy()
    ns = {"os": SimpleNamespace(environ=env, pathsep=":"), "re": re, "contextmanager": contextmanager,
          "_preloaded_cuda_driver_dirs": lambda: [], "_python_cuda_library_dirs": lambda: ["/active/lib"]}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "original_helpers", "exec",
                 flags=__future__.annotations.compiler_flag), ns)
    parsed = ns["_split_preloads"]("/compat/libcuda.so.1 : /hook/libtorch_memory_saver.so\t")
    assert parsed == ["/compat/libcuda.so.1", "/hook/libtorch_memory_saver.so"]
    inside = None
    try:
        with ns["_scheduler_spawn_environment"](["3", "5"]):
            inside = env.copy()
            assert env == {"LD_LIBRARY_PATH": "/active/lib:/legacy", "CUDA_VISIBLE_DEVICES": "3,5"}
            raise RuntimeError("local-body-failure")
    except RuntimeError as exc:
        assert str(exc) == "local-body-failure"
    assert env == original
    env.clear()
    with ns["_scheduler_spawn_environment"](None):
        assert env == {"LD_LIBRARY_PATH": "/active/lib"}
    assert env == {}
    assert dict(os.environ) == process_environment
    assert "unirl" not in sys.modules and "sglang" not in sys.modules and "torch_memory_saver" not in sys.modules
    print(json.dumps({"versions": VERSIONS, "source_sha256": hashes,
                      "results": {"parsed_preloads": parsed, "no_driver_context": inside,
                                  "exception_restores_prior_values": True, "normal_exit_restores_absent_keys": True,
                                  "real_process_environment_unchanged": True},
                      "environment": {"python": platform.python_version(), "system": platform.system()},
                      "scope": "two original head helper bodies; local controls, not upstream tests; "
                               "isolated dict plus directory-provider fixtures; all 12 modules syntax checked; "
                               "no real env mutation, scheduler spawn, library loading, TMS wrapper/hook runtime, "
                               "SGLang/Ray/engine/driver/GPU or dependency installation"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
