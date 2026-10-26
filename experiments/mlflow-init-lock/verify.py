"""Bounded original-class control with real cross-process flock, not MLflow.

Usage: python verify.py /path/to/pinned/rollout_trace.py [--revision revised]
The MLflow callback is an explicit recorder/exception injector. No database,
Alembic migration, Ray actor, GPU, or tracing server is started.
"""
import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Optional
from unittest.mock import patch

COMMIT = "8313d7b6961f399609e3aae34ef739c0cac55cb5"
SOURCE_PATH = "verl/utils/rollout_trace.py"
EXPECTED_SHA256 = "6c942a3bb32787f72f0875c6f848095b852d01efaa5d5b96009668c9c748d7ad"
REVISED_COMMIT = "f1dd14b0a843570b2542568678ef286d81aaed7f"
REVISED_SHA256 = "bf795f42a0bb271a4504e4e075563bf1874a4f8e87303ed96abe89863834d2aa"

# A new interpreter opens its own descriptor; errno is not guessed from timing.
PROBE = """
import errno, fcntl, sys
with open(sys.argv[1], 'a') as fd:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EAGAIN):
            raise
        print('blocked')
    else:
        print('acquired')
        fcntl.flock(fd, fcntl.LOCK_UN)
"""


def probe(lock_path):
    return subprocess.run([sys.executable, "-c", PROBE, str(lock_path)],
                          text=True, capture_output=True, check=True,
                          timeout=5).stdout.strip()


def load_class(source):
    tree = ast.parse(source)
    compile(tree, SOURCE_PATH, "exec")
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "RolloutTraceConfig")
    namespace = {"os": os, "Optional": Optional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), SOURCE_PATH, "exec"), namespace)
    return namespace["RolloutTraceConfig"]


def run_case(source, root, mode):
    config_class = load_class(source)
    lock_path = root / (mode + ".db.lock")
    observations = []

    def callback(project):
        observations.append({"project": project, "other_process": probe(lock_path)})
        if len(observations) == 1:
            if mode == "runtime_error":
                raise RuntimeError("injected backend failure")
            if mode == "import_error":
                raise ImportError("injected backend import failure")

    mlflow = types.ModuleType("mlflow")
    mlflow.config = types.SimpleNamespace(enable_async_logging=lambda: None)
    mlflow.set_tracking_uri = lambda uri: None
    mlflow.set_experiment = callback
    error = None
    uri = "sqlite:///" + str(root / (mode + ".db"))
    with patch.dict(sys.modules, {"mlflow": mlflow}), patch.dict(os.environ, {"MLFLOW_TRACKING_URI": uri}):
        try:
            config_class.init("project", "experiment", "mlflow")
        except RuntimeError as exc:
            error = type(exc).__name__
        if mode == "success":
            config_class.init("project", "experiment", "mlflow")
    return {"calls": observations, "error": error,
            "initialized": config_class.get_instance()._initialized,
            "after_return": probe(lock_path), "lock_file_remains": lock_path.exists()}


def run(source_file, revision="original"):
    assert revision in ("original", "revised")
    commit, expected_hash = ((COMMIT, EXPECTED_SHA256) if revision == "original"
                             else (REVISED_COMMIT, REVISED_SHA256))
    source = source_file.read_bytes()
    assert hashlib.sha256(source).hexdigest() == expected_hash
    with tempfile.TemporaryDirectory(prefix="mlflow-init-lock-") as directory:
        cases = {mode: run_case(source, Path(directory), mode)
                 for mode in ("success", "runtime_error", "import_error")}
    assert [x["other_process"] for x in cases["success"]["calls"]] == ["blocked"]
    assert cases["success"]["initialized"]
    assert [x["other_process"] for x in cases["runtime_error"]["calls"]] == ["blocked"]
    assert cases["runtime_error"]["error"] == "RuntimeError" and not cases["runtime_error"]["initialized"]
    assert [x["other_process"] for x in cases["import_error"]["calls"]] == ["blocked", "acquired"]
    assert cases["import_error"]["initialized"]
    assert all(case["after_return"] == "acquired" for case in cases.values())
    return {"commit": commit, "source_path": SOURCE_PATH, "sha256": expected_hash,
            "python": platform.python_version(), "platform": platform.system(),
            "scope": "original AST class, recorder MLflow callback, real fcntl and child lock probes; no MLflow/Alembic/database/Ray execution",
            "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_file", type=Path)
    parser.add_argument("--revision", choices=("original", "revised"), default="original")
    args = parser.parse_args()
    print(json.dumps(run(args.source_file, args.revision), indent=2))
