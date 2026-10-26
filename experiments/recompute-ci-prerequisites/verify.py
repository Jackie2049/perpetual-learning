#!/usr/bin/env python3
"""Check collection of original verl #8338 tests with Megatron absent.

Provide unmodified test files from c099b017 (old) and b028abee (new).
Hashes are enforced. No framework patch, test body, device discovery, model,
installation or network call is run. Refuse this check if Megatron is installed.
The source files must be isolated copies, outside the verl checkout, so its
conftest and configured plugins cannot change this narrow collection control.
Print raw pytest results as JSON; a nonzero expected child exit is not a pass
of the upstream GPU regression tests.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HASHES = {
    "old": "d046908de4585d0efb24063b911d570058a9a4f3617fa980998e27ad836fbc49",
    "new": "1c6240c843152e46ed690a36c379d2cc9fab5cf817a6a8f368a921f9148bcaa6",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-test", type=Path, required=True)
    parser.add_argument("--new-test", type=Path, required=True)
    args = parser.parse_args()
    if importlib.util.find_spec("megatron") is not None:
        parser.error("Megatron is installed: refusing to proceed toward CUDA tests")
    import pytest
    import torch

    paths = {"old": args.old_test.resolve(), "new": args.new_test.resolve()}
    for tag, path in paths.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != HASHES[tag]:
            parser.error(f"{tag} source hash mismatch: {digest}")
    results = []
    for tag, flag, expected in [("old", None, 5), ("new", None, 5), ("new", "1", 2)]:
        path = paths[tag]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        env.pop("VERL_REQUIRE_MCORE_TESTS", None)
        if flag is not None:
            env["VERL_REQUIRE_MCORE_TESTS"] = flag
        command = [sys.executable, "-B", "-m", "pytest", "-c", "/dev/null",
                   "--rootdir", str(path.parent), "--confcutdir", str(path.parent),
                   "--noconftest", "-p", "no:cacheprovider", "-q", path.name]
        child = subprocess.run(command, cwd=path.parent, env=env, text=True,
                               capture_output=True, timeout=30)
        results.append({"source": tag, "flag": flag, "exit_code": child.returncode,
                        "expected_exit_code": expected, "stdout": child.stdout,
                        "stderr": child.stderr})
    print(json.dumps({"python": sys.version.split()[0], "pytest": pytest.__version__,
                      "torch": torch.__version__, "source_sha256": HASHES,
                      "scope": "missing-Megatron collection only; zero GPU test bodies executed",
                      "results": results}, ensure_ascii=False, indent=2))
    skipped = all("1 skipped" in r["stdout"] for r in results[:2])
    missing_mcore = ("ModuleNotFoundError" in results[2]["stdout"]
                     and "No module named 'megatron'" in results[2]["stdout"])
    if not (skipped and missing_mcore) or any(
            r["exit_code"] != r["expected_exit_code"] for r in results):
        raise SystemExit("unexpected collection outcome; inspect raw output")


if __name__ == "__main__":
    main()
