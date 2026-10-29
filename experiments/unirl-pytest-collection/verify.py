"""Native pytest collection on two original-file snapshots of UniRL #632.

The snapshots contain original pyproject.toml and the original smoke CLI only;
Git tree inventory checks there are no omitted matching tests/conftest files.
No CLI function/test body, HTTP request, service, model or device is executed.
Requires Python >=3.12, pytest and requests. Fixed Git blobs must be cached.
Original inputs: Tencent-Hunyuan/UniRL, Apache-2.0. Prints JSON, writes no results.
"""

import argparse
import ast
import fnmatch
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


BASE = "8cfbac17765004afb68fef08a4e011a4c8daf555"
HEAD = "ff13197a2ba6208a2287c32d72daf202fd4f68e0"
PREFIX = "unirl-reward-service/"
SCRIPTS = {"base": "scripts/test_videoalign.py", "candidate": "scripts/videoalign_smoke.py"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", PYTHONDONTWRITEBYTECODE="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="", PYTEST_PLUGINS="")
    hashes, inventories, trees, results = {}, {}, {}, {}

    def git(*arguments, git_env=env):
        return subprocess.check_output(["git", *arguments], cwd=args.source_repo, env=git_env)

    with tempfile.TemporaryDirectory(prefix="unirl-pytest-collection-") as temporary:
        root = Path(temporary)
        for label, revision in [("base", BASE), ("candidate", HEAD)]:
            inventory_raw = git("ls-tree", "-r", "--name-only", revision, PREFIX)
            inventory = inventory_raw.decode().splitlines()
            matching = [p for p in inventory if fnmatch.fnmatch(Path(p).name, "test_*.py")
                        or fnmatch.fnmatch(Path(p).name, "*_test.py")]
            assert matching == ([PREFIX + SCRIPTS[label]] if label == "base" else [])
            assert not any(Path(p).name == "conftest.py" for p in inventory)
            assert not any(p.startswith(PREFIX + "tests/") for p in inventory)
            assert PREFIX + "scripts/__init__.py" not in inventory
            inventories[label] = {"tracked_paths": len(inventory),
                "path_list_sha256": hashlib.sha256(inventory_raw).hexdigest(),
                "matching_test_files": matching, "tests_tree_present": False,
                "conftest_present": False}
            paths = [PREFIX + "pyproject.toml", PREFIX + SCRIPTS[label]]
            original = {}
            for path in paths:
                raw = git("show", revision + ":" + path)
                original[path] = raw
                hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
            trees[label] = ast.parse(original[PREFIX + SCRIPTS[label]])
            # A disposable independent index exports copies, leaving source
            # worktree/index untouched. No generated config or fake test file.
            snapshot = root / label
            snapshot.mkdir()
            copy_env = dict(env, GIT_INDEX_FILE=str(root / (label + ".index")))
            git("read-tree", revision, git_env=copy_env)
            git("checkout-index", "--prefix=" + str(snapshot) + "/", "--", *paths, git_env=copy_env)
            for path, raw in original.items():
                assert (snapshot / path).read_bytes() == raw
            cwd = snapshot / PREFIX
            cases = [("default", []), ("no_recursive_exclusions", ["-o", "norecursedirs="])]
            if label == "base":
                cases = [("default", []),
                         ("marker_filter", ["-m", "not gpu and not slow and not integration"]),
                         ("remove_stale_testpaths", ["-o", "testpaths="]),
                         ("exclude_scripts_only", ["-o", "norecursedirs=scripts"])]
            else:
                cases += [("explicit_smoke_file", [SCRIPTS[label]]),
                          ("missing_integration_path", ["tests/integration/", "-m", "integration"])]
            results[label] = {}
            for name, extra in cases:
                command = [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q",
                           "--color=no", "-p", "no:cacheprovider", *extra]
                run = subprocess.run(command, cwd=cwd, env=env, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
                nodes = [line for line in run.stdout.splitlines()
                         if line.startswith("scripts/") and "::test_" in line]
                results[label][name] = {"arguments_after_pytest": command[4:],
                    "exit_code": run.returncode, "collected_nodeids": nodes,
                    "missing_testpaths_warning": "No files were found in testpaths" in run.stdout,
                    "unknown_asyncio_mode_warning": "Unknown config option: asyncio_mode" in run.stdout,
                    "missing_path_error": "file or directory not found: tests/integration/" in run.stdout}
                expected_nodes = 6 if (label == "base" and name != "exclude_scripts_only") or name == "explicit_smoke_file" else 0
                expected_code = 0 if expected_nodes else (4 if name == "missing_integration_path" else 5)
                assert (len(nodes), run.returncode) == (expected_nodes, expected_code), run.stdout

        # Rename changes only the module docstring/usage; executable AST is equal.
        executable_ast_equal = ast.dump(ast.Module(body=trees["base"].body[1:], type_ignores=[])) == ast.dump(
            ast.Module(body=trees["candidate"].body[1:], type_ignores=[]))
        assert executable_ast_equal
        print(json.dumps({"base": BASE, "head": HEAD,
            "environment": {"python": platform.python_version(),
                "packages": {p: version(p) for p in ["pytest", "requests", "pluggy", "iniconfig",
                    "packaging", "pygments", "charset-normalizer", "idna", "urllib3", "certifi"]},
                "plugin_autoload": False}, "source_sha256": hashes,
            "inventory": inventories, "smoke_executable_ast_equal": executable_ast_equal,
            "native_collection_results": results,
            "scope": "Two original files per snapshot, inventory checked; collection only, no test bodies or HTTP/service/model/GPU execution"},
            ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
