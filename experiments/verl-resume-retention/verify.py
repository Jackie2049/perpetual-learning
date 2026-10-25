"""Run original #7396 CPU fixtures in isolated synthetic checkpoint directories."""

import __future__
import argparse
import ast
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from enum import Enum
from types import ModuleType

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_verl_checkpoint_retention as retained

HEAD = "8a1a031a9c03105839dcc6659e15cd975bfd5246"
VERSIONS = {
    "head": HEAD,
    "parent": "09ac37258ea66b0cb69b2738eec3074ea4e7261c",
    "main": "3c6d9b43b67c74ec343d77874cd8fa6e841523da",
}
REGRESSION = "test_restart_loaded_checkpoint_is_removed_after_replacement_commits"


def module(name):
    obj = ModuleType(name)
    obj.__path__ = []
    sys.modules[name] = obj
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("--revision", choices=VERSIONS, default="head")
    parser.add_argument("--scratch-spelling", choices=("canonical", "system"), default="canonical")
    args = parser.parse_args()
    root, revision = args.source_repo.resolve(), VERSIONS[args.revision]
    hashes = {}

    def source(rev, path):
        raw = subprocess.check_output(
            ["git", "show", rev + ":" + path], cwd=root,
            env=dict(os.environ, GIT_NO_LAZY_FETCH="1"),
        )
        hashes[rev + ":" + path] = hashlib.sha256(raw).hexdigest()
        return raw

    def definitions(raw, names, namespace):
        nodes = [n for n in ast.parse(raw).body if getattr(n, "name", None) in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "original_helpers", "exec",
                     flags=__future__.annotations.compiler_flag), namespace)

    assert "verl" not in sys.modules
    for name in ("verl", "verl.utils", "verl.utils.checkpoint"):
        module(name)
    retained.REVISION = revision
    manager = retained.load(root, "verl/utils/checkpoint/checkpoint_manager.py",
                            "verl.utils.checkpoint.checkpoint_manager", select_class=True)
    manager.re = re
    raw = source(revision, "verl/utils/checkpoint/checkpoint_manager.py")
    definitions(raw, {"find_latest_ckpt_path", "get_checkpoint_tracker_filename"}, manager.__dict__)
    fs = module("verl.utils.fs")
    fs.__dict__.update(os=os, tempfile=tempfile)
    definitions(source(revision, "verl/utils/fs.py"), {"local_mkdir_safe"}, fs.__dict__)
    log_namespace = {"logging": logging}
    definitions(source(revision, "verl/utils/logger/aggregate_logger.py"),
                {"log_with_rank"}, log_namespace)
    hdfs = module("verl.utils.hdfs_io")

    def remote_io_forbidden(*args, **kwargs):
        raise AssertionError("unmocked remote filesystem access")

    hdfs.copy = hdfs.makedirs = remote_io_forbidden
    handler = module("verl.utils.checkpoint.checkpoint_handler")
    handler.__file__ = "isolated_original_checkpoint_handler.py"
    handler.__dict__.update(
        json=json, logging=logging, os=os, re=re, Enum=Enum, torch=retained.torch,
        hdfs_io=hdfs, log_with_rank=log_namespace["log_with_rank"],
        find_latest_ckpt_path=manager.find_latest_ckpt_path,
        get_checkpoint_tracker_filename=manager.get_checkpoint_tracker_filename,
    )
    raw_handler = source(revision, "verl/utils/checkpoint/checkpoint_handler.py")
    tree = ast.parse(raw_handler)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, handler.__file__, "exec",
                 flags=__future__.annotations.compiler_flag), handler.__dict__)
    retained.REVISION = HEAD
    tests = ModuleType("_combined_original_retention_tests")
    tests.__file__ = str(Path(__file__).resolve())
    for file in ("test_checkpoint_cleanup_on_cpu.py", "test_checkpoint_handler_on_cpu.py"):
        path = "tests/utils/ckpt/" + file
        source(HEAD, path)
        original = retained.load(root, path, "_original_" + file[:-3])
        tests.__dict__.update({k: v for k, v in vars(original).items() if not k.startswith("__")})

    class OriginalTests(retained.pytest.Module):
        def _getobj(self):
            return tests

    class Adapter:
        def __init__(self):
            self.calls, self.failures, self.skips = [], [], []

        def pytest_pycollect_makemodule(self, module_path, parent):
            if module_path == Path(__file__).resolve():
                return OriginalTests.from_parent(parent, path=module_path)

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                self.calls.append(report.nodeid.split("::")[-1])
            if report.failed:
                self.failures.append((report.nodeid.split("::")[-1], report.when, str(report.longrepr)))
            if report.skipped:
                self.skips.append(report.nodeid)

    old_tempdir = tempfile.tempdir
    with tempfile.TemporaryDirectory(prefix="verl-resume-retention-") as scratch:
        allowed = Path(scratch).resolve()
        path_alias_distinct = os.path.abspath(scratch) != str(allowed)

        def guard(event, arguments):
            if event == "shutil.rmtree":
                target, dir_fd = arguments
                assert dir_fd is None
                resolved = Path(target).resolve()
                assert resolved == allowed or allowed in resolved.parents, resolved

        sys.addaudithook(guard)
        tempfile.tempdir = str(allowed) if args.scratch_spelling == "canonical" else scratch
        try:
            adapter = Adapter()
            options = [
                "-q", "--noconftest", "-c", os.devnull,
                "--rootdir", str(Path(__file__).resolve().parent),
                "--confcutdir", str(Path(__file__).resolve().parent),
                "-p", "no:cacheprovider", "--basetemp", str(allowed / "pytest"),
                str(Path(__file__).resolve()),
            ]
            if args.revision != "head":
                options += ["-k", REGRESSION]
            status = retained.pytest.main(options, plugins=[adapter])
        finally:
            tempfile.tempdir = old_tempdir
    assert not allowed.exists()
    assert not adapter.skips
    alias_case = args.revision == "head" and args.scratch_spelling == "system" and path_alias_distinct
    if args.revision == "head" and not alias_case:
        assert status == 0 and len(adapter.calls) == 22 and not adapter.failures
    elif alias_case:
        assert status == 1 and len(adapter.calls) == 22 and len(adapter.failures) == 1
        name, phase, failure = adapter.failures[0]
        assert name == "test_loaded_checkpoint_accepts_relative_registered_path" and phase == "call"
        assert "assert not os.path.exists(ckpt_100)" in failure
    else:
        assert status == 1 and adapter.calls == [REGRESSION] and len(adapter.failures) == 1
        name, phase, failure = adapter.failures[0]
        assert phase == "call" and "assert not old_checkpoint.exists()" in failure
    assert "verl.utils.device" not in sys.modules
    result = {
        "revision": revision, "label": args.revision,
        "python": sys.version.split()[0], "torch": retained.torch.__version__,
        "numpy": retained.np.__version__, "pytest": retained.pytest.__version__,
        "calls": adapter.calls, "pytest_exit": int(status),
        "expected_assertion_failures": len(adapter.failures),
        "failure_names": [name for name, phase, failure in adapter.failures],
        "scratch_spelling": args.scratch_spelling,
        "system_scratch_has_alias": path_alias_distinct,
        "scratch_removed": True, "sha256": hashes,
        "scope": "original AST manager/handler/helpers plus original Mock engine/dataloader/barrier/HDFS failure; no real model shards, process group, Ray, HDFS, GPU, crash recovery or training",
    }
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

