#!/usr/bin/env python3
"""Run #8435's unchanged CPU test file against original AST slices.

../deepinfer-eplb-venv/bin/python -B experiments/verl-sglang-bucket-lifetime/verify.py ../verl-packaging-readonly
No SGLang, Ray, distributed group, network, or accelerator is initialized.
"""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

BASE = "2ab35292039a51175a4b3d3b3b2d76e951a0d3e4"
HEAD = "3b1cef9143498004301d4ec941b8acfc816f66e2"
UTILS = "verl/workers/rollout/sglang_rollout/utils.py"
ADAPTER = "verl/workers/rollout/sglang_rollout/sglang_rollout.py"
ITERATOR = "verl/workers/rollout/utils.py"
TEST = "tests/checkpoint_engine/test_sglang_fusion_bucketing_on_cpu.py"


def select(source, names):
    nodes = []
    for node in ast.parse(source).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
        if name in names:
            nodes.append(node)
    assert len(nodes) == len(names), names
    return nodes


def render(nodes, imports):
    text = imports + "\n\n" + "\n\n".join(ast.unparse(n) for n in nodes) + "\n"
    parsed = ast.parse(text).body[-len(nodes):]
    assert [ast.dump(n) for n in parsed] == [ast.dump(n) for n in nodes]
    return text


def main():
    repo = sys.argv[1]
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    sources, hashes = {}, {}
    for revision, paths in ((BASE, (UTILS, ADAPTER)), (HEAD, (UTILS, ADAPTER, ITERATOR, TEST))):
        for path in paths:
            raw = subprocess.check_output(["git", "-C", repo, "show", f"{revision}:{path}"], env=env)
            compile(raw, path, "exec")
            sources[revision, path] = raw
            hashes[f"{revision}:{path}"] = hashlib.sha256(raw).hexdigest()
    iterator_nodes = select(sources[HEAD, ITERATOR], {"ensure_async_iterator"})
    slices = {}
    ast_hashes = {}
    for revision in (BASE, HEAD):
        utils_nodes = select(sources[revision, UTILS], {"_DEEPSEEK_V4_FUSION_MEMBERS", "DEEPSEEK_V4_FUSION_GROUPS", "_fusion_key", "_compact_for_bucket", "get_named_tensor_buckets"})
        slices[revision, UTILS] = render(iterator_nodes + utils_nodes, "from __future__ import annotations\nfrom typing import Iterator\nimport torch")
        helpers = select(sources[revision, ADAPTER], {"_strip_lora_base_layer", "_uses_quantized_refit", "_to_ipc_device"})
        original_class = select(sources[revision, ADAPTER], {"ServerAdapter"})[0]
        method = next(n for n in original_class.body if getattr(n, "name", None) == "update_weights")
        # Only the class container is simplified; the whole original method is retained.
        container = ast.ClassDef(name="ServerAdapter", bases=[], keywords=[], body=[method], decorator_list=[], type_params=[])
        ast.fix_missing_locations(container)
        imports = "from __future__ import annotations\nimport torch\nfrom sglang.srt.weight_sync.utils import update_weights as sgl_update_weights\nfrom verl.workers.rollout.sglang_rollout.utils import DEEPSEEK_V4_FUSION_GROUPS, get_named_tensor_buckets"
        slices[revision, ADAPTER] = render(helpers + [container], imports)
        extracted = ast.parse(slices[revision, ADAPTER]).body[-1].body[0]
        assert ast.dump(extracted) == ast.dump(method)
        ast_hashes[revision] = {
            "get_named_tensor_buckets": hashlib.sha256(ast.dump(utils_nodes[-1]).encode()).hexdigest(),
            "ServerAdapter.update_weights": hashlib.sha256(ast.dump(method).encode()).hexdigest(),
        }
    runs = []
    for label, gen, adapter in (("base", BASE, BASE), ("generator_only", HEAD, BASE), ("adapter_only", BASE, HEAD), ("both", HEAD, HEAD)):
        with tempfile.TemporaryDirectory(prefix="verl-bucket-lifetime-") as tmp:
            root = Path(tmp)
            for package in ("verl", "verl/workers", "verl/workers/rollout", "verl/workers/rollout/sglang_rollout"):
                folder = root / package
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "__init__.py").write_bytes(b"")
            for revision, path in ((gen, UTILS), (adapter, ADAPTER)):
                (root / path).write_text(slices[revision, path])
            (root / "test_original.py").write_bytes(sources[HEAD, TEST])
            assert hashlib.sha256((root / "test_original.py").read_bytes()).hexdigest() == hashes[f"{HEAD}:{TEST}"]
            report = root / "report.xml"
            run = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider", f"--junitxml={report}", "test_original.py"], cwd=root, env=dict(env, PYTHONPATH=tmp), capture_output=True, text=True, timeout=45)
            cases = ET.parse(report).getroot().findall(".//testcase") if report.exists() else []
            results = []
            for case in cases:
                failure = case.find("failure")
                error = case.find("error")
                status = "failed" if failure is not None else "error" if error is not None else "skipped" if case.find("skipped") is not None else "passed"
                message = (failure if failure is not None else error).attrib.get("message", "").replace(tmp, "<temporary-root>") if status in ("failed", "error") else None
                results.append({"name": case.attrib["name"], "status": status, **({"message": message} if message else {})})
            runs.append({"label": label, "generator": gen, "adapter": adapter, "exit_code": run.returncode, "junit_available": report.exists(), "counts": {status: sum(c["status"] == status for c in results) for status in ("passed", "failed", "error", "skipped")}, "cases": results, "stdout_tail": run.stdout[-1200:].replace(tmp, "<temporary-root>"), "stderr": run.stderr.replace(tmp, "<temporary-root>")})
    output = {"scope": "Unchanged complete original bucketer test file; original AST functions and complete adapter method; empty package initializers and simplified class container. Original test supplies HTTP/SGLang stubs. Not full verl import or real IPC/backend.", "environment": {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ("torch", "numpy", "pytest")}}, "source_sha256": hashes, "ast_sha256": ast_hashes, "runs": runs}
    print(json.dumps(output, indent=2))
    assert runs[-1]["counts"] == {"passed": 20, "failed": 0, "error": 0, "skipped": 0}, runs[-1]


if __name__ == "__main__":
    main()
