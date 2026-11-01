#!/usr/bin/env python3
"""Pinned #8443 CPU tests with original helper and diagnostics, no device imports.

../deepinfer-eplb-venv/bin/python -B verify.py ../verl-packaging-readonly > result.json
The original Gloo test uses one CPU rank and a temporary file store, not a server.
"""
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import ModuleType
import xml.etree.ElementTree as ET

import torch

HEAD = "edede6a14cadfb7bfcaa00198a8c00fd2ee547f3"
PATHS = ["verl/utils/fsdp_utils.py", "tests/checkpoint_engine/weight_diagnostics.py", "tests/utils/test_fsdp_dtensor_export_on_cpu.py", "tests/checkpoint_engine/test_weight_diagnostics_on_cpu.py", "verl/workers/engine/fsdp/transformer_impl.py"]


def main():
    repo = sys.argv[1]
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    sources, hashes = {}, {}
    for path in PATHS:
        raw = subprocess.check_output(["git", "-C", repo, "show", f"{HEAD}:{path}"], env=env)
        compile(raw, path, "exec")
        sources[path] = raw
        hashes[path] = hashlib.sha256(raw).hexdigest()
    tree = ast.parse(sources[PATHS[0]])
    helper = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "materialize_fsdp_dtensor"]
    assert len(helper) == 1
    helper_text = "import torch\n\n" + ast.unparse(helper[0]) + "\n"
    assert ast.dump(ast.parse(helper_text).body[1]) == ast.dump(helper[0])
    with tempfile.TemporaryDirectory(prefix="verl-export-completion-") as tmp:
        root = Path(tmp)
        for package in ("verl", "verl/utils", "tests", "tests/checkpoint_engine"):
            folder = root / package
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "__init__.py").write_bytes(b"")
        (root / "verl/utils/fsdp_utils.py").write_text(helper_text)
        (root / "tests/checkpoint_engine/weight_diagnostics.py").write_bytes(sources[PATHS[1]])
        (root / "test_fsdp_dtensor_export_on_cpu.py").write_bytes(sources[PATHS[2]])
        (root / "test_weight_diagnostics_on_cpu.py").write_bytes(sources[PATHS[3]])
        report = root / "report.xml"
        run = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider", f"--junitxml={report}", "test_fsdp_dtensor_export_on_cpu.py", "test_weight_diagnostics_on_cpu.py"], cwd=root, env=dict(env, PYTHONPATH=tmp), capture_output=True, text=True, timeout=45)
        native_stdout = run.stdout.replace(tmp, "<temporary-root>")
        native_stderr = run.stderr.replace(tmp, "<temporary-root>")
        # Native backend initialization may abort before pytest can publish XML.
        # Preserve that process failure instead of masking it with FileNotFoundError.
        report_available = report.exists()
        cases = ET.parse(report).getroot().findall(".//testcase") if report_available else []
        results = [{"name": c.attrib["name"], "status": "failed" if c.find("failure") is not None else "error" if c.find("error") is not None else "skipped" if c.find("skipped") is not None else "passed"} for c in cases]
    diag = ModuleType("original_weight_diagnostics")
    exec(compile(sources[PATHS[1]], PATHS[1], "exec"), diag.__dict__)
    # Explicit CPU controls distinguish exact-byte audit from tolerant value checks.
    original = torch.tensor([1.0], dtype=torch.float32, device="cpu")
    close = torch.tensor([1.000001], dtype=torch.float32, device="cpu")
    assert diag.fingerprint(original) != diag.fingerprint(close)
    details = diag.mismatch_details(original, close)
    assert details["cpu_isclose_mismatches"] == 0
    with tempfile.TemporaryDirectory(prefix="verl-empty-audit-") as tmp, contextlib.redirect_stdout(io.StringIO()):
        empty = diag.summarize(tmp)
        assert empty["compared"] == 0 and empty["differences"] == [] and empty["incomplete"] == 0
    # A CPU plain value is not enough to test pending NPU writes; the original test
    # supplies a callback stand-in explicitly. No torch-npu or CUDA API is queried here.
    summary = {name: sum(r["status"] == name for r in results) for name in ("passed", "failed", "error", "skipped")}
    output = {
        "head": HEAD, "environment": {"python": platform.python_version(), "platform": sys.platform, "torch": torch.__version__, "torch_git": torch.version.git_version, "pytest": importlib.metadata.version("pytest"), "gloo_socket_ifname": os.getenv("GLOO_SOCKET_IFNAME")},
        "source_sha256_compile": hashes,
        "helper_ast_unchanged": True,
        "original_native_pytest": {"returncode": run.returncode, "junit_report_available": report_available, "summary": summary, "items": results, "stdout": native_stdout, "stderr": native_stderr},
        "own_controls": {"exact_fingerprint_differs_inside_allclose_tolerance": True, "allclose_mismatch_details": details, "empty_audit": {k: empty[k] for k in ("compared", "incomplete", "unreadable", "differences")}},
        "scope": {"original": "two full byte-identical original test files; complete diagnostic module; original materialize helper AST", "seams": "four empty package initializers; no full fsdp_utils or verl startup; original NPU/CUDA test cases use CPU stand-ins", "real_dtensor_test": "original source_audit_records_real_dtensor uses world_size=1 CPU Gloo and file store if available", "not_run": ["actual exporter method", "NPU/CUDA streams or DMA", "FSDP training or multi-rank transfer", "Ray/HCCL/checkpoint engine", "workflow or shutdown/SIGSEGV reproduction", "performance or allocator peak"]},
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if run.returncode:
        raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
