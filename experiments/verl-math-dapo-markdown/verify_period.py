#!/usr/bin/env python3
"""#8450 sentence-period delta; original complete scorer and native pytest.

../sft-hydra-cpu-venv/bin/python -B verify_period.py ../verl-packaging-readonly > result-period.json
Three empty package initializers isolate the original scorer from verl startup.
Original scorer and 53-case test file are copied byte-for-byte into temporary dirs.
No original assertion, parameter, decorator or import is rewritten.
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
from types import ModuleType

BASE = "dc709afb7a74fd2645640a4e5f2fdf40166b4fbb"
PREVIOUS = "f5d5123b6d34ec1c10ed914b9bd8bab50d714fca"
HEAD = "658a7781e8dbb35a6e1be1ee9f7fd33da237a702"
SOURCE = "verl/utils/reward_score/math_dapo.py"
TEST = "tests/utils/reward_score/test_math_dapo_minerva_markdown_on_cpu.py"


def main():
    repo = sys.argv[1]
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    hashes = {}
    def read(rev, path):
        raw = subprocess.check_output(["git", "-C", repo, "show", f"{rev}:{path}"], env=env)
        compile(raw, path, "exec")
        hashes[f"{rev}:{path}"] = hashlib.sha256(raw).hexdigest()
        return raw
    test_source = read(HEAD, TEST)
    previous_test = read(PREVIOUS, TEST)
    old_test_ast = {n.name: ast.dump(n) for n in ast.parse(previous_test).body if isinstance(n, ast.FunctionDef)}
    new_test_ast = {n.name: ast.dump(n) for n in ast.parse(test_source).body if isinstance(n, ast.FunctionDef)}
    assert all(new_test_ast[k] == v for k, v in old_test_ast.items())
    modules, test_results, sources = {}, {}, {}
    for label, rev in (("baseline", BASE), ("previous", PREVIOUS), ("period_head", HEAD)):
        raw = read(rev, SOURCE)
        sources[label] = raw
        module = ModuleType("scorer_" + rev[:8])
        exec(compile(raw, SOURCE, "exec"), module.__dict__)
        modules[label] = module
        with tempfile.TemporaryDirectory(prefix="minerva-period-") as tmp:
            root = Path(tmp)
            package = root / "verl" / "utils" / "reward_score"
            package.mkdir(parents=True)
            for path in (root / "verl", root / "verl" / "utils", package):
                (path / "__init__.py").write_bytes(b"")
            (package / "math_dapo.py").write_bytes(raw)
            (root / "test_minerva.py").write_bytes(test_source)
            report = root / "report.xml"
            run = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=no", "-p", "no:cacheprovider", f"--junitxml={report}", "test_minerva.py"], cwd=root, env=dict(env, PYTHONPATH=tmp), capture_output=True, text=True, timeout=45)
            import xml.etree.ElementTree as ET
            cases = ET.parse(report).getroot().findall(".//testcase")
            counts = {"total": len(cases), "failed": sum(c.find("failure") is not None for c in cases), "errors": sum(c.find("error") is not None for c in cases), "skipped": sum(c.find("skipped") is not None for c in cases)}
            counts["passed"] = counts["total"] - counts["failed"] - counts["errors"] - counts["skipped"]
            test_results[label] = dict(counts=counts, returncode=run.returncode, stdout=run.stdout.replace(tmp, "<temporary-test-root>"), stderr=run.stderr.replace(tmp, "<temporary-test-root>"))
    assert test_results["baseline"]["counts"] == dict(total=53, failed=39, errors=0, skipped=0, passed=14)
    assert test_results["previous"]["counts"] == dict(total=53, failed=8, errors=0, skipped=0, passed=45)
    assert test_results["period_head"]["counts"] == dict(total=53, failed=0, errors=0, skipped=0, passed=53)
    assert [test_results[x]["returncode"] for x in test_results] == [1, 1, 0]
    unchanged = {"last_boxed_only_string", "remove_boxed", "normalize_final_answer", "is_correct_strict_box", "verify", "compute_score", "_strip_markdown_emphasis", "_marker_run"}
    defs = [{n.name: ast.dump(n) for n in ast.parse(sources[label]).body if isinstance(n, ast.FunctionDef) and n.name in unchanged} for label in ("previous", "period_head")]
    assert len(defs[0]) == 8 and defs[0] == defs[1]
    probes = [
        ("plain_period", "Answer: 42.", "42", False, (-1, 1)),
        ("period_then_space", "Answer: 42. ", "42", False, (-1, -1)),
        ("period_then_crlf", "Answer: 42.\r\n", "42", False, (-1, -1)),
        ("period_inside_bold", "Answer: **42.**", "42", False, (-1, -1)),
        ("period_outside_bold", "Answer: **42**.", "42", False, (-1, 1)),
        ("period_inside_math_dollars", "Answer: $42.$", "42", False, (1, 1)),
        ("target_period_then_space", "Answer: 42", "42. ", False, (-1, -1)),
        ("target_period", "Answer: 42", "42.", False, (-1, 1)),
        ("boxed_route_with_target_period", "\\boxed{42}", "42.", False, (-1, -1)),
        ("strict_box_does_not_strip_period", "\\boxed{42.}", "42", True, (-1, -1)),
        ("wrong_minerva_still_blocks_box", "Answer: 41.\n\\boxed{42}", "42", False, (-1, -1)),
        ("dot_only_answer_releases_box_fallback", "Answer: .\n\\boxed{42}", "42", False, (-1, 1)),
    ]
    controls = []
    for name, text, gt, strict, expected in probes:
        result = {label: modules[label].compute_score(text, gt, strict_box_verify=strict) for label in ("previous", "period_head")}
        assert tuple(result[k]["score"] for k in result) == expected, (name, result)
        controls.append(dict(name=name, text=text, ground_truth=gt, strict=strict, **result))
    direct = modules["period_head"]._strip_trailing_sentence_period
    helper = {x: direct(x) for x in ("42.", "42...", "42. ", "42。", "3.14")}
    assert helper == {"42.": "42", "42...": "42", "42. ": "42. ", "42。": "42。", "3.14": "3.14"}
    extracted_gt = {label: modules[label].is_correct_minerva("Answer: 42", r"reference \boxed{42.}", gt_need_extract=True) for label in ("previous", "period_head")}
    assert extracted_gt == {"previous": (False, "42"), "period_head": (True, "42")}
    print(json.dumps({"head": HEAD, "previous": PREVIOUS, "baseline": BASE, "environment": {"python": platform.python_version(), "pytest": importlib.metadata.version("pytest")}, "source_sha256": hashes, "native_pytest_same_head_tests": test_results, "old_test_function_ast_retained": len(old_test_ast), "unchanged_scorer_helpers_previous_to_head": sorted(unchanged), "own_public_entry_controls": controls, "own_direct_helper_controls": helper, "own_extracted_gt_control": extracted_gt, "scope": {"original": "complete stdlib scorer and complete byte-identical head tests, native pytest in fresh process per revision", "isolation": "three empty package initializers; no verl package startup/conftest", "not_run": ["reward manager", "model/rollout/training", "arbitrary regex or general ReDoS proof", "GPU/NPU/server access"]}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
