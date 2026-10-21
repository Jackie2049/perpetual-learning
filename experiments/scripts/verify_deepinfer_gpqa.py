#!/usr/bin/env python3
"""Run #160's original seven GPQA tests against fixed base/fix Git objects.

Python >=3.10 and pytest. Full common package initializer, common.io, bench,
and original tests execute unchanged. gpqa is a namespace package; no AST
extraction, parser stubs, dataset/model requests, installs or checkout edits.
The pytest collector adapter only supplies the in-memory original test module.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys
from types import ModuleType

from verify_deepinfer_sampling_params import load
import pytest

BASE = "9bd9cf243fc53ef9afbf8f04d391d989117d6ab9"
FIX = "4bda76dda5b9158380f2b89eecad68fa4d1614e4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("--revision", choices=("base", "fix"), default="fix")
    args = parser.parse_args()
    root = args.source_repo.resolve()
    revision = BASE if args.revision == "base" else FIX
    assert "common" not in sys.modules and "gpqa" not in sys.modules
    common = load(root, revision, "eval/common/__init__.py", "common")
    common.__path__ = []
    common.io = load(root, revision, "eval/common/io.py", "common.io")
    gpqa = ModuleType("gpqa")
    gpqa.__path__ = []
    sys.modules["gpqa"] = gpqa
    bench = load(root, revision, "eval/gpqa/bench.py", "gpqa.bench")
    gpqa.bench = bench
    tests = load(root, FIX, "tests/eval/test_gpqa_boxed_case.py", "_original_gpqa_tests")

    class OriginalTests(pytest.Module):
        def _getobj(self):
            return tests

    class Adapter:
        results = []

        def pytest_configure(self, config):
            config.addinivalue_line("markers", "ci: original upstream regression marker")
            config.addinivalue_line("markers", "ci_npu: original upstream regression marker")

        def pytest_pycollect_makemodule(self, module_path, parent):
            if module_path == Path(__file__).resolve():
                return OriginalTests.from_parent(parent, path=module_path)

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                self.results.append((report.nodeid.rsplit("::", 1)[-1], report.outcome))

    adapter = Adapter()
    code = pytest.main(["-q", "--tb=line", "--noconftest", "-p", "no:cacheprovider",
                        str(Path(__file__).resolve())], plugins=[adapter])
    counts = Counter(outcome for _, outcome in adapter.results)
    failed = {name.split("[", 1)[0] for name, outcome in adapter.results if outcome == "failed"}
    if args.revision == "base":
        assert code == 1 and counts == {"passed": 4, "failed": 3}, (code, counts)
        assert failed == {"test_extract_answer_normalizes_boxed_case", "test_extract_answer_uses_last_boxed",
                          "test_score_rollout_accepts_lowercase_boxed"}, failed
    else:
        assert code == 0 and counts == {"passed": 7}, (code, counts)

    fixed = args.revision == "fix"
    cases = [
        ("ANSWER: A\n" + r"\boxed{d}", ("A", "answer_line")),
        (r"\boxed{A} then \boxed{b}", ("B" if fixed else "A", "boxed")),
        (r"\BOXED{d}", ("D", "boxed") if fixed else (None, "none")),
        ("A choice.", ("A", "tail_letter")),
        ("a choice.", (None, "none")),
        ("answer： (b)", ("B", "answer_line")),
        ("ANSWER: D\nANSWER: b", ("B", "answer_line")),
        ("", (None, "none")),
    ]
    for text, expected in cases:
        assert bench.extract_answer_with_source(text) == expected, (text, expected)
    problem = bench._normalize_row({"question": "placeholder", "solution": r"\boxed{d}"}, 0)
    assert problem.answer == "d"
    record = bench.build_records([problem])[0]
    row = bench.score_rollout({"raw_text": "ANSWER: D"}, record)
    assert row["correct"] is True and row["gold"] == "d"
    assert all(name not in sys.modules for name in ("torch", "deepinfer", "openai", "datasets"))
    print("Python", sys.version.split()[0], "pytest", pytest.__version__, "original tests", dict(counts))
    print("9 independent native parser/scoring controls passed; no data file, model, judge client, engine or device")


if __name__ == "__main__":
    main()
