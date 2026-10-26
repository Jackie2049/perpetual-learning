"""Run pinned #8365 CPU bodies; --revised-source checks the later boundary fix."""
import __future__
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import pytest
import torch


VERSIONS = {"base": "3c6d9b43b67c74ec343d77874cd8fa6e841523da",
            "head": "0ed8b37033984ba9862395af9ec621125f53eb10"}
SOURCE = "verl/utils/torch_functional.py"
TEST = "tests/utils/test_torch_functional.py"
FUNCTION = "calculate_sum_pi_squared_from_logits"
REGULAR = "test_calculate_sum_pi_squared_from_logits"
EXTREME = REGULAR + "_extreme_values"
CHUNKING = REGULAR + "_chunking"
REJECTS = REGULAR + "_rejects_non_positive_chunk_size"
REVISED = "23d891cd2ac16d3d7063c4eb94330d6aa29a9b5d"
REVISED_HASHES = {
    "old-production.py": "82a559e4f0aa3c3d33b0c2e584974f7e00394e130268957e8847d87d5792b8a4",
    "old-test.py": "26ea5c909b1184eb1c943934233ee6e21f990b3933a7cb7ebd53450ab1147858",
    "new-production.py": "9b3dccdb1cd38dd642a75293b5d62a03062002cccdc9207511f281400493133c",
    "new-test.py": "3485abd190764db43ca897c72e6d7966c5891f833a2f16069809be366ed24f9b",
}


def revised_result(source_directory):
    """Execute the new original tests unchanged against the two original helpers."""
    trees, hashes = {}, {}
    for name, expected in REVISED_HASHES.items():
        raw = (source_directory / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        assert hashes[name] == expected, name
        trees[name] = ast.parse(raw, filename=name)
        compile(trees[name], name, "exec")
    for name in (REGULAR, EXTREME):
        nodes = [next(n for n in trees[version + "-test.py"].body if getattr(n, "name", None) == name)
                 for version in ("old", "new")]
        assert ast.dump(nodes[0], include_attributes=False) == ast.dump(nodes[1], include_attributes=False)
    original_tests = [n for n in trees["new-test.py"].body
                      if getattr(n, "name", None) in (REGULAR, EXTREME, CHUNKING, REJECTS)]
    assert len(original_tests) == 4
    cases = [(REGULAR, (shape, dtype)) for shape in ((8, 17), (3, 5, 32), (1, 1024))
             for dtype in (torch.float32, torch.float64)]
    cases += [(EXTREME, ())]
    cases += [(CHUNKING, (shape, chunk)) for shape in ((10, 33), (3, 7, 33), (0, 33))
              for chunk in (1, 4, 2048)]
    cases += [(REJECTS, (chunk,)) for chunk in (0, -1)]
    assert len(cases) == 18
    torch.set_default_device("cpu")
    results = {}
    for version in ("old", "new"):
        helper = next(n for n in trees[version + "-production.py"].body if getattr(n, "name", None) == FUNCTION)
        ns = {"torch": torch, "pytest": pytest}
        exec(compile(ast.Module(body=[helper, *original_tests], type_ignores=[]), "original_definitions", "exec",
                     flags=__future__.annotations.compiler_flag), ns)
        outcomes = []
        for name, arguments in cases:
            record = {"test": name, "parameters": [str(arg) for arg in arguments]}
            try:
                ns[name](*arguments)
            except (Exception, pytest.fail.Exception) as exc:
                record.update(outcome="failed", error_type=type(exc).__name__, error=str(exc))
            else:
                record["outcome"] = "passed"
            outcomes.append(record)
        results[version] = {"passed": sum(r["outcome"] == "passed" for r in outcomes), "cases": outcomes}
    assert results["old"]["passed"] == 13 and results["new"]["passed"] == 18
    old_failures = [r for r in results["old"]["cases"] if r["outcome"] == "failed"]
    assert [r["test"] for r in old_failures] == [CHUNKING] * 3 + [REJECTS] * 2
    assert "verl" not in sys.modules
    return {"versions": {"old": VERSIONS["head"], "new": REVISED},
            "production_path": SOURCE, "test_path": TEST, "tests_from": REVISED,
            "source_sha256": hashes, "results": results,
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "pytest": pytest.__version__, "system": platform.system()},
            "scope": "18 new original CPU test cases against both AST-extracted original helpers; "
                     "empty backward and nonpositive parameters included; no native pytest collection, "
                     "uninitialized-value reads, GPU/memory/timing, V1 wiring, or OTB training"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path, nargs="?")
    parser.add_argument("--revised-source", type=Path)
    args = parser.parse_args()
    if args.revised_source is not None:
        print(json.dumps(revised_result(args.revised_source), indent=2, sort_keys=True))
        return
    if args.source_repo is None:
        parser.error("source_repo is required without --revised-source")
    hashes, trees = {}, {}
    for label, revision in VERSIONS.items():
        for path in (SOURCE, TEST):
            raw = subprocess.check_output(["git", "show", revision + ":" + path], cwd=args.source_repo,
                                          env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
            hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
            tree = ast.parse(raw, filename=path)
            compile(tree, path, "exec")
            trees[label, path] = tree
    for name in (REGULAR, EXTREME):
        original_tests = [next(node for node in trees[label, TEST].body if getattr(node, "name", None) == name)
                          for label in VERSIONS]
        assert ast.dump(original_tests[0], include_attributes=False) == ast.dump(original_tests[1], include_attributes=False)
    torch.set_default_device("cpu")
    results = {}
    for label in VERSIONS:
        ns = {"torch": torch, "pytest": pytest}
        selected = [node for node in trees[label, SOURCE].body if getattr(node, "name", None) == FUNCTION]
        names = {REGULAR, EXTREME} | ({CHUNKING} if label == "head" else set())
        selected += [node for node in trees["head", TEST].body if getattr(node, "name", None) in names]
        assert len(selected) == len(names) + 1
        exec(compile(ast.Module(body=selected, type_ignores=[]), "original_definitions", "exec",
                     flags=__future__.annotations.compiler_flag), ns)
        cases = []
        for shape in ((8, 17), (3, 5, 32), (1, 1024)):
            for dtype in (torch.float32, torch.float64):
                ns[REGULAR](shape, dtype)
                cases.append(REGULAR + ":" + repr((shape, str(dtype))))
        ns[EXTREME]()
        cases.append(EXTREME)
        if label == "head":
            for shape in ((10, 33), (3, 7, 33), (0, 33)):
                for chunk_size in (1, 4, 2048):
                    ns[CHUNKING](shape, chunk_size)
                    cases.append(CHUNKING + ":" + repr((shape, chunk_size)))
        assert len(cases) == (16 if label == "head" else 7)
        empty = torch.empty((0, 33), dtype=torch.float64, requires_grad=True)
        output = ns[FUNCTION](empty)
        empty_result = {"output_shape": list(output.shape), "requires_grad": output.requires_grad}
        if label == "base":
            gradient, = torch.autograd.grad(output.sum(), empty)
            assert gradient.shape == empty.shape
            empty_result.update(backward="passed", gradient_shape=list(gradient.shape))
        else:
            assert not output.requires_grad
            try:
                torch.autograd.grad(output.sum(), empty)
            except RuntimeError as exc:
                assert "does not require grad" in str(exc)
                empty_result.update(backward="expected_runtime_error", error=str(exc))
            else:
                raise AssertionError("Empty gradient regression did not reproduce")
        negative_result = None
        if label == "head":
            values = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
            output = ns[FUNCTION](values, chunk_size=-1)
            assert output.shape == (2,) and not output.requires_grad
            negative_result = {"rejected": False, "output_shape": list(output.shape),
                               "requires_grad": output.requires_grad, "uninitialized_values_not_read": True}
        results[label] = {"passed_original_cases": cases, "empty_backward_control": empty_result,
                          "negative_chunk_control": negative_result}
    assert "verl" not in sys.modules
    print(json.dumps({"versions": VERSIONS, "source_sha256": hashes, "results": results,
                      "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                      "pytest": pytest.__version__, "system": platform.system()},
                      "scope": "unchanged original function/test bodies; explicit CPU tensors and autograd; "
                               "7 existing base cases and 16 head cases, not full pytest collection; "
                               "two reported-boundary controls, negative output values never read; "
                               "no GPU/allocator peak or timing/FSDP/trainer/V1 wiring/OTB training"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
