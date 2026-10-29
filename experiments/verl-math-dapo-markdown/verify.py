"""Pinned original math_dapo modules and original test bodies, stdlib only.

No native pytest collection, package imports, model, worker or device execution.
Only test imports and parametrize decorators are removed; parameter values,
function bodies and assertions remain original. Separate probes record known
review boundaries, without applying a proposed fix.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

BASE = "dc709afb7a74fd2645640a4e5f2fdf40166b4fbb"
MAIN = "a08ea02a784d7d4982b1d81df36d188a476192bb"
HEAD = "e8e82c2fc349c64e199392eaf6f7aaf79d185d42"
SCORER = "verl/utils/reward_score/math_dapo.py"
TEST = "tests/utils/reward_score/test_math_dapo_minerva_markdown_on_cpu.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    hashes = {}

    def source(rev, path):
        raw = subprocess.check_output(["git", "show", f"{rev}:{path}"], cwd=args.repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[f"{rev}:{path}"] = hashlib.sha256(raw).hexdigest()
        return raw

    def module(rev):
        result = ModuleType("original_math_dapo_" + rev[:8])
        exec(compile(source(rev, SCORER), f"{rev}:{SCORER}", "exec"), result.__dict__)
        return result

    assert source(BASE, SCORER) == source(MAIN, SCORER)
    base, head = module(BASE), module(HEAD)
    def unchanged_definitions(rev):
        names = {"last_boxed_only_string", "remove_boxed", "normalize_final_answer",
                 "is_correct_strict_box", "verify", "compute_score"}
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(source(rev, SCORER)).body
                if isinstance(node, ast.FunctionDef) and node.name in names}
    assert unchanged_definitions(BASE) == unchanged_definitions(HEAD)
    assert len(unchanged_definitions(HEAD)) == 6
    original_tree = ast.parse(source(HEAD, TEST))

    def original_tests(scorer):
        tree = copy.deepcopy(original_tree)
        nodes = []
        functions = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.FunctionDef):
                functions.append((node.name, list(node.decorator_list)))
                node.decorator_list = []
            nodes.append(node)
        tree.body = nodes
        namespace = {"compute_score": scorer.compute_score,
                     "is_correct_minerva": scorer.is_correct_minerva}
        exec(compile(tree, TEST, "exec"), namespace)
        results = []
        for name, decorators in functions:
            params = [()]
            if decorators:
                assert len(decorators) == 1
                call = decorators[0]
                assert ast.unparse(call.func) == "pytest.mark.parametrize"
                assert len(call.args) == 2 and not call.keywords
                arg_name = ast.literal_eval(call.args[0])
                assert arg_name in ("solution_str", "answer")
                value_node = call.args[1]
                values = namespace[value_node.id] if isinstance(value_node, ast.Name) else ast.literal_eval(value_node)
                params = [(value,) for value in values]
            for index, param in enumerate(params):
                try:
                    namespace[name](*param)
                    status = "passed"
                except AssertionError:
                    status = "assertion_failed"
                results.append({"case": f"{name}[{index}]", "status": status})
        assert len(results) == 21
        return {"passed": sum(r["status"] == "passed" for r in results),
                "assertion_failed": sum(r["status"] == "assertion_failed" for r in results),
                "items": results}

    old_tests, new_tests = original_tests(base), original_tests(head)
    assert (old_tests["passed"], old_tests["assertion_failed"]) == (7, 14)
    assert (new_tests["passed"], new_tests["assertion_failed"]) == (21, 0)

    probes = [
        ("backtick label across real newlines", "`Answer:`\n\n42", "42", False),
        ("bold label ends before colon", "**Answer**: 42", "42", False),
        ("literal trailing star correct target", "Answer: x*", "x*", False),
        ("literal trailing star different target", "Answer: x*", "x", False),
        ("empty decorated label permits boxed fallback", "\\boxed{42}\n**Answer:**", "42", False),
        ("wrong but valid Minerva blocks boxed fallback", "\\boxed{42}\nAnswer: 41", "42", False),
        ("strict box ignores wrong Minerva", "\\boxed{42}\nAnswer: 41", "42", True),
        ("boxed comma normalization not added", "\\boxed{1,000}", "1000", False),
    ]
    rows = []
    expected_scores = [(-1, -1), (-1, -1), (1, -1), (-1, 1), (-1, 1), (-1, -1), (1, 1), (-1, -1)]
    for (name, text, target, strict), expected in zip(probes, expected_scores):
        old = base.compute_score(text, target, strict_box_verify=strict)
        new = head.compute_score(text, target, strict_box_verify=strict)
        assert (old["score"], new["score"]) == expected, (name, old, new)
        rows.append({"case": name, "input": text, "target": target, "strict": strict,
                     "base": old, "head": new})
    print(json.dumps({"python": sys.version.split()[0], "base": BASE, "main": MAIN, "head": HEAD,
                      "original_test_bodies": {"base": old_tests, "head": new_tests},
                      "local_boundary_probes": rows, "source_sha256": hashes,
                      "scope": "complete original scorer modules and 21 original test bodies; 8 local probes; no native pytest, manager or training"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
