"""Run PR #8046's nine new policy cases, not vLLM sleep or weight restoration."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from types import ModuleType

import pytest

HEAD = "576b8e91a1f4df718e5dcf5171efdff8c331825e"
BASE = "2035f58dcb06d8cad4caee107b30f4045d724367"
SERVER = "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
TEST = "tests/workers/rollout/rollout_vllm/test_mtp_hybrid_sleep_acceptance_on_cpu.py"
METHODS = {"lora_as_adapter", "_lora_configured", "_merged_lora",
           "_fsdp_peft_merged_lora", "_resolve_sleep_level", "_sleep_hybrid"}
TESTS = ["test_hybrid_full_param_sleep_discards_weights",
         "test_hybrid_fsdp_merged_lora_sleep_keeps_weights",
         "test_hybrid_megatron_merged_lora_sleep_discards_weights",
         "test_hybrid_lora_adapter_sleep_keeps_weights"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    def source(ref, path):
        return subprocess.check_output(["git", "-C", str(args.repo), "show", f"{ref}:{path}"])

    test_source = source(HEAD, TEST)
    test_tree = ast.parse(test_source)
    kept = [n for n in test_tree.body if
            (isinstance(n, ast.Import) and all(a.name in {"asyncio", "pytest"} for a in n.names))
            or (isinstance(n, ast.ImportFrom) and n.module == "types")]
    kept += [n for n in test_tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in {"_FakeSleepEngine", "_hybrid_server", *TESTS}]
    # Do not execute importorskip, framework imports, or the older fake-MTP case.
    results = {"scope": __doc__, "python": platform.python_version(),
               "pytest": pytest.__version__, "head": HEAD, "base": BASE,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "original_test_sha256": hashlib.sha256(test_source).hexdigest(), "runs": {}}
    for ref in (BASE, HEAD):
        server_source = source(ref, SERVER)
        server_tree = ast.parse(server_source)
        original = next(n for n in server_tree.body if isinstance(n, ast.ClassDef) and n.name == "vLLMHttpServer")
        shell = ast.ClassDef(name="vLLMHttpServer", bases=[], keywords=[], decorator_list=[],
                             body=[n for n in original.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                   and n.name in METHODS])
        server = ModuleType("isolated_vllm_async_server")
        server.is_torch_npu_available = lambda check_device=False: False
        exec(compile(ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[])), SERVER, "exec"), server.__dict__)
        ns = {"vllm_async_server": server}
        exec(compile(ast.Module(body=kept, type_ignores=[]), TEST, "exec"), ns)
        cases = []
        for name in TESTS:
            fn = ns[name]
            marks = getattr(fn, "pytestmark", [])
            params = next((list(m.args[1]) for m in marks if m.name == "parametrize"), ["not-parametrized"])
            for param in params:
                with pytest.MonkeyPatch.context() as monkeypatch:
                    try:
                        if marks:
                            fn(monkeypatch, param)
                        else:
                            fn(monkeypatch)
                        outcome = "passed"
                    except AssertionError:
                        outcome = "assertion_failed"
                    cases.append({"name": name, "strategy": param, "outcome": outcome})
        failures = [c for c in cases if c["outcome"] != "passed"]
        assert len(cases) == 9
        assert len(failures) == (3 if ref == BASE else 0)
        if ref == BASE:
            assert {c["strategy"] for c in failures} == {"fsdp", "fsdp2", "fsdp_turbo"}
        results["runs"][ref] = {"server_sha256": hashlib.sha256(server_source).hexdigest(),
                                 "cases": cases, "passed": len(cases) - len(failures),
                                 "assertion_failed": len(failures)}
    output = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
