"""Run #8311's original CPU tests with original leaf modules, no model downloads."""
from __future__ import annotations

import __future__
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

HEAD = "9fc9e2646e4543c3cba1eef8f490622499583911"
BASE = "5b550b0c3dad8c7bb7aab834a3fe195f8cdfc517"


def main() -> int:
    import pytest

    repo = Path(sys.argv[1]).resolve()
    revision = sys.argv[2] if len(sys.argv) > 2 else HEAD
    if revision not in (HEAD, BASE):
        raise ValueError("Use one of the two reviewed fixed revisions")

    def read(rev: str, path: str) -> bytes:
        data = subprocess.check_output(["git", "show", f"{rev}:{path}"], cwd=repo,
                                       env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        print(hashlib.sha256(data).hexdigest(), rev, path, flush=True)
        return data

    # Package shells only avoid unrelated verl initialization. All called
    # tokenizer/builder/normalization functions below are the original code.
    for name in ("verl", "verl.utils", "verl.utils.tokenizer"):
        package = ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    for leaf in ("tokenizer", "chat_template", "continuous_token", "deepseek", "continuous_token_wiring"):
        name = "verl.utils.tokenizer." + leaf
        module = ModuleType(name)
        module.__package__ = "verl.utils.tokenizer"
        module.__file__ = f"git:{revision}:verl/utils/tokenizer/{leaf}.py"
        sys.modules[name] = module
        data = read(revision, f"verl/utils/tokenizer/{leaf}.py")
        if leaf == "chat_template":
            # Transformers is unavailable. Only this original function is used;
            # its two Transformers types occur solely in the annotation.
            tree = ast.parse(data)
            nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_chat_template"]
            assert len(nodes) == 1
            code = compile(ast.Module(body=nodes, type_ignores=[]), module.__file__, "exec",
                           flags=__future__.annotations.compiler_flag, dont_inherit=True)
        else:
            code = compile(data, module.__file__, "exec", dont_inherit=True)
        exec(code, module.__dict__)

    tests = ModuleType("original_continuous_token_cpu_tests")
    tests.__file__ = str(Path(__file__).resolve())
    sys.modules[tests.__name__] = tests
    exec(compile(read(HEAD, "tests/utils/test_continuous_token_on_cpu.py"),
                 "git:#8311:tests/utils/test_continuous_token_on_cpu.py", "exec", dont_inherit=True), tests.__dict__)

    class OriginalModule(pytest.Module):
        def _getobj(self):
            return tests

    class OriginalCollection:
        def pytest_pycollect_makemodule(self, module_path, parent):
            if module_path == Path(__file__).resolve():
                return OriginalModule.from_parent(parent, path=module_path)

    def no_network(event, args):
        if event.startswith("socket."):
            raise RuntimeError("Network is outside this CPU test's scope")

    sys.addaudithook(no_network)
    result = pytest.main([str(Path(__file__).resolve()), "--noconftest",
                          "--confcutdir=" + str(Path(__file__).parent), "-o", "addopts=",
                          "-p", "no:cacheprovider", "-q", "--tb=short"], plugins=[OriginalCollection()])
    assert all(name not in sys.modules for name in ("torch", "transformers", "ray", "vllm"))
    print("Original test tokenizers/processors, not real HF tokenizers or rollout integration", flush=True)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
