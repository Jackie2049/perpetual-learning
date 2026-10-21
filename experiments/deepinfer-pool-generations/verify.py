"""Run #187's unchanged tests against original CPU pool classes, without package init.

Real torch CPU tensors, shared storage, multiprocessing locks and fork are used.
Only host pin/unpin is replaced by the original test fixture. No device work.
"""
from __future__ import annotations

import ast
import hashlib
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

BASE = "9bd9cf243fc53ef9afbf8f04d391d989117d6ab9"
HEAD = "5c2bb4625506e5cec761eb5a7f08f4336ab45bd7"


def main() -> int:
    import pytest
    import torch

    repo = Path(sys.argv[1]).resolve()
    revision = sys.argv[2] if len(sys.argv) > 2 else HEAD
    if revision not in (BASE, HEAD):
        raise ValueError("Only the two reviewed fixed revisions are supported")
    torch.set_num_threads(1)

    def read(rev: str, path: str) -> bytes:
        data = subprocess.check_output(
            ["git", "show", f"{rev}:{path}"], cwd=repo,
            env=dict(os.environ, GIT_NO_LAZY_FETCH="1"),
        )
        print(hashlib.sha256(data).hexdigest(), rev, path, flush=True)
        return data

    source = read(revision, "deepinfer/shared_memory.py")
    tests = read(HEAD, "tests/test_shared_pool_generations.py")
    names = {"BlockHandle", "_shared_array", "SharedTorchMemoryPool"}
    tree = ast.parse(source)
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in selected} == names
    # Preserve every selected node, including decorators, method bodies and imports
    # needed by those nodes. Unused posix_ipc SharedMemory and lazy pool factory
    # are deliberately not loaded, nor is DeepInfer's platform discovery.
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
               and not (isinstance(node, ast.Import) and any(alias.name == "posix_ipc" for alias in node.names))
               and not (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("deepinfer"))]

    package = ModuleType("deepinfer")
    package.__path__ = []
    module = ModuleType("deepinfer.shared_memory")
    module.__file__ = f"git:{revision}:deepinfer/shared_memory.py"

    def forbidden_pin(*args: object) -> None:
        raise AssertionError("The original fixture must replace host pin/unpin")

    module.PLATFORM = SimpleNamespace(pin_host_memory=forbidden_pin, unpin_host_memory=forbidden_pin)
    module.logger = logging.getLogger("isolated-original-pool")
    package.shared_memory = module
    sys.modules["deepinfer"] = package
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=imports + selected, type_ignores=[]), module.__file__, "exec"), module.__dict__)

    original_tests = ModuleType("original_pool_generation_tests")
    original_tests.__file__ = str(Path(__file__).resolve())
    sys.modules[original_tests.__name__] = original_tests
    exec(compile(tests, "git:#187:tests/test_shared_pool_generations.py", "exec"), original_tests.__dict__)

    class OriginalModule(pytest.Module):
        def _getobj(self):
            return original_tests

    class OriginalCollection:
        def pytest_pycollect_makemodule(self, module_path, parent):
            if module_path == Path(__file__).resolve():
                return OriginalModule.from_parent(parent, path=module_path)

    # Collect the unchanged in-memory test module using this existing file as
    # a collection anchor, so neither source nor temporary test files are written.
    print("Original pool classes + original 9 tests; CPU pin/unpin fixture only", flush=True)
    result = pytest.main([
        str(Path(__file__).resolve()), "--noconftest", "--confcutdir=" + str(Path(__file__).parent),
        "-o", "addopts=", "-o", "markers=ci: CPU isolated check\nci_npu: original upstream marker",
        "-p", "no:cacheprovider", "-q", "--tb=short",
    ], plugins=[OriginalCollection()])
    if result == 0:
        # Separate, self-authored lifetime control; NOT an extra upstream test.
        # Reuse the unchanged upstream fixture, including its pin/unpin no-ops.
        with pytest.MonkeyPatch.context() as patch:
            fixture = original_tests.pool.__wrapped__(patch)
            pool = next(fixture)
            try:
                old = pool.alloc(512)
                retained = pool.get_buffer_slice(old)
                pool.free(old)
                current = pool.alloc(512)
                live = pool.get_buffer_slice(current)
                live.fill_(17)
                assert torch.all(retained == 17)
                retained[0] = 73
                assert live[0].item() == 73
                pool.free(current)
                del retained, live
                print("1 separate retained-view control: old view aliases the reused allocation", flush=True)
            finally:
                fixture.close()
    assert "deepinfer.platform" not in sys.modules
    assert "deep_cluster" not in sys.modules
    print("No platform discovery, device registration or full package initialization", flush=True)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
