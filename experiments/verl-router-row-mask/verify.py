"""Run the original CPU controller pytest module in memory, plus value/mask controls.

Bypasses verl package initialization only. No VeOmni model/EP/backend execution.
"""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType

import pytest
import torch

HEAD = "7ad2b4fbd6f0afbd59b89c3022847ebb35bd9b1e"
SOURCE = "verl/utils/veomni/router_replay.py"
TESTS = "tests/utils/veomni/test_router_replay_on_cpu.py"
READING = ("verl/experimental/agent_loop/agent_loop.py", "verl/workers/engine/veomni/transformer_impl.py",
           "verl/workers/utils/padding.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources, hashes = {}, {}
    for path in (SOURCE, TESTS, *READING):
        raw = subprocess.check_output(["git", "show", f"{HEAD}:{path}"], cwd=args.repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        compile(ast.parse(raw), path, "exec")
        sources[path] = raw
        hashes[path] = hashlib.sha256(raw).hexdigest()
    module = ModuleType("verl.utils.veomni.router_replay")
    module.__file__ = f"{HEAD}:{SOURCE}"
    tests = ModuleType("original_router_replay_cpu_tests")
    tests.__file__ = str(Path(__file__).resolve())
    reports = []

    class OriginalModule(pytest.Module):
        def _getobj(self):
            return tests

    class Plugin:
        @pytest.hookimpl(tryfirst=True)
        def pytest_pycollect_makemodule(self, module_path, parent):
            return OriginalModule.from_parent(parent, path=module_path)

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or report.failed:
                reports.append({"test": report.nodeid.rsplit("::", 1)[-1], "outcome": report.outcome})

    captured = io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("VERL_ROUTER_REPLAY_DEBUG", raising=False)
        mp.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        mp.setitem(sys.modules, module.__name__, module)
        mp.setitem(sys.modules, tests.__name__, tests)
        exec(compile(sources[SOURCE], SOURCE, "exec"), module.__dict__)
        exec(compile(sources[TESTS], TESTS, "exec"), tests.__dict__)
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            status = pytest.main(["-q", "--noconftest", "--import-mode=importlib", str(Path(__file__).resolve())],
                                 plugins=[Plugin()])
        if status != 0:
            raise AssertionError(captured.getvalue())
        assert len(reports) == 22 and all(item["outcome"] == "passed" for item in reports)

        def replay(target, native, mask, debug):
            mp.setenv("VERL_ROUTER_REPLAY_DEBUG", "1" if debug else "0")
            controller = module.VeOmniRouterReplay()
            controller.begin_replay()
            controller.begin_microbatch(targets=[target], mask=mask)
            return controller.on_router_forward(torch.nn.Identity(), torch.zeros(target.shape[0], 8), native)

        target = torch.tensor([[0], [3]])
        native = torch.tensor([[7], [7]])
        fallback = replay(target, native, None, False).tolist()
        explicit = replay(target, native, torch.tensor([True, False]), False).tolist()
        try:
            replay(target, native, torch.tensor([True, False]), True)
        except AssertionError as error:
            debug_zero = str(error)
        else:
            raise AssertionError("Expected original debug guard to reject masked-in expert zero")
        duplicate = replay(torch.tensor([[5, 5]]), torch.tensor([[1, 2]]), torch.tensor([True]), True).tolist()
        assert fallback == [[7], [3]] and explicit == [[0], [7]] and duplicate == [[5, 5]]
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "pytest": pytest.__version__,
                      "head": HEAD, "source_sha256": hashes, "native_pytest_controller_reports": reports,
                      "own_controls": {"topk1_no_mask": fallback, "topk1_explicit_mask_debug_off": explicit,
                                       "topk1_explicit_mask_debug_on_error": debug_zero,
                                       "nonzero_duplicate_masked_in_debug_on": duplicate},
                      "scope": "Native pytest collection/fixtures/assertions from the unchanged original 22-test "
                      "controller module, loaded in memory through a custom pytest.Module; complete production "
                      "controller module, only verl package initialization bypassed. Original router nn.Module "
                      "fixtures and install/uninstall VeOmni hook stubs are not real SparseMoeBlock/EP execution. "
                      "Four own CPU value/mask controls. Five full inputs compile; producer/padding/engine modules "
                      "read and compile only. No VeOmni backend/model, TQ, SP, EP, training, GPU/NPU or performance."},
                     indent=2))


if __name__ == "__main__":
    main()
