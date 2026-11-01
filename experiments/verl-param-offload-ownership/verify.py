"""Run 19 original CPU test parameters against extracted offload helpers.

No Megatron import or native pytest collection. Original pin/broadcast mocks stay
in force; real CPU tensors check storage ownership, not device transfer.
"""

import argparse
import ast
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, create_autospec

import pytest
import torch

HEAD = "81a672a48c1e3b76041b7f07c16343799b971796"
BASE = "2035f58dcb06d8cad4caee107b30f4045d724367"
SOURCE = "verl/utils/megatron_utils.py"
TEST = "tests/utils/megatron/test_param_offload_deduplication.py"
READING = ("verl/workers/config/engine.py",
           "verl/workers/engine/megatron/transformer_impl.py",
           "verl/experimental/separation/engine_workers.py",
           "tests/workers/config/test_engine_config_on_cpu.py")
HELPERS = {
    "MegatronParamOffloadGroup", "MegatronParamOffloadGroups",
    "_make_param_offload_group", "_get_megatron_param_replica_process_groups",
    "build_megatron_param_offload_groups", "_parameter_offload_group",
    "_ddp_param_buffer_groups", "_broadcast_offloaded_parameter",
    "_offload_ddp_param_buffer", "_load_ddp_param_buffer",
    "_offload_replicated_parameter", "_load_replicated_parameter",
    "_can_safely_resize_storage", "copy_megatron_model_to_cpu",
    "restore_megatron_model_from_cpu",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources, hashes = {}, {}
    for path in (SOURCE, TEST, *READING):
        raw = subprocess.check_output(["git", "show", f"{HEAD}:{path}"], cwd=args.repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        compile(ast.parse(raw), path, "exec")
        sources[path] = raw
        hashes[path] = hashlib.sha256(raw).hexdigest()
    module = ModuleType("isolated_megatron_offload_helpers")
    module.__dict__.update(torch=torch, inspect=inspect, dataclass=dataclass, Any=Any,
                           parallel_state=SimpleNamespace(get_data_parallel_group=lambda: None,
                                                          get_expert_data_parallel_group=lambda: None),
                           DDP=type("UnusedDDPImportSeam", (), {}))
    nodes = [node for node in ast.parse(sources[SOURCE]).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in HELPERS]
    assert {node.name for node in nodes} == HELPERS
    with pytest.MonkeyPatch.context() as registration:
        registration.setitem(sys.modules, module.__name__, module)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), SOURCE, "exec"), module.__dict__)
        namespace = dict(torch=torch, pytest=pytest, SimpleNamespace=SimpleNamespace,
                         Mock=Mock, create_autospec=create_autospec, megatron_utils=module)
        tests = [node for node in ast.parse(sources[TEST]).body if isinstance(node, ast.FunctionDef)]
        for node in tests:
            node.decorator_list = []  # manual original parameter grid and original fixture
        exec(compile(ast.Module(body=tests, type_ignores=[]), TEST, "exec"), namespace)
        reports = []

        def run(name, parameters=None, use_fixture=False):
            parameters = parameters or {}
            function = namespace[name]
            with pytest.MonkeyPatch.context() as monkeypatch:
                if use_fixture:
                    namespace["cpu_offload"](monkeypatch)
                kwargs = dict(parameters)
                if "monkeypatch" in inspect.signature(function).parameters:
                    kwargs["monkeypatch"] = monkeypatch
                if "cpu_offload" in inspect.signature(function).parameters:
                    kwargs["cpu_offload"] = None
                function(**kwargs)
            reports.append(dict(test=name, parameters=parameters, outcome="passed"))

        for supported in (False, True):
            run("test_param_offload_group_builder", {"supports_gtp_remat": supported})
        run("test_param_offload_group_builder_returns_none_without_replicas")
        for owner in (None, True, False):
            run("test_ddp_offload_round_trip", {"is_owner": owner}, True)
        run("test_deduplicated_buffer_requires_its_group_on_reload")
        run("test_repeated_default_offload_validates_cpu_backing_size")
        run("test_expert_parameter_uses_expert_replica_group")
        for offloaded in (False, True):
            for owner in (None, True, False):
                run("test_ddp_snapshot_round_trip", {"is_owner": owner, "restore_offloaded": offloaded}, True)
            for owner in (True, False):
                run("test_non_ddp_snapshot_round_trip", {"is_owner": owner, "restore_offloaded": offloaded}, True)
        assert len(reports) == 19
    print(json.dumps({"head": HEAD, "base": BASE, "python": platform.python_version(),
                      "torch": torch.__version__, "torch_git": torch.version.git_version,
                      "pytest": pytest.__version__, "source_sha256": hashes,
                      "extracted_definitions": sorted(HELPERS), "original_test_cases": reports,
                      "scope": "15 complete production definitions extracted via AST, decorators retained. "
                      "All original functions in the focused test module extracted; only decorators removed, "
                      "bodies/assertions unchanged; 19 parameter cases run manually, not native pytest. "
                      "Original CPU fixture disables pinning; original broadcast mocks supply expected data. "
                      "SimpleNamespace parallel-state/DDP import seams, actual CPU tensors/storage/strides. "
                      "Six complete source files compile. No full module import, config test execution, "
                      "offload/load engine entry, real Megatron groups/DDP, collective, pinned allocator, "
                      "device copies/streams, optimizer/gradient equivalence, durable checkpoint, GPU/NPU, "
                      "training, memory-peak measurement or performance."}, indent=2))


if __name__ == "__main__":
    main()
