"""Original full-stream helpers with own CPU inputs and explicit model/mesh seams."""

import argparse
import ast
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

HEAD = "c1eca9d05a03c2b5f85a145bddd3f5f91b333a68"
OLD = "0ed94ba020747101bb22db6005849c9a6ca06aa6"
UTIL = "verl/models/mcore/util.py"
FUSED = "verl/models/mcore/model_forward_fused.py"
READING = ("verl/models/mcore/model_forward.py", "tests/models/test_mcore_full_stream_thd_on_cpu.py",
           "tests/models/test_glm53flash_model_forward.py", "tests/models/test_model_forward_fused.py")


def extract(raw, namespace, names):
    nodes = [node for node in ast.parse(raw).body
             if (isinstance(node, ast.FunctionDef) and node.name in names)
             or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "original_helpers", "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources, hashes = {}, {}
    for ref, paths in ((OLD, (FUSED,)), (HEAD, (UTIL, FUSED, *READING))):
        for path in paths:
            raw = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=args.repo,
                                          env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
            compile(ast.parse(raw), path, "exec")
            sources[ref, path] = raw
            hashes[f"{ref}:{path}"] = hashlib.sha256(raw).hexdigest()
    ns = {"torch": torch, "PackedSeqParams": object, "contextmanager": contextmanager,
          "mpu": SimpleNamespace(get_context_parallel_world_size=lambda: 2)}
    extract(sources[HEAD, UTIL], ns,
            {"preprocess_thd_full_stream", "check_full_stream_cp_layout", "language_model_forward_kwargs"})
    nested = torch.nested.nested_tensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8])],
                                        layout=torch.jagged)
    packed = SimpleNamespace(cu_seqlens_q_padded=torch.tensor([0, 4, 12]))
    layouts = []
    expected = torch.tensor([[1, 2, 3, 0, 4, 5, 6, 7, 8, 0, 0, 0]])
    for roll in (False, True):
        actual = ns["preprocess_thd_full_stream"](nested, packed, need_roll=roll)
        torch.testing.assert_close(actual, expected.roll(-1, 1) if roll else expected)
        layouts.append({"need_roll": roll, "stream": actual.tolist()})
    guards = []
    for kwargs in ({"local_cp_size": 2}, {"cp_layout": "contiguous"}):
        with pytest.raises(ValueError) as error:
            ns["check_full_stream_cp_layout"](**kwargs)
        guards.append(str(error.value))
    ns["check_full_stream_cp_layout"](cp_layout="zigzag")
    ns["mpu"].get_context_parallel_world_size = lambda: 1
    ns["check_full_stream_cp_layout"](cp_layout="contiguous")

    class Inner(torch.nn.Module):
        def forward(self, output_processor=None, output_processor_context=None, **kwargs):
            return dict(kwargs, output_processor=output_processor, output_processor_context=output_processor_context)

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Inner()

        def forward(self, **kwargs):
            return self.language_model(**kwargs)

    mode_names = {"_HOOK_MODE", "_LANGUAGE_MODEL_HOOK_MODE", "_LEGACY_MODE",
                  "_supports_output_processor_hook", "_resolve_fused_forward_mode"}
    modes = []
    for label, model in (("inner_explicit_hook", Outer()), ("outer_explicit_hook", Inner()),
                         ("neither_explicit_hook", torch.nn.Module())):
        found = {}
        for ref in (OLD, HEAD):
            mode_ns = {"torch": torch, "inspect": inspect}
            extract(sources[ref, FUSED], mode_ns, mode_names)
            found[ref] = mode_ns["_resolve_fused_forward_mode"](model)
        modes.append({"model": label, "old": found[OLD], "head": found[HEAD]})
    assert [(item["old"], item["head"]) for item in modes] == [
        ("legacy", "language_model_hook"), ("hook", "hook"), ("legacy", "legacy")]

    wrapper = Outer()
    seam = ModuleType("verl.utils.megatron_utils")
    seam.unwrap_model = lambda value: value  # plain CPU module only, not wrapper-class identification
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, seam.__name__, seam)
        inject = ns["language_model_forward_kwargs"]
        with inject(wrapper, compute_mtp_loss=False, output_processor="hook"):
            injected = wrapper.language_model(compute_mtp_loss=True, input_ids=1)
        assert injected["compute_mtp_loss"] is False and injected["output_processor"] == "hook"
        assert "forward" not in vars(wrapper.language_model)
        existing = wrapper.language_model.forward
        wrapper.language_model.forward = existing
        with pytest.raises(RuntimeError):
            with inject(wrapper, compute_mtp_loss=False):
                raise RuntimeError("own context failure")
        assert vars(wrapper.language_model)["forward"] is existing
        with inject(wrapper, compute_mtp_loss=False):
            with inject(wrapper, compute_mtp_loss=True):
                nested_call = wrapper.language_model()
            after_inner = wrapper.language_model()
        assert nested_call["compute_mtp_loss"] is False and after_inner["compute_mtp_loss"] is False
        assert vars(wrapper.language_model)["forward"] is existing
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "head": HEAD, "old": OLD,
                      "source_sha256": hashes, "own_layout_controls": layouts, "own_cp_guard_errors": guards,
                      "own_mode_controls": modes, "own_context_controls": {
                          "injected_call_overrides_callsite": injected, "normal_class_forward_restored": True,
                          "exception_restores_existing_instance_forward": True,
                          "nested_context_outer_value_wins": nested_call["compute_mtp_loss"],
                          "after_inner_outer_value": after_inner["compute_mtp_loss"],
                          "after_nested_existing_forward_restored": True},
                      "scope": "Seven complete inputs compile. Exact original three util helpers and two mode "
                      "resolver functions/constants; own real CPU jagged tensor with supplied padded metadata, "
                      "plain nn.Module signature/forward fixtures, fake mesh-size provider and identity unwrap "
                      "import seam. Two layouts, two guard groups, three mode and three context groups. "
                      "No original test bodies, original preprocess_thd_engine, actual Bridge GLM class detector, "
                      "model engine forward, real wrapper unwrapping, CP/SP/TP collectives, MTP computation, "
                      "fusion kernels, GPU/NPU, model/training or performance."}, indent=2))


if __name__ == "__main__":
    main()
