#!/usr/bin/env python3
"""Check UniRL #617 original metadata and CPU view helpers, NOT CUDA export/IPC.

Run: python -B experiments/scripts/verify_unirl_export_metadata.py /path/to/UniRL
Requires Python >=3.10, torch, and the fixed Git object already present locally.
No network fetch, package shell, AST extraction, model backend or device probing.
"""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import torch

REV = "19f263968deecdecd26cebe8966f8ba012a5575d"
PATH = "unirl/utils/peft_merge.py"
SHA256 = "e03bf04d38966e548bf0a0f742260dc805cf29ec96270e5ffe3eb8526abc8798"


class StateFixture(torch.nn.Module):
    """Hand-authored state layout, not a Qwen model or inference backend."""
    def __init__(self, tied):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tied, model_type="ordinary")
        self.embed_tokens = torch.nn.Module()
        weight = torch.arange(20, dtype=torch.float64, device="cpu").reshape(5, 4)
        self.embed_tokens.register_parameter("weight", torch.nn.Parameter(weight))
        self.lm_head = torch.nn.Module()
        output = self.embed_tokens.weight if tied else torch.nn.Parameter(torch.ones(5, 4, dtype=torch.float64, device="cpu"))
        self.lm_head.register_parameter("weight", output)
        self.register_buffer("gain", torch.tensor(2.0, dtype=torch.float32, device="cpu"))
        self.register_buffer("count", torch.tensor(3, dtype=torch.int64, device="cpu"))


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")
    source = subprocess.check_output(["git", "-C", str(Path(sys.argv[1]).resolve()), "show", f"{REV}:{PATH}"], env=env)
    assert hashlib.sha256(source).hexdigest() == SHA256, "fixed source hash mismatch"
    ns = {"__name__": "unirl617_original_export"}
    exec(compile(source, PATH, "exec"), ns)
    print("Python", sys.version.split()[0], "torch", torch.__version__, "git", torch.version.git_version)
    print("complete original module", REV, SHA256)

    for tied in (False, True):
        model = StateFixture(tied)
        for dtype in (None, torch.float32, torch.bfloat16):
            metadata = list(ns["raw_state_dict_metadata"](model, dtype=dtype))
            expected_names = ["gain", "count", "embed_tokens.weight"] + ([] if tied else ["lm_head.weight"])
            assert [name for name, _ in metadata] == expected_names
            for name, tensor in metadata:
                original = model.state_dict()[name]
                assert tensor.is_meta and tensor.shape == original.shape
                expected_dtype = dtype if dtype is not None and original.is_floating_point() else original.dtype
                assert tensor.dtype == expected_dtype
            print("metadata tied=", tied, "wire=", dtype, [(name, tuple(t.shape), str(t.dtype)) for name, t in metadata])

    gate = torch.arange(48, dtype=torch.float32, device="cpu").reshape(2, 6, 4)
    down = torch.arange(24, dtype=torch.float32, device="cpu").reshape(2, 4, 3)
    base = "model.layers.0.mlp.experts"
    expected_gate = [(f"{base}.{e}.{kind}.weight", gate[e, s:s+3])
                     for e in range(2) for kind, s in [("gate_proj", 0), ("up_proj", 3)]]
    expected_down = [(f"{base}.{e}.down_proj.weight", down[e]) for e in range(2)]
    for suffix, tensor, expected in [("gate_up_proj", gate, expected_gate), ("down_proj", down, expected_down)]:
        outputs = list(ns["_iter_rollout_tensor_views"](f"{base}.{suffix}", tensor, unpack_qwen_moe=True, moe_intermediate=3))
        assert [name for name, _ in outputs] == [name for name, _ in expected]
        for (_, actual), (_, wanted) in zip(outputs, expected):
            assert torch.equal(actual, wanted) and actual.device.type == "cpu"
            assert actual.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        print("CPU packed views:", suffix, [(name, tuple(t.shape)) for name, t in outputs])

    for suffix, shape in [("gate_up_proj", (2, 5, 4)), ("gate_up_proj", (2, 8, 4)), ("down_proj", (2, 4, 2))]:
        tensor = torch.empty(shape, device="cpu")
        try:
            list(ns["_iter_rollout_tensor_views"](f"{base}.{suffix}", tensor, unpack_qwen_moe=True, moe_intermediate=3))
        except RuntimeError:
            pass
        else:
            raise AssertionError("malformed packed shape accepted")
    print("6 metadata cases, 2 CPU packed-view layouts and 3 malformed-shape rejections passed")
    print("NOT RUN: raw_state_dict/merged_state_dict/_to_full_tensor (CUDA), DTensor collectives, PEFT, IPC, model/reload or device discovery")


if __name__ == "__main__":
    main()
