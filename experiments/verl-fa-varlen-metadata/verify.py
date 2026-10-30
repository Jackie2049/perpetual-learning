"""Original #8464 CPU test bodies + fixed Transformers boundary helper.

Usage: python -B verify.py /path/to/verl-git /path/to/modeling_flash_attention_utils.py
The second input is the unmodified file at HF commit 93c8b7b485963a10800c91f55304db6be211c2bd.
Functions/constants are AST-isolated; imports/framework initialization are skipped.
Original pytest decorators and assertions stay intact, but direct function calls
are not native pytest collection. No FSDP, HF model, flash-attn, device discovery,
GPU/NPU, gradients or timing. Original sources: verl / Transformers, Apache-2.0.
"""

import __future__
import argparse
import ast
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace

import pytest
import torch

HEAD = "fc66f413f71c0c85a37847aa9e2b50e9bbea225b"


def select(raw, names, namespace):
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if getattr(n, "name", None) in names or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    assert len(nodes) == len(names), names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "original-definitions", "exec",
                 flags=__future__.annotations.compiler_flag), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("hf_source", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    ns = dict(torch=torch, pytest=pytest, SimpleNamespace=SimpleNamespace)
    hashes = {}
    inputs = {}
    for path in ["verl/workers/engine/utils.py", "verl/workers/engine/fsdp/transformer_impl.py",
                 "verl/models/transformers/dense_common.py", "tests/workers/test_fsdp_flash_attn_varlen_kwargs_on_cpu.py"]:
        raw = subprocess.check_output(["git", "show", HEAD + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        compile(raw, path, "exec")
        hashes[path] = hashlib.sha256(raw).hexdigest()
        inputs[path] = raw
    reference = args.hf_source.read_bytes()
    hashes["transformers@93c8b7b4:modeling_flash_attention_utils.py"] = hashlib.sha256(reference).hexdigest()
    assert hashlib.sha256(reference).hexdigest() == "dea5b555b5bb8fba1b4ab2c62c8df9f96019b17555b3714f75be17c54e801f95"
    select(reference, {"prepare_fa_kwargs_from_position_ids"}, ns)
    select(inputs["verl/workers/engine/utils.py"], {"pad_packed_inputs", "build_flash_attn_varlen_kwargs"}, ns)
    select(inputs["verl/workers/engine/fsdp/transformer_impl.py"],
           {"_FLASH_ATTN_VARLEN_IMPLEMENTATIONS", "_supports_flash_attn_varlen_kwargs"}, ns)
    raw_tests = inputs["tests/workers/test_fsdp_flash_attn_varlen_kwargs_on_cpu.py"]
    names = {n.name for n in ast.parse(raw_tests).body if isinstance(n, ast.FunctionDef)}
    select(raw_tests, names, ns)
    count = 0
    for name in sorted(n for n in names if n.startswith("test_")):
        fn = ns[name]
        marks = getattr(fn, "pytestmark", [])
        combinations = itertools.product(*[mark.args[1] for mark in marks]) if marks else [()]
        for values in combinations:
            kwargs = {}
            for mark, value in zip(marks, values):
                keys = [k.strip() for k in mark.args[0].split(",")]
                kwargs.update(zip(keys, value if len(keys) > 1 else (value,)))
            fn(**kwargs)
            count += 1
    assert count == 15
    build = ns["build_flash_attn_varlen_kwargs"]
    kwargs = build(torch.tensor([0, 5, 8]), pad_size=8)
    assert kwargs["cu_seq_lens_q"].tolist() == [0, 5, 8, 16] and kwargs["max_length_q"] == 8
    # A local signature fixture, not an actual remote-code model or fused wrapper.
    def strict_forward(input_ids=None, position_ids=None):
        return None
    fixture = ns["_module"]("flash_attention_2")
    fixture.forward = strict_forward
    assert ns["_supports_flash_attn_varlen_kwargs"](fixture)
    try:
        inspect.signature(fixture.forward).bind(input_ids=None, position_ids=None, **kwargs)
    except TypeError as exc:
        strict_error = str(exc)
        assert "unexpected keyword" in strict_error
    else:
        raise AssertionError("config eligibility is not forward compatibility")
    # Local zero-length-row contrast: producer admissibility is not established here.
    zero_row = build(torch.tensor([0, 5, 5, 8]))["cu_seq_lens_q"]
    positions = ns["_packed_position_ids"]([5, 0, 3])
    inferred = ns["prepare_fa_kwargs_from_position_ids"](positions)[0][0]
    assert zero_row.tolist() == [0, 5, 5, 8] and inferred.tolist() == [0, 5, 8]
    print(json.dumps(dict(head=HEAD, source_sha256=hashes,
        environment=dict(python=platform.python_version(), torch=torch.__version__, pytest=pytest.__version__, device="cpu"),
        original_cpu_parameter_cases_passed=count,
        static_pad=dict(offsets=kwargs["cu_seq_lens_q"].tolist(), max_length=kwargs["max_length_q"]),
        local_controls=dict(config_gate_accepts_strict_forward=True, signature_bind_error=strict_error,
                            zero_row_offsets=zero_row.tolist(), zero_row_inferred_offsets=inferred.tolist()),
        scope="AST-isolated original CPU tests/reference, local signature and zero-row fixtures; no model/engine/attention/gradient/timing validation"),
        ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
