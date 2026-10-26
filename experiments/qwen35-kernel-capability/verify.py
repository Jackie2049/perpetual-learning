#!/usr/bin/env python3
"""Conditional CPU reference control for verl #8390, not model compatibility.

Read six hash-pinned source copies. Execute original signature/packing helpers
and the official Torch reference body with its Hub decorator explicitly removed.
This does NOT execute the Hub dispatcher, FLA, full GDN, SP, or installed
Transformers. A reference-body counterexample is not a new-regression verdict.
"""

import argparse
import ast
import copy
import hashlib
from inspect import signature
import json
from pathlib import Path
import sys
from typing import Optional

import torch
import torch.nn.functional as F

HASHES = {
    "hf-512.py": "395439341ea5ba4dd14c103e830383caa75686cfaf6b693dd7efb58622224da6",
    "hf-516.py": "788d4bad50a8d39be2fe79125f0f40134773cd23b1791606fb6b3ab0bc6d2263",
    "hub-516.py": "fa5143bbbc6a928c70f7e05580b358caae16e88f53f49059c7002f0d01f6c832",
    "sp-test.py": "7f96db7e982e35f3457bed536217a008673fe63c5bd799348914990810cf834f",
    "verl-new.py": "cdbe7654d5e26bb5fa31cd602cceb7d1242998ba9fc3997400caab4751955d45",
    "verl-old.py": "a6abf55ac6dd846e2d8691d75eab09ef95885394ac84d532185f2ca78dacb99f",
}


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    root = parser.parse_args().source
    trees = {}
    for name, expected in HASHES.items():
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            parser.error(f"source hash mismatch: {name}")
        trees[name] = ast.parse(data, filename=name)
        compile(trees[name], name, "exec")
    reference = copy.deepcopy(function(trees["hf-516.py"], "torch_chunk_gated_delta_rule"))
    decorators_removed = [ast.unparse(d) for d in reference.decorator_list]
    reference.decorator_list = []  # Intentionally test the reference body, NOT the dispatcher.
    old_reference = copy.deepcopy(function(trees["hf-512.py"], "torch_chunk_gated_delta_rule"))
    old_reference.decorator_list = []
    namespace = {"torch": torch, "F": F, "Optional": Optional, "signature": signature}
    selected = [function(trees["hf-516.py"], "l2norm"), reference]
    selected += [function(trees["verl-new.py"], n) for n in
                 ["_call_accepts_kwarg", "_split_packed_args", "_packed_chunk_gated_delta_rule"]]
    selected += [function(trees["sp-test.py"], "_cases_for_world_size")]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "selected_original_bodies", "exec"), namespace)
    kernel = namespace["torch_chunk_gated_delta_rule"]
    accepts = {key: namespace["_call_accepts_kwarg"](kernel, key) for key in ["cu_seqlens", "cp_context"]}
    assert all(accepts.values())
    assert not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == "kwargs"
                   for n in ast.walk(reference))
    q = torch.ones((1, 4, 1, 1), dtype=torch.float32, device="cpu")
    k = q.clone()
    v = torch.tensor([1., 1., 0., 0.], dtype=torch.float32, device="cpu").reshape(1, 4, 1, 1)
    g = torch.zeros((1, 4, 1), dtype=torch.float32, device="cpu")
    beta = torch.full((1, 4, 1), .5, dtype=torch.float32, device="cpu")
    cu = torch.tensor([0, 2, 4], dtype=torch.int32, device="cpu")
    packed, _ = namespace["_packed_chunk_gated_delta_rule"](kernel, q, k, v, g, beta, cu, cu)
    independent = []
    for start, end in [(0, 2), (2, 4)]:
        out, _ = kernel(q[:, start:end], k[:, start:end], v[:, start:end], g=g[:, start:end],
                        beta=beta[:, start:end], initial_state=None, output_final_state=False,
                        use_qk_l2norm_in_kernel=True)
        independent.append(out)
    independent = torch.cat(independent, dim=1)
    assert torch.equal(packed[:, :2], independent[:, :2])
    assert independent[:, 2:].abs().max().item() == 0
    assert packed[:, 2:].abs().max().item() > .3
    print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__,
                      "scope": "undecorated Torch reference + original packing helper, CPU forward only",
                      "source_sha256": HASHES, "decorators_removed": decorators_removed,
                      "reference_AST_same_as_512": ast.dump(reference) == ast.dump(old_reference),
                      "signature_accepts": accepts, "reference_reads_kwargs": False,
                      "packed_helper_output": packed.flatten().tolist(),
                      "independent_sequences_output": independent.flatten().tolist(),
                      "world_size_1_cases": namespace["_cases_for_world_size"](1)}, indent=2))


if __name__ == "__main__":
    main()
