#!/usr/bin/env python3
"""CPU tensor ownership controls for verl #8338, NOT a Megatron patch test.

Python >=3.10 and torch; pass a local verl Git checkout with the pinned objects.
Executes only the original grad-tuple assignment and detached-input cleanup loop
selected by AST. No patched checkpoint, RNG/distribution/streams or CUDA tests.
Destructive resize controls affect only fresh task tensors; after resize they
inspect metadata only, never dereference the invalidated tensor/aliases.
No device discovery, allocator/RSS measurements, framework imports or stubs.
"""

import argparse
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import torch

HEAD = "c099b0179cfa70987b7b75e5b1c9b685d39c4d5c"
ANCESTOR = "d1390eea0c94467d4ae1387eeda9e2f78f8d73ea"
PATCH = "verl/models/mcore/patch.py"
TEST = "tests/models/test_mcore_recompute_reclaim.py"


def read(root, revision, path):
    data = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=root,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    print(revision[:8], path, "sha256=" + hashlib.sha256(data).hexdigest())
    parsed = ast.parse(data, filename=path)
    compile(parsed, path, "exec")
    return parsed


def cleanup_code(root):
    read(root, ANCESTOR, PATCH)
    parsed = read(root, HEAD, PATCH)
    read(root, HEAD, TEST)  # Compile only; never import its CUDA availability guard.
    outer = next(n for n in parsed.body if isinstance(n, ast.FunctionDef)
                 and n.name == "apply_patch_megatron_recomputation_backward")
    backward = next(n for n in outer.body if isinstance(n, ast.FunctionDef) and n.name == "patch_backward")
    grads = [n for n in backward.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "grads" for t in n.targets)]
    release = [n for n in backward.body if isinstance(n, ast.For)
               and isinstance(n.iter, ast.Name) and n.iter.id == "detached_inputs"]
    assert len(grads) == len(release) == 1
    selected = ast.Module(body=[grads[0], release[0]], type_ignores=[])
    print("selected two-statement AST sha256=" + hashlib.sha256(ast.dump(selected).encode()).hexdigest())
    return compile(selected, f"{HEAD}:{PATCH}:cleanup-only", "exec")


def release(code, inputs):
    namespace = {"torch": torch, "detached_inputs": inputs}
    exec(code, namespace)
    return namespace["grads"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verl", type=Path)
    code = cleanup_code(parser.parse_args().verl)
    print("Python", sys.version.split()[0], "torch", torch.__version__, "git", torch.version.git_version)

    # A detach/view is not an independent allocation. Do NOT read values after resize.
    base = torch.arange(6, dtype=torch.float64, device="cpu")
    view, detached = base[1:5], base.detach()
    assert base.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()
    assert base.untyped_storage().data_ptr() == detached.untyped_storage().data_ptr()
    detached.untyped_storage().resize_(0)
    assert [t.untyped_storage().nbytes() for t in (base, view, detached)] == [0, 0, 0]
    assert [t.numel() for t in (base, view, detached)] == [6, 4, 6]
    print("shared storage resize: all 0 bytes, logical numel still 6/4/6; no reads")

    # Returning an alias cannot survive truncation, but an independent clone can.
    gradient = torch.full((4,), 2., dtype=torch.float64, device="cpu")
    returned_alias, copied = gradient, gradient.clone()
    gradient.untyped_storage().resize_(0)
    assert returned_alias.untyped_storage().nbytes() == 0  # Metadata only.
    assert torch.equal(copied, torch.full((4,), 2., dtype=torch.float64, device="cpu"))
    print("gradient resize: returned alias invalidated; separate clone intact")

    # Delayed real autograd needs the base storage after detached-leaf cleanup.
    p = torch.nn.Parameter(torch.arange(1, 7, dtype=torch.float64, device="cpu"))
    w = torch.nn.Parameter(torch.full((6,), 2., dtype=torch.float64, device="cpu"))
    source = p * 3
    delayed_loss = (source * w).sum()  # Its backward saves source for w.grad.
    source_view = source[1:5]
    leaf = source_view.detach().requires_grad_()
    another_alias = leaf.detach()
    (leaf.square().sum()).backward()
    grad_before = leaf.grad
    original_pointer = source.untyped_storage().data_ptr()
    (returned,) = release(code, (leaf,))
    assert returned is grad_before and leaf.grad is None and leaf.numel() == 0
    assert source.untyped_storage().data_ptr() == original_pointer
    assert source.untyped_storage().nbytes() == 48 and source_view.numel() == 4
    assert torch.equal(another_alias, source_view)
    assert torch.equal(returned, torch.tensor([12., 18., 24., 30.], dtype=torch.float64, device="cpu"))
    delayed_loss.backward()
    assert torch.equal(p.grad, torch.full((6,), 6., dtype=torch.float64, device="cpu"))
    assert torch.equal(w.grad, torch.arange(1, 7, dtype=torch.float64, device="cpu") * 3)
    print("original cleanup: leaf empty/grad None; input aliases, returned grad and delayed backward intact")

    frozen = torch.arange(3, dtype=torch.float64, device="cpu")
    pointer = frozen.untyped_storage().data_ptr()
    assert release(code, (frozen,)) == (None,)
    assert frozen.untyped_storage().data_ptr() == pointer and frozen.numel() == 3
    assert torch.equal(frozen, torch.arange(3, dtype=torch.float64, device="cpu"))
    print("non-requires-grad input is untouched")
    print("PASS: 4 CPU ownership controls; 3 full-file syntax checks; original 5 CUDA tests NOT run")


if __name__ == "__main__":
    main()
