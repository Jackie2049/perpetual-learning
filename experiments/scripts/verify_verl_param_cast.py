"""Run #8189's six original CPU test bodies and scoped initialization controls.

Usage: Python3.12+ verify_verl_param_cast.py /path/to/verl
Needs torch (verified 2.9.1). Full original torch_dtypes/test modules load from Git;
only verl/utils package initialization is skipped. No Ray/HF/FSDP/device discovery.
The paired base/head check executes the original cast call, not _build_module.
Direct tests, not native pytest collection, official RoPE/model or training proof.
"""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import types

import torch

BASE = "1d367a4f459a43768f76312c0d3430a99ec46b8c"
HEAD = "4e830c72bd0eb2b0d6ecfbd4c2c4144040637324"
DTYPES = "verl/utils/torch_dtypes.py"
ENGINE = "verl/workers/engine/fsdp/transformer_impl.py"
TEST = "tests/utils/test_cast_module_params_on_cpu.py"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("repo", type=Path)
args = parser.parse_args()
sys.dont_write_bytecode = True
torch.set_default_device("cpu")
torch.manual_seed(0)


def source(rev, path):
    raw = subprocess.check_output(["git", "show", rev + ":" + path], cwd=args.repo,
                                  env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    compile(raw, rev + ":" + path, "exec")
    print(rev[:8], path, hashlib.sha256(raw).hexdigest())
    return raw


for name in ("verl", "verl.utils"):
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
module = types.ModuleType("verl.utils.torch_dtypes")
sys.modules[module.__name__] = module
exec(compile(source(HEAD, DTYPES), HEAD + ":" + DTYPES, "exec"), module.__dict__)
source(BASE, DTYPES)
tests = {"__name__": "original_param_cast_tests"}
exec(compile(source(HEAD, TEST), HEAD + ":" + TEST, "exec"), tests)
test_names = sorted(name for name in tests if name.startswith("test_"))
assert len(test_names) == 6
for name in test_names:
    tests[name]()
print("Original tests passed:", test_names)

fixture = tests["_RotaryLikeModel"]
models = {}
for rev in (BASE, HEAD):
    tree = ast.parse(source(rev, ENGINE))
    build = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "_build_module")
    calls = [node for node in ast.walk(build) if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call)
             and ast.unparse(node.value) in ("module.to(torch_dtype)",
                                             "cast_module_params_to_dtype(module, torch_dtype)")]
    assert len(calls) == 1
    model = fixture()
    before = model.inv_freq.clone()
    env = dict(module=model, torch_dtype=torch.bfloat16,
               cast_module_params_to_dtype=module.cast_module_params_to_dtype)
    exec(compile(ast.Module(calls, type_ignores=[]), rev + ":cast_call", "exec"), env)
    assert model.weight.dtype == torch.bfloat16
    assert torch.equal(model.inv_freq.float(), before) == (rev == HEAD)
    models[rev] = model

rounded = models[BASE].inv_freq.float().clone()
fresh = models[HEAD].inv_freq.clone()
error = (fresh - rounded).abs().max().item()
models[BASE].to(torch.float32)
assert torch.equal(models[BASE].inv_freq, rounded) and not torch.equal(rounded, fresh)
module.cast_module_params_to_dtype(models[BASE], torch.bfloat16)
assert torch.equal(models[BASE].inv_freq, rounded)  # Prevention, not reconstruction.
print("Paired original cast calls; irrecoverable frequency max error:", error)

model = fixture()
model.register_buffer("low_precision", torch.tensor([0.1], dtype=torch.bfloat16))
module.cast_module_params_to_dtype(model, torch.bfloat16)
assert model.low_precision.dtype == torch.bfloat16
print("Construction dtype retained, not all buffers forced to FP32")

model = fixture()
model._keep_in_fp32_modules = ["weight"]
model.weight.requires_grad_(False)
module.cast_module_params_to_dtype(model, torch.bfloat16)
assert model.weight.dtype == torch.bfloat16 and not model.weight.requires_grad
print("Parameter selection ignores trainability and HF keep-list attributes")

model = fixture()
model.weight.grad = torch.ones_like(model.weight)
gradient = model.weight.grad
module.cast_module_params_to_dtype(model, torch.bfloat16)
assert model.weight.grad is gradient and gradient.dtype == torch.float32
print("Existing gradient not converted: initialization helper, not training-state migration")
print("PASS: 6 original tests + 4 scoped CPU control groups; torch", torch.__version__)
