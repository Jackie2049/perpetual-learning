"""Check fixed #8352 staging AST with real CPU tensors, not IPC/vLLM.

Usage: python verify_verl_lora_host_staging.py /path/to/verl-git
Requires Python 3.10+ and torch; the fixed Git objects must already be present.
No package import, GPU discovery, pinned allocation, network, or timing claims.
The original AST statements are extracted, not the complete callback/method.
"""
import ast
import gc
import hashlib
import json
import os
import subprocess
import sys

import torch

BASE = 'a4b9354968705a441fb7da7367924b15b902e771'
HEAD = '7b9b25a551eb8cded7d0dda0c6b8437bb720b796'
PATH = 'verl/workers/rollout/vllm_rollout/utils.py'
repo = sys.argv[1]
env = dict(os.environ, GIT_NO_LAZY_FETCH='1')
hashes = {}


def get(rev, path):
    return subprocess.check_output(['git', '-C', repo, 'show', f'{rev}:{path}'], env=env)


def extract(rev):
    raw = get(rev, PATH)
    hashes[rev] = hashlib.sha256(raw).hexdigest()
    compile(raw, f'{rev}:{PATH}', 'exec')
    tree = ast.parse(raw)
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name == 'update_weights_from_ipc']
    assert len(methods) == 1
    callback = next(n for n in methods[0].body if isinstance(n, ast.FunctionDef)
                    and n.name == 'on_bucket_received')
    outer = callback.body[0]
    assert isinstance(outer, ast.If) and ast.unparse(outer.test) == 'lora_weights is not None'
    stage = outer.body[0]
    if rev == HEAD:
        assert isinstance(stage, ast.If)
        assert ast.unparse(stage.test) == 'isinstance(lora_dtype, torch.dtype) and weights'
    else:
        assert isinstance(stage, ast.Expr)
        assert 'tensor.clone()' in ast.unparse(stage)
    code = compile(ast.fix_missing_locations(ast.Module(body=[stage], type_ignores=[])),
                   f'{rev}:original_staging_AST', 'exec')
    update = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_update_weights')
    assignment = update.body[0].body[0]
    assert isinstance(assignment, ast.Assign)
    finalize = compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
                       f'{rev}:original_dict_assignment', 'exec')
    return code, finalize


codes = {rev: extract(rev) for rev in (BASE, HEAD)}
receiver = get(HEAD, 'verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py')
compile(receiver, 'head:bucketed_weight_transfer.py', 'exec')
hashes['receiver'] = hashlib.sha256(receiver).hexdigest()


def stage(rev, inputs, output=None, dtype=torch.bfloat16):
    output = {} if output is None else output
    scope = dict(torch=torch, weights=inputs, lora_weights=output,
                 lora_dtype=dtype, pin_memory=False)
    exec(codes[rev][0], scope)
    return output


def storage_bytes(weights):
    storages = {t.untyped_storage()._cdata: t.untyped_storage().nbytes()
                for t in weights.values() if t.numel()}
    return len(storages), sum(storages.values())


source = torch.arange(10, dtype=torch.float32)
inputs = [('a', source[:6].view(2, 3)), ('b', source[6:].view(2, 2)),
          ('empty', source[:0].view(0, 2))]
expected = {k: t.clone() for k, t in inputs}
baseline, staged = stage(BASE, inputs), stage(HEAD, inputs)
assert storage_bytes(baseline) == (2, 40)
assert storage_bytes(staged) == (1, 20)
assert all(staged[k].shape == t.shape for k, t in inputs)
assert all(torch.equal(staged[k], t.to(torch.bfloat16)) for k, t in inputs)
assert staged['a'].untyped_storage()._cdata == staged['b'].untyped_storage()._cdata
assert staged['b'].storage_offset() == 6
source.fill_(999)
assert all(torch.equal(baseline[k], t) for k, t in expected.items())
assert all(torch.equal(staged[k], t.to(torch.bfloat16)) for k, t in expected.items())
for t in staged.values():
    assert t.to('cpu', dtype=torch.bfloat16) is t

# A second bucket adds a second backing block; existing views retain the first.
stage(HEAD, [('c', torch.tensor([1.1, 2.2, 3.3]))], staged)
assert storage_bytes(staged) == (2, 26)
assert torch.equal(staged['c'], torch.tensor([1.1, 2.2, 3.3]).to(torch.bfloat16))

# The config fallback is the original per-tensor clone, with no dtype conversion.
fallback = stage(HEAD, list(expected.items()), dtype='auto')
assert storage_bytes(fallback) == (2, 40)
assert all(fallback[k].dtype == torch.float32 for k in fallback)
before = list(staged.items())
stage(HEAD, [], staged)
assert all(staged[k] is t for k, t in before)

# Synthetic noncontiguous and scalar inputs: layout, not receiver coverage.
matrix = torch.arange(12, dtype=torch.float32).view(3, 4).t()
special = stage(HEAD, [('strided', matrix), ('scalar', torch.tensor(1.2345))])
assert special['strided'].is_contiguous() and torch.equal(special['strided'], matrix.to(torch.bfloat16))
assert special['scalar'].shape == torch.Size([])
assert storage_bytes(special) == (1, 26)

# Execute only the original assignment, not TensorLoRARequest/add_lora.
scope = {'weights': list(staged.items())}
exec(codes[HEAD][1], scope)
owned = scope['weights']
assert all(owned[k] is t for k, t in staged.items())
staged.clear()
gc.collect()
assert storage_bytes(owned) == (2, 26)
assert owned['a'][0, 0].item() == 0

print(json.dumps(dict(torch=torch.__version__, python=sys.version.split()[0],
                      compiled_versions=3, baseline_blocks_payload=[2, 40],
                      head_first_bucket_blocks_payload=[1, 20],
                      head_two_buckets_blocks_payload=[2, 26],
                      fallback_blocks_payload=[2, 40], pin_memory=False,
                      controls='shape/cast, source overwrite, CPU to identity, multiple buckets, '
                      'dtype fallback, empty bucket, synthetic strided/scalar, dict lifetime',
                      sha256=hashes), indent=2))
