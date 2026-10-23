"""Fixed #7503 original CPU test bodies plus real torch storage controls.

Usage: python -B verify_verl_merger_storage.py /path/to/verl-git
Needs torch (tested 2.9.1), Python3.12+, and the fixed local Git objects.
AST-loads the entire original merger class with an object base in place of its
HF-dependent BaseModelMerger, and the original test definitions without imports.
Only the pure _merge_by_placement method runs; no native verl import, DTensor
construction, FSDP loading, HF serialization, distributed group or device probe.
Storage nbytes below are reachable allocation sizes, NOT RSS or a measured peak.
"""
import __future__
import argparse
import ast
import hashlib
import io
import os
from pathlib import Path
import subprocess

import torch
from torch.distributed.tensor import Partial, Replicate, Shard

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('repo', type=Path)
args = parser.parse_args()
BASE = '8df88be746801b6d87c42e15f9a5a0ec1d5eeeae'
HEAD = 'e2d9f145d9176f8a0763bb3cd43ed430a4ca6d8b'
TARGET = '3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e'
PATH = 'verl/model_merger/fsdp_model_merger.py'
TEST = 'tests/model_merger/test_fsdp_merger_storage_aliasing_on_cpu.py'


def read(rev, path):
    raw = subprocess.check_output(['git', 'show', f'{rev}:{path}'], cwd=args.repo,
                                  env=dict(os.environ, GIT_NO_LAZY_FETCH='1'))
    compile(raw, path, 'exec')
    print(rev[:8], path, 'sha256=' + hashlib.sha256(raw).hexdigest())
    return ast.parse(raw)


def execute(nodes, ns):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original-definitions>', 'exec',
                 flags=__future__.annotations.compiler_flag), ns)


def storage_bytes(tensors):
    allocations = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
    return sum(allocations.values())


test_nodes = [n for n in read(HEAD, TEST).body if isinstance(n, ast.FunctionDef)]
target_class = next(n for n in read(TARGET, PATH).body if isinstance(n, ast.ClassDef) and n.name == 'FSDPModelMerger')
target_method = next(n for n in target_class.body if isinstance(n, ast.FunctionDef) and n.name == '_merge_by_placement')
for rev in (BASE, HEAD):
    ns = dict(torch=torch, Replicate=Replicate, Shard=Shard, BaseModelMerger=object)
    original_class = next(n for n in read(rev, PATH).body if isinstance(n, ast.ClassDef) and n.name == 'FSDPModelMerger')
    if rev == BASE:
        baseline_method = next(n for n in original_class.body if isinstance(n, ast.FunctionDef) and n.name == '_merge_by_placement')
        assert ast.dump(baseline_method) == ast.dump(target_method)
        print('PASS target-base method AST matches branch ancestor; full modules may differ')
    execute([original_class], ns)
    execute(test_nodes, ns)
    tests = [n.name for n in test_nodes if n.name.startswith('test_')]
    assert len(tests) == 3
    for name in tests:
        try:
            ns[name]()
        except AssertionError:
            assert rev == BASE and name == 'test_replicate_placement_clones_to_independent_storage'
            print('EXPECTED BASE FAIL', name)
        else:
            assert rev == HEAD or name != 'test_replicate_placement_clones_to_independent_storage'
            print('PASS original test body', name)
    merge = ns['_merger']()._merge_by_placement

    # Real torch save/load restores shared storage for disjoint BF16 views.
    source = torch.arange(1024, dtype=torch.bfloat16)
    stream = io.BytesIO()
    torch.save({'a': source[:4], 'b': source[4:8]}, stream)
    stream.seek(0)
    loaded = torch.load(stream, map_location='cpu', weights_only=True)
    a, b = loaded['a'], loaded['b']
    assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    assert a.data_ptr() != b.data_ptr() and a.storage_offset() == 0 and b.storage_offset() == 4
    merged_a, merged_b = merge([a], Replicate()), merge([b], Replicate())
    merged_a.add_(100)
    assert torch.equal(merged_b, torch.arange(4, 8, dtype=torch.bfloat16))
    assert torch.equal(a, torch.arange(4, dtype=torch.bfloat16) + (100 if rev == BASE else 0))
    print('PASS serialization/disjoint-view and mutation-isolation controls')

    # Conversion used by the real loader may itself break aliasing, depending on dtype.
    assert a.bfloat16() is a
    fp32_source = torch.arange(8, dtype=torch.float32)
    converted_a, converted_b = fp32_source[:4].bfloat16(), fp32_source[4:].bfloat16()
    assert converted_a.untyped_storage().data_ptr() != converted_b.untyped_storage().data_ptr()
    print('PASS dtype-dependent conversion control: BF16 identity, FP32 separate allocations')

    output_only = storage_bytes([merged_a, merged_b])
    with_source_owner = storage_bytes([a, b, merged_a, merged_b])
    assert (output_only, with_source_owner) == ((2048, 2048) if rev == BASE else (16, 2064))
    print('reachable storage bytes output-only/source-retained:', output_only, with_source_owner)

    # Cloning neither validates replica agreement nor enables partial reductions.
    assert torch.equal(merge([torch.tensor([1]), torch.tensor([99])], Replicate()), torch.tensor([1]))
    try:
        merge([torch.tensor([1])], Partial())
    except NotImplementedError:
        pass
    else:
        raise AssertionError('Partial unexpectedly supported')
    print('PASS first-replica/Partial unchanged controls')

assert not torch.distributed.is_initialized()
print('torch', torch.__version__, '; CPU tensors only; 3 original test bodies + 4 control groups per revision.')
print('No native import, full merger/HF test, issue-checkpoint reproduction or peak-memory measurement.')
