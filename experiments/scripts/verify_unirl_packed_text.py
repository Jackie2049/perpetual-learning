#!/usr/bin/env python3
"""Check original UniRL packed TextSegment row operations with CPU payloads.

Python3.12/torch; run with /path/to/UniRL at 6fd82cba. One package initializer
is bypassed to avoid unrelated Ray transport imports; no fake classes/methods.
"""
import hashlib
from itertools import permutations
from pathlib import Path
import sys
from types import ModuleType

sys.dont_write_bytecode = True
root = Path(sys.argv[1]).resolve()
for path, expected in {
    'unirl/distributed/tensor/batch.py': 'a33122410e68c2295631bbc3e70ce4b69ba41081da7ed49abd06774640676e3c',
    'unirl/types/segments/base.py': 'f7b9ce0a6ae77bcdbde81a93b43096bdfe975368234753ef74ab6191d794ff32',
    'unirl/types/segments/text.py': '9f58114b40f3949eea70fba0692fb96b746e3e47af0c94c7db99e314cf237b34',
}.items():
    actual = hashlib.sha256((root/path).read_bytes()).hexdigest()
    assert actual == expected, (path, actual)
    print(path, actual)
sys.path.insert(0, str(root))
import unirl.distributed
package = ModuleType('unirl.distributed.tensor')
package.__path__ = [str(root/'unirl/distributed/tensor')]
sys.modules[package.__name__] = package
unirl.distributed.tensor = package
from unirl.types.segments.text import TextSegment
import torch


def check(segment, rows, selected):
    lengths = [len(rows['tokens'][i]) for i in selected]
    cumulative = [0]
    for size in lengths:
        cumulative.append(cumulative[-1] + size)
    assert segment.batch_size == len(selected)
    assert segment.cu_seqlens.device.type == 'cpu'
    assert segment.cu_seqlens.dtype == torch.int64
    assert segment.cu_seqlens.tolist() == cumulative
    assert segment.lengths.tolist() == lengths
    assert segment.status.tolist() == [10 * (i+1) for i in selected]
    for field in ('tokens', 'log_probs', 'loss_mask'):
        expected = [value for i in selected for value in rows[field][i]]
        actual = getattr(segment, field)
        assert actual.device.type == 'cpu' and actual.ndim == 1
        assert actual.dtype == (torch.int64 if field == 'tokens' else torch.float64)
        assert actual.tolist() == expected, (field, selected)


for lengths in ((2,0,3), (0,0,0), (1,2,1), (0,1,0)):
    tokens = [[100*(i+1)+j for j in range(n)] for i, n in enumerate(lengths)]
    rows = dict(tokens=tokens,
                log_probs=[[-float(value) for value in row] for row in tokens],
                loss_mask=[[float(j % 2) for j in range(len(row))] for row in tokens])
    fields = {name: [torch.tensor(row, dtype=torch.int64 if name == 'tokens' else torch.float64,
                                 device='cpu') for row in values] for name, values in rows.items()}
    segment = TextSegment.pack(status=torch.tensor([10,20,30], device='cpu'), **fields)
    check(segment, rows, [0,1,2])
    for selected in (*permutations(range(3)), (2,0,2), ()):
        check(segment.select(selected), rows, selected)
    check(segment.select(torch.tensor([2,0,1], device='cpu')), rows, [2,0,1])
    for start in range(4):
        for end in range(start,4):
            check(segment.slice(start,end), rows, list(range(start,end)))
    for cut in range(4):
        check(TextSegment.concat([segment.slice(0,cut), segment.slice(cut,3)]), rows, [0,1,2])
    check(TextSegment.concat(segment.chunk(3)), rows, [0,1,2])
    if lengths == (2,0,3):
        selected = segment.select([2,0,2])
        print('lengths', lengths, 'cu', segment.cu_seqlens.tolist(),
              'select2/0/2 tokens', selected.tokens.tolist(), 'cu', selected.cu_seqlens.tolist())
    print('layout', lengths, ': pack, 9 selections, 10 slices, 4 concat cuts and chunk roundtrip passed')


plain = TextSegment.pack(tokens=[torch.empty(0, dtype=torch.int64, device='cpu') for _ in range(3)])
assert plain.status is None and plain.batch_size == 3 and plain.tokens.numel() == 0
assert plain.cu_seqlens.tolist() == [0,0,0,0] and plain.lengths.tolist() == [0,0,0]
print('No status field: three zero-token rows still have batch_size=3 via cumulative offsets')


def rejected(kind, call, message):
    try:
        call()
    except kind as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError('missing expected ' + kind.__name__)


rejected(TypeError, lambda: TextSegment.pack(tokens=torch.tensor([1,2], device='cpu')), 'already-packed Tensor')
rejected(ValueError, lambda: TextSegment.pack(
    tokens=[torch.tensor([1,2], device='cpu')],
    loss_mask=[torch.tensor([1.0], device='cpu')]), "don't match earlier")
rejected(ValueError, lambda: segment.chunk(2), 'not divisible')
rejected(ValueError, lambda: TextSegment(tokens=torch.tensor([1,2], device='cpu')).slice(0,1),
         'requires _packed_cu_seqlens')
assert not any(n in sys.modules for n in ('ray', 'hydra', 'omegaconf', 'unirl.distributed.tensor.worker_local'))
print('Four layouts / 36 selects / 40 slices / 16 concat cuts / 4 chunk roundtrips / 4 guards passed')
print('Python', sys.version.split()[0], 'torch', torch.__version__, 'torch git', torch.version.git_version)
print('Original CPU containers, not upstream tests; NOT RUN: Part/planner, TensorRef transport, model/attention/gradient or devices')
