"""Real torchdata streams with #7402's original dataloader-load method.

Usage: python -B verify_verl_sft_loader_resume.py /path/to/verl --deps /path/to/torchdata-target
Python3.12+, torch (tested 2.9.1), torchdata0.11.0 and fixed local Git objects.
AST-loads the exact base/head method and original log helper; self supplies only
loader/rank fields. Saves real state files in a temporary directory. No engine,
Ray/SPMD orchestration, model/optimizer, fit loop, distributed group or device probe.
This is not the author's original 24-test suite or a complete SFT restore test.
"""
import argparse
import ast
import copy
import hashlib
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('repo', type=Path)
parser.add_argument('--deps', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.deps.resolve()))
import torch
import torchdata
from torch.utils.data import DistributedSampler, TensorDataset
from torchdata.stateful_dataloader import StatefulDataLoader

BASE = '09ac37258ea66b0cb69b2738eec3074ea4e7261c'
HEAD = 'b7d1f36e446da4caeb85a7f9d770daa416a9e76f'
PATH = 'verl/utils/checkpoint/checkpoint_handler.py'


def source(rev, path):
    raw = subprocess.check_output(['git', 'show', rev + ':' + path], cwd=args.repo,
                                  env=dict(os.environ, GIT_NO_LAZY_FETCH='1'))
    compile(raw, path, 'exec')
    print(rev[:8], path, 'sha256=' + hashlib.sha256(raw).hexdigest())
    return ast.parse(raw)


def load_nodes(nodes, ns):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original-definitions>', 'exec'), ns)


def loader(batch_size=2, offset=0):
    dataset = TensorDataset(torch.arange(8) + offset)
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=0, drop_last=True)
    result = StatefulDataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0,
                                drop_last=True, pin_memory=False)
    return sampler, result


def values(data):
    return [batch[0].tolist() for batch in data]


def saved_state(consumed, observed_stop=False):
    _, data = loader()
    iterator = iter(data)
    for _ in range(consumed):
        next(iterator)
    if observed_stop:
        assert list(iterator) == []
    return copy.deepcopy(data.state_dict())


log = logging.getLogger('loader-resume-control')
log.setLevel(logging.WARNING)
ns = dict(torch=torch, os=os, logging=logging, logger=log)
logging_tree = source(HEAD, 'verl/utils/logger/aggregate_logger.py')
load_nodes([n for n in logging_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'log_with_rank'], ns)
methods = {}
for rev in (BASE, HEAD):
    cls = next(n for n in source(rev, PATH).body if isinstance(n, ast.ClassDef) and n.name == 'CheckpointHandler')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_load_dataloader_state')
    load_nodes([method], ns)
    methods[rev] = ns['_load_dataloader_state']

before, after = saved_state(4), saved_state(4, observed_stop=True)
assert before['_iterator_finished'] is False and after['_iterator_finished'] is True
for label, state, expected_count in [('last-batch-before-StopIteration', before, 0),
                                     ('after-StopIteration', after, 4)]:
    sampler, data = loader()
    data.load_state_dict(copy.deepcopy(state))
    sampler.set_epoch(1)
    rows = values(data)
    assert len(rows) == expected_count
    print('PASS native state', label, 'resumed batches', rows)

cases = [
    ('matching boundary', 4, 2, True, False, 0),
    ('matching mid-epoch', 2, 2, True, False, 0),
    ('shrink epoch', 2, 4, True, False, 0),
    ('grow epoch', 4, 1, True, False, 0),
    ('legacy mid-epoch', 2, 2, False, False, 0),
    ('legacy boundary', 4, 2, False, False, 0),
    ('boundary after observed stop', 4, 2, True, True, 0),
    ('same length changed data', 4, 2, True, False, 100),
]
for name, step, batch_size, metadata, observed_stop, offset in cases:
    raw = saved_state(step, observed_stop)
    outputs = {}
    for rev in (BASE, HEAD):
        state = copy.deepcopy(raw)
        if metadata:
            state['_verl_steps_per_epoch'] = 4
        sampler, data = loader(batch_size, offset)
        owner = SimpleNamespace(dp_rank=0, rank=0, train_dataloader=data)
        with tempfile.TemporaryDirectory(prefix='verl-sft-loader-') as directory:
            torch.save(state, Path(directory) / 'data_0.pt')
            if rev == BASE:
                methods[rev](owner, directory)
            else:
                methods[rev](owner, directory, step)
        sampler.set_epoch(step // len(data))
        outputs[rev] = values(data)
        ref_sampler, reference = loader(batch_size, offset)
        skip_state = rev == HEAD and metadata and len(reference) == 4 and step > 0 and step % 4 == 0
        if not skip_state:
            reference.load_state_dict(copy.deepcopy(raw))
        ref_sampler.set_epoch(step // len(reference))
        assert outputs[rev] == values(reference), (name, rev, outputs[rev])
    if name in ('matching boundary', 'same length changed data'):
        assert outputs[BASE] == [] and len(outputs[HEAD]) == 4
    if name == 'legacy boundary':
        assert outputs[HEAD] == []
    print('PASS', name, 'base/head batches', outputs[BASE], outputs[HEAD])

assert not torch.distributed.is_initialized()
print('Python', sys.version.split()[0], 'torch', torch.__version__, 'torchdata', torchdata.__version__)
print('2 native iterator-state controls + 8 original-method cases, real CPU loaders/files only.')
print('No native verl import, original 24-test collection, Ray/SPMD, fit, engine or training restore validation.')
