"""Check original UniRL data cursors with read-only JSONL fixtures; see readme.md."""
import hashlib
import json
from pathlib import Path
import random
import sys
from types import ModuleType

source = Path(sys.argv[1]).resolve()
scratch = Path(__file__).resolve().parent
sys.path.insert(0, str(source))
sys.dont_write_bytecode = True
assert hashlib.sha256((source/'unirl/data/sft.py').read_bytes()).hexdigest() == '45dc13244d15a0168919e48e33f126b6c9f259f02587007f4adcbc2c24ed782c'
assert hashlib.sha256((source/'unirl/trainer/sft.py').read_bytes()).hexdigest() == 'de66b86676f81422a832ce560942fd22e239b2052a00059b685631661ea3d2c9'
import unirl.distributed
package = ModuleType('unirl.distributed.tensor')
package.__path__ = [str(source / 'unirl/distributed/tensor')]
sys.modules[package.__name__] = package
unirl.distributed.tensor = package
from unirl.data.sft import SupervisedDataSource as DS
import torch

train = str(scratch / 'train.jsonl')
def ids(batch):
    return [row['sample_id'] for row in batch]

def fails(kind, call, fragment):
    try:
        call()
    except kind as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError('expected ' + kind.__name__)

data = DS(train, shuffle=False)
assert ids(data.get_samples(3)) == ['r0', 'r1', 'r2']
before = json.loads(json.dumps(data.state_dict()))
peek = data.peek_next_batch(3)
assert ids(peek) == ['r3', 'r4', 'r0'] and data.state_dict() == before
assert data.epoch == 0.6
fails(RuntimeError, lambda: data.get_samples(1), 'commit the pending batch')
fails(RuntimeError, lambda: data.peek_next_batch(1), 'already pending')
resumed = DS(train, shuffle=False)
resumed.load_state_dict(before)
assert ids(resumed.get_samples(3)) == ids(peek)
assert data.commit_peeked_batch() is peek
assert data.state_dict() == resumed.state_dict() and data.epoch == 1.2
fails(RuntimeError, data.commit_peeked_batch, 'no batch is pending')
print('peek across epoch: r3/r4/r0; committed cursor 0/3 stays unchanged until commit -> 1/1')
print('saved pre-commit cursor replays pending rows after loading; guards reject double peek/advance/commit')

rng = random.getstate()
for shuffle in (False, True):
    direct = DS(train, seed=7, shuffle=shuffle)
    prefetched = DS(train, seed=7, shuffle=shuffle)
    for index in range(10):
        saved = prefetched.state_dict()
        pending = prefetched.peek_next_batch(3)
        assert prefetched.state_dict() == saved
        assert ids(prefetched.commit_peeked_batch()) == ids(direct.get_samples(3))
        assert prefetched.state_dict() == direct.state_dict()
assert random.getstate() == rng
print('10 batches x shuffle false/true: prefetch and direct streams/cursors match; global random unchanged')

saved = DS(train, seed=7, shuffle=True)
saved.get_samples(3)
state = saved.state_dict()
fails(ValueError, lambda: DS(train, seed=8).load_state_dict(state), 'different sample order')
fails(ValueError, lambda: DS(train, seed=7, shuffle=False).load_state_dict(state), 'different sample order')
fails(ValueError, lambda: DS(str(scratch/'changed.jsonl'), seed=7).load_state_dict(state), 'fingerprint mismatch')
legacy = dict(state)
del legacy['manifest_fingerprint']
restored = DS(train, seed=7)
restored.load_state_dict(legacy)
assert ids(restored.get_samples(3)) == ids(saved.get_samples(3))
print('changed seed/shuffle/content rejected; legacy missing fingerprint warns but remains accepted')

data = DS(train, eval_manifest_path=str(scratch/'eval.jsonl'), shuffle=False)
data.get_samples(3)
peek = data.peek_next_batch(3)
state = data.state_dict()
assert [ids(batch) for batch in data.iter_eval_batches(2)] == [['e0','e1'],['e2']]
assert data.state_dict() == state and data.commit_peeked_batch() is peek
print('eval 2+1 rows leaves training cursor and pending batch intact')
for name in ('unirl/data/sft.py', 'unirl/trainer/sft.py'):
    print(name, hashlib.sha256((source/name).read_bytes()).hexdigest())
assert not any(name in sys.modules for name in ('ray','hydra','omegaconf','unirl.distributed.tensor.worker_local'))
print('Python', sys.version.split()[0], 'torch', torch.__version__, 'all real DataSource controls passed')
print('One package initializer bypassed; native dataset/normalization/Batch, no fake classes or method extraction')
print('NOT RUN: trainer, prefetch worker, optimizer, disk checkpoints/recovery, crash or GPU/NPU')
