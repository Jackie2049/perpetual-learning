"""Check #597's native CPU records and local sink, not trainer/transport/model behavior.

Usage: python -B verify_unirl_sample_records.py /path/to/UniRL /path/to/Pillow-target
Python3.12+/torch/Pillow, fixed local Git objects. Loads complete original modules
from Git; only tensor package initialization skips Ray transport and re-exports
the original Batch/ref helpers. No method extraction or fake Sample/media classes.
"""
import copy
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

sys.dont_write_bytecode = True
repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(Path(sys.argv[2]).resolve()))
HEAD = 'd507c4dc1e10db427cadff4d6e29482801a12065'
BASE = '6c828f5ed81649153315d996f0f597e8808ff144'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=repo, env=dict(os.environ, GIT_NO_LAZY_FETCH='1'))


paths = set(git('ls-tree', '-r', '--name-only', HEAD).decode().splitlines())


class OriginalModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'unirl' and not fullname.startswith('unirl.'):
            return None
        stem = fullname.replace('.', '/')
        module = stem + '/__init__.py' if stem + '/__init__.py' in paths else stem + '.py'
        if module not in paths:
            return None
        spec = importlib.util.spec_from_loader(fullname, self, is_package=module.endswith('/__init__.py'))
        spec.loader_state = module
        return spec

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = module.__spec__.loader_state
        if module.__name__ == 'unirl.distributed.tensor':
            from unirl.distributed.tensor import batch, ref
            for source in (batch, ref):
                for name in dir(source):
                    if not name.startswith('_'):
                        setattr(module, name, getattr(source, name))
            return
        raw = git('show', HEAD + ':' + path)
        exec(compile(raw, HEAD + ':' + path, 'exec'), module.__dict__)


sys.meta_path.insert(0, OriginalModules())
for path in git('diff', '--name-only', BASE + '...' + HEAD).decode().splitlines():
    if path.endswith('.py'):
        compile(git('show', HEAD + ':' + path), path, 'exec')
for path in ('unirl/types/sample_record.py', 'unirl/types/media_preview.py', 'unirl/utils/sample_dump.py'):
    print(path, hashlib.sha256(git('show', HEAD + ':' + path)).hexdigest())

import torch
import PIL
from unirl.types.sample import Sample, Part
from unirl.types.primitives import Texts, Audios, Audio
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
from unirl.types.sample_record import build_sample_records
from unirl.utils.sample_dump import dump_sample_records

root = Part.input(['A', 'B'], primitives={'text': Texts(['prompt A', 'prompt B'])})
recap = Part(sample_ids=['B/1', 'A/0', 'B/0', 'A/1'],
             sampling_params=ARSamplingParams(),
             primitives={'text': Texts(['recap B1', 'recap A0', 'recap B0', 'recap A1'])},
             rewards=torch.tensor([11., 0., 10., 1.]))
final = Part(sample_ids=['A/1/0', 'B/0/0', 'A/0/0', 'B/1/0'],
             sampling_params=DiffusionSamplingParams(),
             primitives={'text': Texts(['out A1', 'out B0', 'out A0', 'out B1'])},
             rewards=torch.tensor([1., 10., 0., 11.]),
             component_rewards={'test': torch.tensor([.1, .2, .3, .4])})
sample = Sample(parts=[root, recap, final])
records = build_sample_records(sample, part_index=2, step=1, phase='rollout', max_items=2)
assert [(r.root_id, r.prompt, r.context_texts, r.output_text, r.reward) for r in records] == [
    ('A', 'prompt A', ['recap A1'], 'out A1', 1.),
    ('B', 'prompt B', ['recap B0'], 'out B0', 10.)]
assert [r.group_id for r in records] == ['A/1', 'B/0']
other = build_sample_records(sample, part_index=1, step=1, phase='rollout', max_items=2)
assert len(other + records) == 4 and {r.track for r in other + records} == {'ar', 'diffusion'}
print('PASS reordered lineage + per-Part cap: two generated Parts, cap2 -> 4 records')
for cap, count in ((0, 0), (-1, 0), (1, 1), (99, 4)):
    assert len(build_sample_records(sample, part_index=2, step=1, phase='eval', max_items=cap)) == count
print('PASS zero/negative/one/oversized caps')

try:
    Sample(parts=[root, Part(sample_ids=['missing/0'], sampling_params=ARSamplingParams(),
                            primitives={'text': Texts(['still observed'])}, rewards=torch.tensor([3.]))])
except ValueError as exc:
    assert 'parent id is not in the preceding part' in str(exc)
else:
    raise AssertionError('native Sample must reject malformed parent lineage')
print('PASS native Sample rejects malformed lineage before observation')

audio = Audios.from_list([Audio(torch.linspace(-1, 1, n)) for n in (3, 4, 5, 6)])
final.primitives = {'audio': audio}
final.primitive_metadata = {'audio': {'sample_rate': 8000}}
before = audio.waveform.clone()
audio_records = build_sample_records(sample, part_index=2, step=7, phase='eval/test', max_items=2)
assert [len(r.media.audio) for r in audio_records] == [3, 4]
assert all(r.media.audio_sample_rate == 8000 for r in audio_records)
assert final.primitives['audio'] is audio and torch.equal(audio.waveform, before)
assert all(r.output_text is None for r in audio_records)
print('PASS resident audio prefix, sample rate, training primitive identity/value retained')

with tempfile.TemporaryDirectory(prefix='unirl-records-') as directory:
    dump_sample_records(directory, audio_records)
    index = Path(directory) / 'eval/test/step_000007/samples.jsonl'
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    assert [row['root_id'] for row in rows] == ['A', 'B']
    for row, record in zip(rows, audio_records):
        path = index.parent / row['media']['audio']
        with wave.open(str(path), 'rb') as output:
            assert (output.getframerate(), output.getnchannels(), output.getnframes()) == (8000, 1, len(record.media.audio))
            expected = record.media.audio.clamp(-1, 1).mul(32767).to(torch.int16).numpy().tobytes()
            assert output.readframes(output.getnframes()) == expected
    dump_sample_records(directory, audio_records[:1])
    assert len(index.read_text().splitlines()) == 1
    assert (index.parent / rows[1]['media']['audio']).exists()
    print('PASS decoded local JSONL/WAV + same-step index replacement leaves prior extra file')

    blocked = Path(directory) / 'ordinary-file'
    blocked.touch()
    dump_sample_records(str(blocked), audio_records)
    partial = copy.copy(audio_records[0])
    partial.metadata = {'not_json': object()}
    dump_sample_records(str(Path(directory) / 'partial'), [partial])
    partial_dir = Path(directory) / 'partial/eval/test/step_000007'
    assert (partial_dir / 'diffusion_0.wav').exists() and not (partial_dir / 'samples.jsonl').exists()
    print('PASS local path/JSON failures return; a written media file can remain without index')

assert not torch.distributed.is_initialized()
assert not any(name in sys.modules for name in ('ray', 'av', 'wandb', 'unirl.trainer.base'))
print('Python', sys.version.split()[0], 'torch', torch.__version__, 'Pillow', PIL.__version__)
print('6 native CPU control groups; 11 changed Python modules compile. No upstream tests claimed.')
print('NOT RUN: TensorRef transport/hydration volume, BaseTrainer, engine, W&B, video or model/device training.')
