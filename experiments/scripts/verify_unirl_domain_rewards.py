#!/usr/bin/env python3
"""Check original UniRL per-domain routing with built-in model-free MC scorers.

Python3.12/torch/Pillow; args: /path/to/UniRL [isolated-Pillow-deps].
Two package initializers are bypassed; native classes, no service/model weights.
"""
import hashlib
import math
from pathlib import Path
import sys
from types import ModuleType

sys.dont_write_bytecode = True
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
if len(sys.argv) > 2:
    sys.path.insert(0, str(Path(sys.argv[2]).resolve()))
for name, expected in {
    'unirl/reward/local/per_domain.py': '6aca28149c4dcc48b3f8e29e74ef7195daa997aa833daac778618d4f3b2b73f8',
    'unirl/reward/local/mc_exact_match.py': '452413e2c7db3cf722fd3a887f660a61e5a13513f015a539f103b0a32d1cf526',
    'unirl/reward/local/base.py': '88b84e60f9ea04214c3c7b43aae9b29a530b9bc553d319b1b51d5fa5ce502b3a',
    'unirl/reward/base.py': 'b5c51aa0728db14bfab1ff49772a73910bdab3715f80dcb73765fa5c794a82e3',
    'unirl/types/reward.py': '00ab4ad2b7072514f363e4c8cd4f97572a1fc601b152591052fee77ed1ea849b',
    'unirl/reward/service.py': '601f64b32f967b54d2ad718f6755a208f4a5d2df26688d93544b5ed070555c61',
    'unirl/utils/wandb_metrics.py': 'e92b9c40ebab4b78759ad15e36db4a833f853f6b122db4e395577966cc150648',
}.items():
    actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
    assert actual == expected, (name, actual)
    print(name, actual)
import unirl.distributed
import unirl.reward
for name, folder in (('unirl.distributed.tensor', 'unirl/distributed/tensor'),
                     ('unirl.reward.local', 'unirl/reward/local')):
    package = ModuleType(name)
    package.__path__ = [str(root/folder)]
    sys.modules[name] = package
    parent, child = name.rsplit('.', 1)
    setattr(sys.modules[parent], child, package)
from unirl.reward.local.mc_exact_match import MCExactMatchRewardScorer, MCExactMatchSpec
from unirl.reward.local.per_domain import PerDomainRewardScorer, PerDomainSpec
from unirl.types.primitives import Texts
from unirl.types.reward import RewardRequest
from unirl.utils.wandb_metrics import _tensor_stats
import PIL
import torch

graded = MCExactMatchRewardScorer(config=MCExactMatchSpec(graded_format_reward=True), base_device='cpu')
strict = MCExactMatchRewardScorer(config=MCExactMatchSpec(require_answer_tag=True), base_device='cpu')
assert graded.model is None and strict.model is None
router = PerDomainRewardScorer(config=PerDomainSpec(scorers={'graded': graded, 'strict': strict}), base_device='cpu')
rows = [('A', {'domain':'graded', 'answer':'A'}),
        ('A', {'domain':'strict', 'answer':'A'}),
        ('<answer>B</answer>', {'domain':'strict', 'answer':'B'}),
        ('<answer>D</answer>', {'domain':'graded', 'answer':'D'}),
        ('A', {'domain':'unknown', 'answer':'A'}),
        ('A', {'answer':'A'})]
expected = [0.5, 0.0, 1.0, 1.0, 0.0, 0.0]


def score(order):
    response = router.compute_rewards(RewardRequest(
        generated={'text': Texts(texts=[rows[i][0] for i in order])},
        metadata=[dict(rows[i][1]) for i in order], sample_ids=[f's{i}' for i in order],
        group_ids=[f'p{i//2}' for i in order]))
    assert response.rewards == [expected[i] for i in order]
    assert response.successes == [i < 4 for i in order]
    assert [(error is None) for error in response.errors] == [i < 4 for i in order]
    for domain, values in response.component_rewards.items():
        for i, value in zip(order, values):
            if rows[i][1].get('domain') == domain:
                assert value == expected[i]
            else:
                assert math.isnan(value)
    return response


valid = score([0,1,2,3])
score([3,0,2,1])
score([3,1,3,0])
mixed = score([0,1,2,3,4,5])
assert all('no configured scorer' in error for error in mixed.errors[4:])
empty = score([])
assert empty.batch_size == 0 and empty.component_rewards == {'graded': [], 'strict': []}
missing = router.compute_rewards(RewardRequest(generated={'text': Texts(texts=['A','B'])}))
assert missing.rewards == [0.0,0.0] and missing.successes == [False,False]
assert all('no configured scorer' in error for error in missing.errors)
for domain, expected_mean, filled_mean in (('graded', 0.75, 0.375), ('strict', 0.5, 0.25)):
    values = torch.tensor(valid.component_rewards[domain], dtype=torch.float64, device='cpu')
    actual = _tensor_stats(domain, values)
    assert actual[domain+'_mean'] == expected_mean
    filled = _tensor_stats(domain, torch.nan_to_num(values, nan=0.0))
    assert filled[domain+'_mean'] == filled_mean
    print(domain, 'finite mean', actual[domain+'_mean'], 'zero-filled contrast', filled[domain+'_mean'])
assert _tensor_stats('none', torch.tensor([float('nan')], device='cpu')) == {}
assert router.preferred_input_kind == 'image' and graded.preferred_input_kind == 'text'
assert not any(n in sys.modules for n in ('ray', 'hydra', 'omegaconf', 'transformers', 'unirl.reward.service'))
print('valid rewards', valid.rewards, 'successes', valid.successes, 'components', valid.component_rewards)
print('unknown/missing domains:', mixed.successes, mixed.errors[4:])
print('Native routing/permutation/repeat/empty/missing-metadata and finite-stat controls passed')
print('Python', sys.version.split()[0], 'torch', torch.__version__, 'Pillow', PIL.__version__)
print('Direct RewardRequest control: router input_kind=image, NOT proof of service acceptance of text-only frontier')
print('No model weights, service/HTTP, CUDA discovery, trainer or GPU/NPU run; not upstream tests')
