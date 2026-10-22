#!/usr/bin/env python3
"""Run native UniRL single-process token-budget planning; args: source [Pillow deps]."""
import hashlib
from pathlib import Path
import sys
from types import ModuleType

sys.dont_write_bytecode = True
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
if len(sys.argv) > 2:
    sys.path.insert(0, str(Path(sys.argv[2]).resolve()))
for name, expected in {
    'unirl/train/stack/planner/packed.py': 'b5f2789f768cbdb7d766bebb5c1d9feb316dbe32db52637e3683e8e6dc8fe70f',
    'unirl/train/stack/planner/types.py': 'd9278a269104724a63abae6eceedde694ad92b1016c415f1b88ed087dcb3e8be',
    'unirl/train/stack/planner/count.py': '1e72752d8cee7507a1ede46217245ea18a6b3a9d1f5f94b6e3954bcbd50a8d27',
    'unirl/train/stack/planner/update.py': 'bad7e8e2886925e633ec544d1d09d76599863fbc594debd6337c9d80bf03665a',
    'unirl/types/sample.py': '04c7a8caa840953e16fea401676ad30f4f1080fdf494fcf1daf095a15d78a460',
    'unirl/types/conditions/text.py': '2ab16ab547b01a5e83bae86a521f390e70a40f6822b75cb9307ecc7bc94b501b',
    'unirl/train/stack/single.py': '2ed27f80b721d800e4dab7f1234bce6d85c5eb91ce18cd95789da307c1eb3992',
}.items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert actual == expected, (name, actual)
    print(name, actual)

import unirl.distributed
import unirl.train
for name, folder in (('unirl.distributed.tensor', 'unirl/distributed/tensor'),
                     ('unirl.train.stack', 'unirl/train/stack')):
    package = ModuleType(name)
    package.__path__ = [str(root / folder)]
    sys.modules[name] = package
    parent, child = name.rsplit('.', 1)
    setattr(sys.modules[parent], child, package)

import PIL
import torch
import torch.distributed as dist
from unirl.train.stack.planner import TokenBudgetPlanner
from unirl.train.stack.planner.packed import _sample, dense, varlen_sum
from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Texts
from unirl.types.sample import Part
from unirl.types.segments import TextSegment

assert not dist.is_initialized()
lengths = [(8, 1), (1, 8), (2, 2), (20, 1), (1, 8), (8, 1), (2, 2), (1, 20)]
samples = [_sample(i, prompt=p, resp=r) for i, (p, r) in enumerate(lengths)]
assert dense(samples[:2]) == 32 and varlen_sum(samples[:2]) == 18
ids = [f'r{i}' for i in range(8)]
prompt_mask = torch.arange(20, device='cpu')[None, :] < torch.tensor(
    [p for p, _ in lengths], device='cpu')[:, None]
condition = TextTokenCondition(
    input_ids=torch.arange(8, device='cpu')[:, None].expand(8, 20).clone(),
    attention_mask=prompt_mask)
segment = TextSegment.pack(tokens=[torch.full((r,), i, dtype=torch.long, device='cpu')
                                   for i, (_, r) in enumerate(lengths)])
part = Part(sample_ids=ids, segment=segment, conditions={'prompt': condition},
            primitives={'text': Texts(texts=ids)}, metadata=[{'row': i} for i in range(8)])
expected_perm = [3, 0, 1, 2, 7, 4, 5, 6]
for model, expected in (
    ('sum', [[(0, 1), (1, 3), (3, 4)], [(4, 5), (5, 7), (7, 8)]]),
    ('dense', [[(0, 1), (1, 2), (2, 3), (3, 4)],
               [(4, 5), (5, 6), (6, 7), (7, 8)]]),
):
    planner = TokenBudgetPlanner(token_budget=18, cost_model=model)
    arranged, plan = planner.arrange(part, num_updates=2, micro_batch_size=1)
    assert plan == expected
    assert arranged.sample_ids == [ids[i] for i in expected_perm]
    assert arranged.primitives['text'].texts == arranged.sample_ids
    assert [m['row'] for m in arranged.metadata] == expected_perm
    assert arranged.segment.lengths.tolist() == [lengths[i][1] for i in expected_perm]
    assert arranged.conditions['prompt'].attention_mask.sum(-1).tolist() == [lengths[i][0] for i in expected_perm]
    assert arranged.conditions['prompt'].input_ids[:, 0].tolist() == expected_perm
    costs = []
    for update_i, micros in enumerate(plan):
        rows = [i for start, end in micros for i in expected_perm[start:end]]
        assert sorted(rows) == list(range(update_i * 4, (update_i + 1) * 4))
        for start, end in micros:
            selected = expected_perm[start:end]
            cost = (varlen_sum if model == 'sum' else dense)([samples[i] for i in selected])
            costs.append(cost)
            if cost > 18:
                assert len(selected) == 1 and selected[0] in (3, 7) and cost == 21
            micro = arranged.slice(start, end)
            assert micro.sample_ids == [ids[i] for i in selected]
            assert micro.segment.tokens.tolist() == [i for i in selected for _ in range(lengths[i][1])]
    other, other_plan = planner.arrange(part, num_updates=2, micro_batch_size=999)
    assert other_plan == plan and other.sample_ids == arranged.sample_ids
    print(model, 'perm', expected_perm, 'plan', plan, 'costs', costs)

fallback = Part(sample_ids=ids, primitives={'text': Texts(texts=ids)})
unchanged, plan = TokenBudgetPlanner(token_budget=18).arrange(fallback, num_updates=2, micro_batch_size=3)
assert unchanged is fallback and plan == [[(0, 3), (3, 4)], [(4, 7), (7, 8)]]
assert part.sample_ids == ids and part.segment.lengths.tolist() == [r for _, r in lengths]
for action in (lambda: TokenBudgetPlanner(token_budget=0),
               lambda: TokenBudgetPlanner(token_budget=18, cost_model='other'),
               lambda: TokenBudgetPlanner(token_budget=18).arrange(part, num_updates=3, micro_batch_size=1)):
    try:
        action()
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError')
assert not dist.is_initialized()
assert not any(n in sys.modules for n in ('ray', 'hydra', 'omegaconf', 'transformers', 'unirl.train.stack.single'))
print('Native Part/planner CPU controls passed; no process group, CUDA discovery, model, backward or training')
print('Two initializer shells only; no method extraction/fake classes or upstream tests')
print('Python', sys.version.split()[0], 'torch', torch.__version__, 'Pillow', PIL.__version__)
