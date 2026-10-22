#!/usr/bin/env python3
"""Check native UniRL EMA on tiny CPU parameters; argument: UniRL source root."""
import hashlib
from pathlib import Path
import sys

sys.dont_write_bytecode = True
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
for name, expected in {
    'unirl/train/ema.py': 'b4b057cd565780b467c56d0ca8032a48c0e5b5366702fbd11cf2f06c9654701b',
    'unirl/train/configs.py': '18dcc241700eeaf83f6e1aa9cc3a73519350c60c83d1c1631f7a8c574e29e03e',
    'unirl/distributed/local.py': 'd11db1a148f341bdca61b4c6093e728232de315e9ca14cbdcafba0156af19621',
    'unirl/models/types/post_materialize.py': '6e18a289256b474d8725b10e9f65ff76857318c1851f6355970d2e6c1345f329',
    'unirl/train/backend/base_backend.py': '27f72e673925eac566beaceb95f6975ba9cb455faeb0434d3743e2efa707bc47',
}.items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert actual == expected, (name, actual)
    print(name, actual)

import torch
from torch import nn
from unirl.models.types.post_materialize import apply_deferred_ops
from unirl.train.configs import EmaFullConfig, EmaLoraConfig
from unirl.train.ema import EMA, inject_mirror, make_decay_fn


def close(actual, expected):
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64, device='cpu'))


def put(model, values):
    with torch.no_grad():
        model.w.copy_(torch.tensor(values, dtype=torch.float64, device='cpu'))


def mirror():
    model = nn.Module()
    model.register_parameter('w', nn.Parameter(torch.tensor([2., 4.], dtype=torch.float64, device='cpu')))
    model.register_parameter('frozen', nn.Parameter(torch.tensor([9.], dtype=torch.float64, device='cpu'), requires_grad=False))
    model.register_buffer('buffer', torch.tensor([7.], dtype=torch.float64, device='cpu'))
    shadow = inject_mirror(model)
    assert set(dict(model.named_parameters())) == {'w', 'frozen', 'shadow_w'}
    assert len(list(shadow.iter_pairs())) == 1 and not model.shadow_w.requires_grad
    assert len(model._deferred_ops) == 1
    put(model, [3., 5.])
    apply_deferred_ops(model)
    assert model._deferred_ops == []
    close(model.shadow_w, [3., 5.])
    return model, shadow


for cfg, times, expected in (
    (EmaFullConfig(target_decay=0.5), [0, 1, 10, 100], [0.1, 2/11, 0.5, 0.5]),
    (EmaLoraConfig(ema_decay=0.25), [0, 1, 10], [0.25, 0.25, 0.25]),
    (EmaLoraConfig(ema_decay_type='linear', ema_uprate=0.1, ema_uphold=0.5), [0, 2, 5, 9], [0., 0.2, 0.5, 0.5]),
    (EmaLoraConfig(ema_decay_type='warmup', ema_flat_steps=2, ema_uprate=0.1, ema_uphold=0.5), [0, 1, 2, 3, 7], [0., 0., 0., 0.1, 0.5]),
):
    actual = [make_decay_fn(cfg)(t) for t in times]
    assert actual == expected, (actual, expected)
    print(type(cfg).__name__, getattr(cfg, 'ema_decay_type', 'full'), list(zip(times, actual)))

step_model, step_shadow = mirror()
step_cfg = EmaFullConfig(target_decay=0.5)
step_ema = EMA(shadow=step_shadow, decay_fn=make_decay_fn(step_cfg), timing=step_cfg.timing)
put(step_model, [13., 25.])
step_ema.on_rollout_end(0)
close(step_model.shadow_w, [3., 5.])
step_ema.step(0)
close(step_model.shadow_w, [12., 23.])
put(step_model, [23., 45.])
step_ema.step(1)
close(step_model.shadow_w, [21., 41.])

roll_model, roll_shadow = mirror()
roll_cfg = EmaFullConfig(target_decay=0.5, timing='rollout_end')
roll_ema = EMA(shadow=roll_shadow, decay_fn=make_decay_fn(roll_cfg), timing=roll_cfg.timing)
put(roll_model, [13., 25.])
roll_ema.step(0)
put(roll_model, [23., 45.])
roll_ema.step(1)
close(roll_model.shadow_w, [3., 5.])
roll_ema.on_rollout_end(2)
close(roll_model.shadow_w, [18., 35.])
print('same live sequence: optimizer_step shadow', step_model.shadow_w.tolist(),
      'rollout_end shadow', roll_model.shadow_w.tolist())

live_id, shadow_id = id(step_model.w), id(step_model.shadow_w)
live_ptr, shadow_ptr = step_model.w.data_ptr(), step_model.shadow_w.data_ptr()
try:
    with step_ema.use_shadow():
        close(step_model.w, [21., 41.])
        close(step_model.shadow_w, [23., 45.])
        assert step_model.w.data_ptr() == shadow_ptr and step_model.shadow_w.data_ptr() == live_ptr
        raise RuntimeError('intentional context exit')
except RuntimeError as exc:
    assert str(exc) == 'intentional context exit'
close(step_model.w, [23., 45.])
close(step_model.shadow_w, [21., 41.])
assert (id(step_model.w), id(step_model.shadow_w)) == (live_id, shadow_id)
assert (step_model.w.data_ptr(), step_model.shadow_w.data_ptr()) == (live_ptr, shadow_ptr)
copy_cfg = EmaLoraConfig(ema_decay=0.0, timing='optimizer_step')
EMA(shadow=step_shadow, decay_fn=make_decay_fn(copy_cfg), timing=copy_cfg.timing).step(2)
close(step_model.shadow_w, [23., 45.])
for model in (step_model, roll_model):
    close(model.frozen, [9.])
    close(model.buffer, [7.])
    assert model.w.requires_grad and not model.shadow_w.requires_grad
    assert all(p.grad is None for p in model.parameters())
assert not torch.distributed.is_initialized()
assert not any(n in sys.modules for n in ('ray', 'hydra', 'omegaconf', 'peft', 'transformers', 'PIL', 'unirl.train.backend.base_backend'))
print('Native config/EMA/mirror/deferred-copy/context controls passed; normal imports, no shells or extracted methods')
print('Python', sys.version.split()[0], 'torch', torch.__version__)
print('Tiny CPU parameters only: no forward/backward, optimizer, LoRA injection, pretrained weights, device discovery or GPU/NPU')
