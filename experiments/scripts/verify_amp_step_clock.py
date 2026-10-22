#!/usr/bin/env python3
"""Real CPU GradScaler: optimizer skips, scheduler clocks and independent steps.

Python >=3.10, torch with torch.amp.GradScaler("cpu") support (tested 2.9.1).
Synthetic FP32 scalar losses, no autocast/FP16 overflow claim, model or devices.
No device discovery, distributed initialization, private scaler state or stubs.
Optimizer post-hooks count ordinary non-fused SGD/AdamW calls, not all AMP paths.
"""

import copy
import hashlib
import inspect
import math
import sys
import warnings

sys.dont_write_bytecode = True
import torch


def parameter(value=10.0):
    return torch.nn.Parameter(torch.tensor(value, dtype=torch.float32, device="cpu"))


def scaler():
    result = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=2)
    assert result.is_enabled()
    return result


def close(actual, expected):
    assert math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-6), (actual, expected)


def count_steps(opt):
    calls = []
    handle = opt.register_step_post_hook(lambda *_: calls.append(1))
    return calls, handle


def scheduler_trace(gate_on_call):
    p = parameter()
    opt = torch.optim.SGD([p], lr=0.1, foreach=False, fused=False)
    schedule = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    amp = scaler()
    calls, handle = count_steps(opt)
    rows = []
    for coefficient in (2.0, float("inf"), 2.0, 2.0):
        opt.zero_grad(set_to_none=True)
        loss = p * coefficient
        amp.scale(loss).backward()
        lr_used, scale_before, previous_calls = opt.param_groups[0]["lr"], amp.get_scale(), len(calls)
        result = amp.step(opt)
        amp.update()
        stepped = len(calls) > previous_calls
        assert result is None  # Both a successful SGD step and a skipped step.
        if not gate_on_call or stepped:
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                schedule.step()
            assert not seen, [str(item.message) for item in seen]
        rows.append((stepped, lr_used, p.item(), scale_before, amp.get_scale(), schedule.last_epoch))
    handle.remove()
    assert [row[0] for row in rows] == [True, False, True, True]
    assert [row[4] for row in rows] == [8.0, 4.0, 4.0, 8.0]
    assert len(calls) == 3
    expected_lrs = [0.1, 0.05, 0.05, 0.025] if gate_on_call else [0.1, 0.05, 0.025, 0.0125]
    for row, expected in zip(rows, expected_lrs):
        close(row[1], expected)
    close(p.item(), 9.65 if gate_on_call else 9.725)
    assert schedule.last_epoch == (3 if gate_on_call else 4)
    print("scheduler", "optimizer-call gated" if gate_on_call else "unconditional", rows)


def adamw_skip(nonfinite):
    p = parameter(1.0)
    opt = torch.optim.AdamW([p], lr=0.1, weight_decay=0.2, foreach=False, fused=False)
    amp = scaler()
    calls, handle = count_steps(opt)
    amp.scale(p * 2).backward()
    amp.step(opt)
    amp.update()
    before = p.detach().clone(), copy.deepcopy(opt.state[p])
    opt.zero_grad(set_to_none=True)
    amp.scale(p * nonfinite).backward()
    assert not torch.isfinite(p.grad).item()
    assert amp.step(opt) is None
    amp.update()
    assert len(calls) == 1 and torch.equal(p, before[0])
    assert opt.state[p].keys() == before[1].keys()
    for key, value in before[1].items():
        assert torch.equal(opt.state[p][key], value), key
    assert amp.get_scale() == 4.0 and opt.state[p]["step"].item() == 1
    # Skipping does not clear the nonfinite gradient. Caller clears before recovery.
    assert not torch.isfinite(p.grad).item()
    opt.zero_grad(set_to_none=True)
    amp.scale(p * 2).backward()
    amp.step(opt)
    amp.update()
    assert len(calls) == 2 and opt.state[p]["step"].item() == 2
    assert torch.isfinite(p).item()
    handle.remove()
    print("AdamW", nonfinite, "skip preserved parameter/moments/step; recovery step=2")


def loss_vs_gradient():
    p = parameter()
    opt = torch.optim.SGD([p], lr=0.1, foreach=False, fused=False)
    amp = scaler()
    calls, handle = count_steps(opt)
    loss = p + torch.tensor(float("inf"), device="cpu")
    assert not torch.isfinite(loss).item()
    amp.scale(loss).backward()
    assert torch.isfinite(p.grad).item()
    amp.step(opt)
    amp.update()
    close(p.item(), 9.9)
    assert len(calls) == 1
    opt.zero_grad(set_to_none=True)
    loss = p * 2
    assert torch.isfinite(loss).item()
    amp.scale(loss).backward()
    p.grad.fill_(float("inf"))  # Explicit synthetic corruption, not FP16 overflow.
    amp.step(opt)
    amp.update()
    close(p.item(), 9.9)
    assert len(calls) == 1 and amp.get_scale() == 4.0
    handle.remove()
    print("loss vs gradient: infinite loss/finite grad stepped; finite loss/injected inf grad skipped")


def independent_optimizers(reverse):
    p, q = parameter(), parameter()
    good = torch.optim.SGD([p], lr=0.1, foreach=False, fused=False)
    bad = torch.optim.SGD([q], lr=0.1, foreach=False, fused=False)
    amp = scaler()
    good_calls, good_handle = count_steps(good)
    bad_calls, bad_handle = count_steps(bad)
    amp.scale(p * 2 + q * float("inf")).backward()
    for opt in ((bad, good) if reverse else (good, bad)):
        assert amp.step(opt) is None
    amp.update()  # Once, after both optimizers.
    close(p.item(), 9.8)
    close(q.item(), 10.0)
    assert len(good_calls) == 1 and not bad_calls and amp.get_scale() == 4.0
    good_handle.remove()
    bad_handle.remove()
    print("two optimizers", "bad-first" if reverse else "good-first", "p=9.8 q=10 scale=4")


def main():
    print("Python", sys.version.split()[0], "torch", torch.__version__, "git", torch.version.git_version)
    for obj in (torch.amp.GradScaler, torch.optim.SGD.step, torch.optim.AdamW.step,
                torch.optim.lr_scheduler.StepLR):
        print(obj.__qualname__, "sha256=" + hashlib.sha256(inspect.getsource(obj).encode()).hexdigest())
    scheduler_trace(False)
    scheduler_trace(True)
    adamw_skip(float("inf"))
    adamw_skip(float("nan"))
    loss_vs_gradient()
    independent_optimizers(False)
    independent_optimizers(True)
    print("PASS: 2 scheduler traces, 2 AdamW skip/recovery controls, loss/gradient pair, 2 optimizer orders")


if __name__ == "__main__":
    main()
