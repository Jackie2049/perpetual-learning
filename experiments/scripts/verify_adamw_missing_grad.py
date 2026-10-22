#!/usr/bin/env python3
"""Real CPU AdamW: missing/zero gradients, conditional branches and stale grads.

Python >=3.10 and torch; run directly, no model/checkpoint/data download.
All tensors are CPU float64; foreach/fused/capturable are disabled. No device
discovery, distributed workers, framework patches, or performance assertions.
"""

import copy
import hashlib
import inspect
import math
import sys

sys.dont_write_bytecode = True
import torch
from torch.optim import adam

LR, WD, B1, B2, EPS = 0.1, 0.2, 0.9, 0.999, 1e-8


def parameter(value=1.0):
    return torch.nn.Parameter(torch.tensor(value, dtype=torch.float64, device="cpu"))


def optimizer(params, *, decay=WD):
    return torch.optim.AdamW(params, lr=LR, betas=(B1, B2), eps=EPS,
                             weight_decay=decay, foreach=False, fused=False,
                             capturable=False, differentiable=False)


def close(value, expected):
    assert math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-12), (value, expected)


def reference(theta, m, v, step, grad, decay=WD):
    """Scalar equation control, not a replacement optimizer under test."""
    step += 1
    m = B1 * m + (1 - B1) * grad
    v = B2 * v + (1 - B2) * grad * grad
    theta = theta * (1 - LR * decay) - LR * (m / (1 - B1**step)) / (math.sqrt(v / (1 - B2**step)) + EPS)
    return theta, m, v, step


def assert_state(opt, p, expected):
    theta, m, v, step = expected
    state = opt.state[p]
    assert p.device.type == "cpu" and p.dtype == torch.float64
    assert all(state[key].device.type == "cpu" for key in ("step", "exp_avg", "exp_avg_sq"))
    assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == p.dtype
    close(p.detach().item(), theta)
    close(state["exp_avg"].item(), m)
    close(state["exp_avg_sq"].item(), v)
    assert state["step"].item() == step


def main():
    print("Python", sys.version.split()[0], "torch", torch.__version__, "git", torch.version.git_version)
    for method in (torch.optim.Optimizer.zero_grad, torch.optim.Adam._init_group,
                   adam._single_tensor_adam):
        print(method.__qualname__, "sha256=" + hashlib.sha256(inspect.getsource(method).encode()).hexdigest())

    # A virgin None gradient is not materialized even by set_to_none=False.
    missing, zero = parameter(), parameter()
    opt = optimizer([missing, zero])
    opt.zero_grad(set_to_none=False)
    assert missing.grad is None and zero.grad is None
    zero.grad = torch.zeros_like(zero)
    opt.step()
    assert missing not in opt.state and missing.grad is None
    close(missing.detach().item(), 1.0)
    assert_state(opt, zero, (0.98, 0.0, 0.0, 1))
    print("fresh None: parameter=1, no optimizer state; fresh zero: parameter=0.98, step=1")

    # The actual implementation matches bias-corrected, decoupled equations.
    for opt_cls, decay in ((torch.optim.AdamW, WD), (torch.optim.Adam, 0.0)):
        p = parameter()
        opt = opt_cls([p], lr=LR, betas=(B1, B2), eps=EPS, weight_decay=decay,
                      foreach=False, fused=False, capturable=False)
        expected = (1.0, 0.0, 0.0, 0)
        for grad in (1.0, 0.0, -2.0):
            p.grad = torch.tensor(grad, dtype=torch.float64, device="cpu")
            opt.step()
            expected = reference(*expected, grad, decay=decay)
            assert_state(opt, p, expected)
        print(opt_cls.__name__, "three equation controls passed; final=", p.detach().item())

    warm_results = {}
    for set_none in (True, False):
        p = parameter()
        opt = optimizer([p])
        p.backward()  # Actual autograd produces gradient 1.
        opt.step()
        expected = reference(1.0, 0.0, 0.0, 0, 1.0)
        assert_state(opt, p, expected)
        saved_state = copy.deepcopy(opt.state[p])
        opt.zero_grad(set_to_none=set_none)
        assert (p.grad is None) == set_none
        if not set_none:
            assert p.grad.item() == 0.0
        opt.step()
        if set_none:
            for key in saved_state:
                assert torch.equal(opt.state[p][key], saved_state[key])
        else:
            expected = reference(*expected, 0.0)
        assert_state(opt, p, expected)
        warm_results[set_none] = p.detach().item()
        print("warm set_to_none=", set_none, "parameter=", p.detach().item(),
              "step=", opt.state[p]["step"].item(), "m=", opt.state[p]["exp_avg"].item())
    assert warm_results[False] < warm_results[True]

    # No decay, but the old first moment still moves a parameter with zero grad.
    p = parameter()
    opt = optimizer([p], decay=0.0)
    p.backward()
    opt.step()
    before = p.detach().item()
    opt.zero_grad(set_to_none=False)
    opt.step()
    assert p.detach().item() < before
    print("weight_decay=0, warm zero grad still moves:", before, "->", p.detach().item())

    # Real two-parameter backward: q is unused only on the SECOND iteration.
    for set_none in (True, False):
        p, q = parameter(), parameter(2.0)
        opt = optimizer([p, q])
        (p + q).backward()
        opt.step()
        q_before = q.detach().item()
        opt.zero_grad(set_to_none=set_none)
        p.square().backward()
        assert (q.grad is None) == set_none
        opt.step()
        assert opt.state[p]["step"].item() == 2
        assert opt.state[q]["step"].item() == (1 if set_none else 2)
        assert (q.detach().item() == q_before) == set_none
        print("branch set_to_none=", set_none, "unused q:", q_before, "->", q.detach().item(),
              "parameter steps=", (opt.state[p]["step"].item(), opt.state[q]["step"].item()))

    # requires_grad_(False) controls new autograd work, not an existing .grad.
    for clear in (False, True):
        p, q = parameter(), parameter(2.0)
        opt = optimizer([p, q])
        (p + q).backward()
        opt.step()
        p_before = p.detach().item()
        p.requires_grad_(False)
        if clear:
            opt.zero_grad(set_to_none=True)
        else:
            q.grad = None  # Intentionally leave p's previous gradient untouched.
            assert p.grad.item() == 1.0
        q.square().backward()
        assert (p.grad is None) == clear
        opt.step()
        assert not p.requires_grad
        assert (p.detach().item() == p_before) == clear
        assert opt.state[p]["step"].item() == (1 if clear else 2)
        print("freeze clear_stale_grad=", clear, "p:", p_before, "->", p.detach().item())

    print("All real CPU optimizer/autograd controls passed.")
    print("NOT RUN: upstream tests, CUDA/fused/foreach, AMP, DDP/FSDP, conditional model training or performance")


if __name__ == "__main__":
    main()
