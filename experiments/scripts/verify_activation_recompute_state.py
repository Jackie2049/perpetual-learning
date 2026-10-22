"""CPU-only eager checkpoint controls; not a model or memory benchmark.

Run with Python 3.10+ and PyTorch 2.9.1. No device discovery or remote access.
"""

import hashlib
import inspect
import random

import torch
import torch.utils.checkpoint as cp


def close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def stochastic(kind, checkpointed, preserve=True):
    # Seed only the CPU generator; do not call torch.manual_seed's device hooks.
    torch.random.default_generator.manual_seed(123)
    external_rng = random.Random(123)
    x = torch.arange(1, 65, dtype=torch.float64, device="cpu").requires_grad_()

    def forward(value):
        if kind == "torch":
            value = torch.nn.functional.dropout(value, p=0.5, training=True)
        else:
            value = value * external_rng.random()
        return value.square()

    output = (cp.checkpoint(forward, x, use_reentrant=False,
                            preserve_rng_state=preserve)
              if checkpointed else forward(x))
    output.sum().backward()
    return output.detach(), x.grad, torch.get_rng_state(), external_rng.getstate()


def mutable_scale(checkpointed, change):
    state = {"scale": 2.0}
    x = torch.tensor([1.0, 2.0], dtype=torch.float64, device="cpu", requires_grad=True)

    def forward(value):
        return (value * state["scale"]).square()

    output = (cp.checkpoint(forward, x, use_reentrant=False,
                            determinism_check="default")
              if checkpointed else forward(x))
    if change:
        state["scale"] = 3.0
    output.sum().backward()
    return output.detach(), x.grad


def side_effects(early_stop):
    visits = {"entry": 0, "exit": 0}
    x = torch.tensor([0.2, 0.4], dtype=torch.float64, device="cpu", requires_grad=True)

    def forward(value):
        visits["entry"] += 1
        output = value.sin().square()
        visits["exit"] += 1
        return output

    output = cp.checkpoint(forward, x, use_reentrant=False, early_stop=early_stop)
    output.sum().backward()
    return visits, x.grad


def metadata_mismatch():
    state = {"shape": (2, 2)}
    x = torch.arange(1, 5, dtype=torch.float64, device="cpu").requires_grad_()

    def forward(value):
        return value.reshape(state["shape"]).square().sum()

    output = cp.checkpoint(forward, x, use_reentrant=False, determinism_check="default")
    state["shape"] = (4,)
    try:
        output.backward()
    except cp.CheckpointError as exc:
        assert "metadata" in str(exc)
        return type(exc).__name__
    raise AssertionError("changed saved-tensor shape was not rejected")


def main():
    print("torch", torch.__version__, "git", torch.version.git_version)
    print("checkpoint source SHA256",
          hashlib.sha256(inspect.getsource(cp).encode()).hexdigest())
    base = stochastic("torch", False)
    kept = stochastic("torch", True)
    dropped = stochastic("torch", True, preserve=False)
    for left, right in zip(base[:3], kept[:3]):
        close(left, right)
    close(base[0], dropped[0])
    assert not torch.equal(base[1], dropped[1])
    assert not torch.equal(base[2], dropped[2])
    print("torch dropout: preserve=True output/gradient/CPU RNG equal; False gradient/RNG differ")

    external_base = stochastic("python", False)
    external_ckpt = stochastic("python", True)
    close(external_base[0], external_ckpt[0])
    assert not torch.equal(external_base[1], external_ckpt[1])
    assert external_base[3] != external_ckpt[3]
    print("Python Random: original output equal, gradient and external RNG differ despite preserve=True")

    stable = mutable_scale(False, change=True)
    unchanged = mutable_scale(True, change=False)
    changed = mutable_scale(True, change=True)
    close(stable[0], changed[0])
    close(stable[1], unchanged[1])
    close(stable[1], torch.tensor([8.0, 16.0], dtype=torch.float64, device="cpu"))
    close(changed[1], torch.tensor([12.0, 24.0], dtype=torch.float64, device="cpu"))
    print("mutable scale: default metadata check passes; baseline grad", stable[1].tolist(),
          "recompute grad", changed[1].tolist())

    partial_visits, partial_grad = side_effects(True)
    full_visits, full_grad = side_effects(False)
    assert partial_visits == {"entry": 2, "exit": 1}
    assert full_visits == {"entry": 2, "exit": 2}
    close(partial_grad, full_grad)
    print("early_stop: visits", partial_visits, "vs", full_visits, "; gradients equal")
    print("saved-tensor shape mismatch:", metadata_mismatch())
    print("PASS: eager CPU controls only; no FSDP, compile, device or performance validation")


if __name__ == "__main__":
    main()
