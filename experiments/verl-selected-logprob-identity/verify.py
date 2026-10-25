#!/usr/bin/env python3
"""One FP64 CPU autograd control for RFC #7712's log-probability identity.

This is a mathematical illustration, not the author's implementation or tests.
The label log-probability comes from a dense PyTorch reference, not a fused
kernel. The dense oracle materializes [T,V]; no memory or performance claim.
No framework import, model, device discovery, padding, SP, or BF16 validation.
"""

import json

import torch


def main():
    generator = torch.Generator(device="cpu").manual_seed(7712)
    h0 = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    w0 = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    labels = torch.tensor([1, 2, 4, 5], device="cpu")
    ids = torch.tensor([1, 4], device="cpu")
    a = torch.tensor([0.2, -0.3, 0.4, 0.7], dtype=torch.float64)
    b = torch.tensor([[0.1, 0.6], [-0.5, 0.2], [0.8, -0.1], [0.3, 0.4]],
                     dtype=torch.float64)
    temperature = 0.7

    def evaluate(mode, conditional=False):
        h = h0.clone().requires_grad_()
        w = w0.clone().requires_grad_()
        logp = (h @ w.t() / temperature).log_softmax(-1)
        label_logp = logp.gather(1, labels[:, None]).squeeze(1)
        if mode == "dense":
            selected = logp.index_select(1, ids)
        else:
            z_ids = h @ w.index_select(0, ids).t() / temperature
            z_label = (h * w.index_select(0, labels)).sum(-1) / temperature
            anchor = label_logp.detach() if mode == "detach_anchor" else label_logp
            selected = anchor[:, None] + z_ids - z_label[:, None]
        if conditional:
            selected = selected - selected.logsumexp(-1, keepdim=True)
            loss = (b * selected).sum()
        else:
            # Arbitrary upstream weights on BOTH outputs, not only selected NLL.
            loss = (a * label_logp).sum() + (b * selected).sum()
        dh, dw = torch.autograd.grad(loss, (h, w))
        return selected.detach(), dh.detach(), dw.detach()

    def differences(lhs, rhs):
        return {
            key: (left - right).abs().max().item()
            for key, left, right in zip(("values", "hidden_grad", "weight_grad"), lhs, rhs)
        }

    dense = evaluate("dense")
    identity = evaluate("identity")
    detached = evaluate("detach_anchor")
    matched = differences(identity, dense)
    negative = differences(detached, dense)
    for error in matched.values():
        assert error < 1e-12, matched
    assert negative["values"] < 1e-12
    assert negative["hidden_grad"] > 1e-3
    assert negative["weight_grad"] > 1e-3

    conditional = differences(evaluate("identity", conditional=True),
                              evaluate("dense", conditional=True))
    for error in conditional.values():
        assert error < 1e-12, conditional

    uniform = torch.zeros(1, 7, dtype=torch.float64).log_softmax(-1)[:, ids]
    full_probs = uniform.exp()
    subset_probs = uniform.softmax(-1)
    assert torch.allclose(full_probs, torch.full_like(full_probs, 1 / 7))
    assert torch.allclose(subset_probs, torch.full_like(subset_probs, 1 / 2))

    print(json.dumps({
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float64",
        "shape": {"tokens": 4, "hidden": 3, "vocab": 7, "selected": 2},
        "temperature": temperature,
        "seed": 7712,
        "identity_vs_dense_max_abs_error": matched,
        "detached_anchor_negative_control": negative,
        "conditional_subset_vs_dense_max_abs_error": conditional,
        "uniform_full_vocab_probs": full_probs.tolist(),
        "uniform_conditional_subset_probs": subset_probs.tolist(),
        "scope": "math-only; dense label oracle; no fused/layout/rounding/performance validation"
    }, indent=2))


if __name__ == "__main__":
    main()
