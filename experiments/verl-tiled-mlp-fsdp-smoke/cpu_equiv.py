#!/usr/bin/env python3
"""Exact (fp64, CPU) equivalence check for the TiledMLP monkey patch.

The GPU engine smoke runs in reduced precision, so a 1e-5..1e-3 relative
difference cannot distinguish "same math, different rounding order" from
"subtly wrong backward". This script removes precision from the question:
same weights, fp64, CPU, compare forward output / input grad / weight grads
between the plain Qwen3MLP.forward and the patched tiled_forward.

It also probes shard counts where torch.chunk returns *fewer* chunks than
requested, because TiledMLP.backward finalises parameter grads only on
"the last shard" (i + 1 == shards).

Usage: PYTHONPATH=<verl-tree> python cpu_equiv.py --out cpu_equiv.json
"""

import argparse
import copy
import json

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from verl.models.transformers.tiled_mlp import apply_tiled_mlp_monkey_patch


def run(mlp, x, patched, shards):
    x = x.clone().detach().requires_grad_(True)
    mlp.zero_grad(set_to_none=True)
    out = mlp(x)
    out.sum().backward()
    grads = {n: p.grad.clone() for n, p in mlp.named_parameters()}
    return out.detach(), x.grad.clone(), grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpu_equiv.json")
    args = ap.parse_args()

    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    torch.manual_seed(0)
    mlp = Qwen3MLP(cfg).double()
    weights = copy.deepcopy(mlp.state_dict())
    orig_forward = Qwen3MLP.forward

    rows = []
    cases = [(16, 4), (18, 4), (17, 4), (9, 4), (5, 4), (3, 4), (1, 4), (64, 8), (64, 16)]
    for tokens, shards in cases:
        torch.manual_seed(1234)
        x = torch.randn(1, tokens, cfg.hidden_size, dtype=torch.float64)

        Qwen3MLP.forward = orig_forward
        mlp.load_state_dict(weights)
        o0, gx0, gw0 = run(mlp, x, False, shards)

        Qwen3MLP.forward = orig_forward
        apply_tiled_mlp_monkey_patch(num_shards=shards, model_type="qwen3")
        mlp.load_state_dict(weights)
        try:
            o1, gx1, gw1 = run(mlp, x, True, shards)
            err = None
        except Exception as exc:  # noqa: BLE001
            rows.append({"tokens": tokens, "shards": shards, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        finally:
            Qwen3MLP.forward = orig_forward

        chunks = len(torch.chunk(torch.arange(tokens), shards))
        rows.append(
            {
                "tokens": tokens,
                "shards": shards,
                "chunks_returned": chunks,
                "last_shard_reached": chunks >= shards,
                "out_max_abs": float((o1 - o0).abs().max()),
                "xgrad_max_abs": float((gx1 - gx0).abs().max()),
                "wgrad_max_abs": max(float((gw1[n] - gw0[n]).abs().max()) for n in gw0),
                "wgrad_norm": float(sum(float(gw0[n].pow(2).sum()) for n in gw0) ** 0.5),
                "error": err,
            }
        )

    print(f"{'tok':>4s} {'sh':>3s} {'chunks':>6s} {'out':>10s} {'xgrad':>10s} {'wgrad':>10s} {'norm':>10s}")
    for r in rows:
        if r.get("error"):
            print(f"{r['tokens']:4d} {r['shards']:3d}   ERROR {r['error']}")
        else:
            print(
                f"{r['tokens']:4d} {r['shards']:3d} {r['chunks_returned']:6d} "
                f"{r['out_max_abs']:10.2e} {r['xgrad_max_abs']:10.2e} "
                f"{r['wgrad_max_abs']:10.2e} {r['wgrad_norm']:10.4f}"
            )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"dtype": "float64", "rows": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
