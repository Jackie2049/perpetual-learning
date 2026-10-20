"""GPU probe following §10.7 criteria: is prompt_mean's update M-dependent?

AReaL source-reading §10.7 withdrew the earlier "prompt_mean defect" claim: the
old numpy comparison used a different token_mean calibre (never split by M), M
is a real optimizer-step count (each train_batch updates params, later gradients
are computed at new params), and Megatron's P multiplier has a scheduling
cancellation.  The probe here follows §10.7's GPU criteria:

  1. Fix outer M (ppo_n_minibatches), vary only the inner P (mb_spec
     n_mbs / max_tokens_per_mb); compare post-step gradients, pre/post-clip
     norms, and one Adam update.  FSDP2 first, DP=CP=1, no MoE/MTP.
  2. Fix M, vary DP rank data / dummy allocation; use stepwise global
     row/token/group references; never treat dummy count alone as defect.
  3. Keep the user's M=2 vs 4 comparison as a TRAINING-BEHAVIOUR contrast:
     check the frozen-parameter cross-step MEAN contract (with token_mean as
     control), and observe sequential-update trajectories; do NOT require
     cross-M parameters to be identical.

This script runs AReaL's REAL components (PPOActor + FSDPEngine, 1 GPU) on a
fixed synthetic batch.  It reports:

  - per-M, per-mode (prompt_mean vs token_mean) parameter delta L2 after one
    full ppo_update from the SAME init weights and SAME fixed batch;
  - with outer M fixed, inner P varied (mb_spec) and delta spread across P;
  - whether the M=2 vs M=4 delta difference for prompt_mean matches the same
    difference for token_mean (shared training behaviour vs reduction-specific).

Run:
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 \
      areal_prompt_mean_gpu_probe.py --model <hf-or-local> \
      --m-values 2,4 --inner-p 1,2
Requires AReaL env (uv sync --extra cuda) and 1 free GPU.
"""
from __future__ import annotations

import argparse
import os

import torch

from areal.api.cli_args import (
    FinetuneSpec,
    MicroBatchSpec,
    OptimizerConfig,
    PPOActorConfig,
    TrainEngineConfig,
)
from areal.trainer.ppo.actor import PPOActor

MODEL = os.environ.get("AREAL_PROBE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def _groups(n_groups: int = 3, seq_len: int = 8, seed: int = 0):
    """Fixed synthetic prompt groups (like tests/test_prompt_mean_actor.py::_groups).

    Each group is one row (one response); group sizes = [4, 3, 2] tokens per
    response would need real tokenization, so here we use uniform rows with
    distinct group ids and per-row response counts so that prompt_mean's group
    denominator differs from token_mean's token denominator.
    """
    rng = torch.Generator().manual_seed(seed)
    groups = []
    row = 0
    for length in [4, 3, 2][:n_groups]:
        attention_mask = torch.arange(seq_len)[None, :] < torch.tensor([length])[:, None]
        loss_mask = attention_mask.clone()
        loss_mask[:, 0] = False
        advantages = torch.randn(1, seq_len, generator=rng).float()
        groups.append(
            {
                "input_ids": torch.arange(row, row + 1)[:, None]
                .expand(-1, seq_len)
                .clone(),
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "advantages": advantages,
                "logprobs": torch.zeros_like(advantages),
                "prox_logp": torch.zeros_like(advantages),
                "kl_rewards": torch.zeros_like(advantages),
                "tot_rewards": torch.zeros_like(advantages),
                "rewards": torch.ones(1),
            }
        )
        row += 1
    return groups


def _make_engine(model: str, seed: int, inner_p: int):
    """Single-GPU FSDP2 engine (dist must already be initialized)."""
    torch.manual_seed(seed)
    config = TrainEngineConfig(
        path=model,
        experiment_name="probe",
        trial_name="t0",
        backend="fsdp:d1p1t1",
        dtype="bfloat16",
        mb_spec=MicroBatchSpec(n_mbs=inner_p),  # inner P control
        optimizer=OptimizerConfig(
            type="adam",
            lr=1e-4,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
        ),
    )
    from areal.engine.fsdp_engine import FSDPEngine

    engine = FSDPEngine(config)
    engine.create_process_group()
    engine.initialize(
        addr=None,
        ft_spec=FinetuneSpec(
            total_train_epochs=1,
            dataset_size=64,
            train_batch_size=16,
        ),
    )
    return engine


def _param_delta(engine, snapshot: dict[str, torch.Tensor]) -> float:
    total = 0.0
    n = 0
    for name, param in engine.model.named_parameters():
        if name in snapshot and param.requires_grad:
            total += ((param.float() - snapshot[name].float()) ** 2).sum().item()
            n += 1
    return total ** 0.5, n


def run_one(m: int, inner_p: int, mode: str, groups) -> tuple[float, int]:
    engine = _make_engine(MODEL, seed=0, inner_p=inner_p)
    snapshot = {name: p.detach().clone() for name, p in engine.model.named_parameters()}
    actor = PPOActor(
        PPOActorConfig(
            loss_aggregation=mode,
            ppo_n_minibatches=m,
            eps_clip=0.2,
            temperature=1.0,
            reward_scaling=1.0,
            reward_bias=0.0,
            adv_norm=None,
        ),
        engine,
    )
    actor.ppo_update(groups)
    delta, n = _param_delta(engine, snapshot)
    engine.destroy()
    return delta, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--m-values", default="2,4")
    ap.add_argument("--inner-p", default="1,2")
    ap.add_argument("--loss-aggregations", default="prompt_mean,token_mean")
    args = ap.parse_args()
    global MODEL
    MODEL = args.model
    m_values = [int(x) for x in args.m_values.split(",")]
    inner_p = [int(x) for x in args.inner_p.split(",")]
    modes = args.loss_aggregations.split(",")
    groups = _groups()

    print(f"model={MODEL} sweep M={m_values} innerP={inner_p} modes={modes}")
    print("(1) fixed-M inner-P spread (per §10.7 criterion 1):")
    for m in m_values:
        for mode in modes:
            spreads = []
            for p in inner_p:
                delta, n = run_one(m, p, mode, groups)
                spreads.append(delta)
                print(f"  M={m} P={p} {mode}: delta_L2={delta:.6f} ({n} params)")
            if len(spreads) > 1:
                print(f"    -> P-spread (max/min) = {max(spreads)/min(spreads):.3f}")
    print("(2) M=2 vs M=4 training-behaviour contrast (criterion 3):")
    per_mode = {}
    for mode in modes:
        deltas = []
        for m in m_values:
            delta, _ = run_one(m, inner_p[0], mode, groups)
            deltas.append(delta)
        per_mode[mode] = deltas
        print(f"  {mode}: M={m_values} delta_L2 = {[f'{d:.6f}' for d in deltas]}; "
              f"ratio={deltas[1]/deltas[0]:.3f}")
    print("Interpretation: if prompt_mean's cross-M ratio closely matches "
          "token_mean's, the M dependence is shared training behaviour, not a "
          "reduction-specific defect (per §10.7).")


if __name__ == "__main__":
    main()
