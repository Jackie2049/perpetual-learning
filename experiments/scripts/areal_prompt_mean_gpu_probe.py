"""GPU probe: is prompt_mean's update sensitive to ppo_n_minibatches (M)?

Runs AReaL's REAL training components (PPOActor + FSDPEngine on a single GPU)
on a FIXED batch, sweeping ppo_n_minibatches in {2, 4} (and optionally {1, 3}),
and compares the resulting parameter deltas after a full ppo_update.

Faithful chain (AReaL main 3b93679, see notebook §10.3):
  - _PromptMeanBatch.for_steps sets step_scale = len(microbatches)/active_groups = M/G
    (loss_reduction.py:115), shared by both backends (actor.py:600).
  - Each of the M minibatches runs engine.train_batch -> one optimizer_step
    (actor.py:609-636). Faithful normalisation chain:
        grad_total = dp_size * (M/G) * sum_all loss*w
    is EXACTLY linear in M (numpy-verified).  token_mean/seq_mean carry no step
    factor, so they are split-invariant.

Under a real adaptive optimizer (AdamW) the absolute scale is partly absorbed by
the per-parameter normalisation, so we measure the OBSERVED parameter delta after
one full ppo_update from identical initial weights on identical data:

  M=2 vs M=4 on the SAME fixed batch and SAME init weights.
  A split-sensitive defect => prompt_mean delta norm differs strongly across M;
  token_mean (control) should show near-invariance.

Run:
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 \
      areal_prompt_mean_gpu_probe.py --model <hf-or-local> --m-values 2,4
Requires AReaL env (uv sync --extra cuda) and 1 free GPU.
"""
from __future__ import annotations

import argparse
import os

import torch

from areal.api.cli_args import (
    FinetuneSpec,
    LossAggregationMode,  # noqa: F401  (type re-export)
    OptimizerConfig,
    PPOActorConfig,
    TrainEngineConfig,
)
from areal.trainer.ppo.actor import PPOActor

MODEL = os.environ.get("AREAL_PROBE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def _groups(n_groups: int = 3, seq_len: int = 8, seed: int = 0):
    """Fixed synthetic prompt groups (like tests/test_prompt_mean_actor.py::_groups)."""
    rng = torch.Generator().manual_seed(seed)
    groups = []
    row = 0
    lengths_all = [4, 3, 2]
    for length in lengths_all[:n_groups]:
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


def _make_engine(model: str, seed: int):
    """Create a single-GPU FSDP2 engine (dist must already be initialized)."""
    torch.manual_seed(seed)
    config = TrainEngineConfig(
        path=model,
        experiment_name="probe",
        trial_name="t0",
        backend="fsdp:d1p1t1",
        dtype="bfloat16",
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
    """L2 norm of the parameter delta w.r.t. a snapshot."""
    total = 0.0
    n = 0
    for name, param in engine.model.named_parameters():
        if name in snapshot and param.requires_grad:
            total += ((param.float() - snapshot[name].float()) ** 2).sum().item()
            n += 1
    return total ** 0.5, n


def run_sweep(model: str, m_values: list[int], groups, loss_aggregation: str) -> None:
    print(f"model={model} sweep={m_values} loss_aggregation={loss_aggregation}")
    for m in m_values:
        engine = _make_engine(model, seed=0)
        snapshot = {
            name: p.detach().clone() for name, p in engine.model.named_parameters()
        }
        actor = PPOActor(
            PPOActorConfig(
                loss_aggregation=loss_aggregation,
                ppo_n_minibatches=m,
                eps_clip=0.2,
                temperature=1.0,
                reward_scaling=1.0,
                reward_bias=0.0,
                adv_norm=None,
            ),
            engine,
        )
        # NOTE: the synthetic batch lacks keys the actor reads (e.g. rewards /
        # original_rewards for stats logging). _log_configuration + logging paths
        # may need `log_agent_stats=False` (default) — fine here.
        actor.ppo_update(groups)
        delta, n = _param_delta(engine, snapshot)
        print(f"  M={m}: param_delta_L2={delta:.6f} over {n} params")
        engine.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--m-values", default="2,4")
    ap.add_argument("--loss-aggregation", default="prompt_mean")
    args = ap.parse_args()
    m_values = [int(x) for x in args.m_values.split(",")]
    run_sweep(args.model, m_values, _groups(), args.loss_aggregation)


if __name__ == "__main__":
    main()
