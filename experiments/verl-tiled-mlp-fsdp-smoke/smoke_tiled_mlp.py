#!/usr/bin/env python3
"""TiledMLP wiring smoke test on the real verl FSDP engine (single GPU).

Purpose: TM-006's local patch only proves the config is *forwarded* to
`apply_monkey_patch` by static analysis. This script exercises the real code
path (`FSDPEngine._build_module` -> `apply_monkey_patch`) so we can tell:

  1. with the patch, does `tiled_mlp.enabled=True` actually patch the MLP class?
  2. without the patch, is the option silently ignored (control)?
  3. do tiled and non-tiled produce the same logprobs and gradients?
  4. does `strategy=fsdp` (FSDP1) error out or silently do nothing / misbehave?
  5. what happens when total tokens are not divisible by num_shards?

One process, one rank. Everything is written to --out as JSON.

Usage:
  PYTHONPATH=<verl-tree> python smoke_tiled_mlp.py \
      --strategy fsdp2 --enabled 1 --shards 4 --seq-len 8 --batch 2 --out out.json
"""

import argparse
import io
import json
import math
import os
import time
import traceback
from contextlib import redirect_stdout
from functools import partial


def build_model_dir(path):
    """Tiny Qwen3 model saved once; every run loads the same weights."""
    if os.path.exists(os.path.join(path, "config.json")):
        return
    from transformers import AutoModelForCausalLM, Qwen3Config

    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    import torch

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    model.save_pretrained(path)
    cfg.save_pretrained(path)


def unit(name):
    return name.endswith(".mlp") or ".mlp." in name


def local(t):
    from torch.distributed.tensor import DTensor

    return t.to_local() if isinstance(t, DTensor) else t


def grad_norms(module):
    mlp, other, count = 0.0, 0.0, 0
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        g = local(p.grad).detach().float()
        sq = float(g.pow(2).sum())
        count += 1
        if unit(name):
            mlp += sq
        else:
            other += sq
    return {"mlp": math.sqrt(mlp), "other": math.sqrt(other), "tensors": count}


def fingerprint(module):
    total = 0.0
    for _, p in module.named_parameters():
        total += float(local(p.detach()).float().sum())
    return total


def make_batch(model_config, engine_config, batch_size, seq_len, world_size):
    import torch
    from verl import DataProto
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding

    response_length = max(1, seq_len // 2)
    vocab = model_config.hf_config.vocab_size
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len)
    input_ids = input_ids.remainder(vocab)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq_len, dtype=torch.long).expand(batch_size, seq_len)
    responses = input_ids[:, -response_length:]
    response_mask = torch.ones_like(responses)

    data = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, : seq_len - response_length],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
            "old_log_probs": torch.zeros_like(responses, dtype=torch.float32),
            "advantages": torch.ones_like(responses, dtype=torch.float32),
        },
        meta_info={"temperature": 1.0},
    )
    td = left_right_2_no_padding(data.to_tensordict())
    tu.assign_non_tensor(
        td,
        global_batch_size=batch_size * world_size,
        use_remove_padding=model_config.get("use_remove_padding", False),
        use_dynamic_bsz=engine_config.use_dynamic_bsz,
        max_token_len_per_gpu=engine_config.max_token_len_per_gpu,
        micro_batch_size_per_gpu=engine_config.micro_batch_size_per_gpu,
        use_fused_kernels=engine_config.use_fused_kernels,
    )
    return td


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="fsdp2", choices=["fsdp", "fsdp2"])
    ap.add_argument("--enabled", type=int, default=0)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-checkpointing", type=int, default=0)
    ap.add_argument("--mp-dtype", default="", help="force FSDP mixed precision param dtype, e.g. fp32")
    ap.add_argument("--use-orig-params", type=int, default=0)
    ap.add_argument("--reshard-after-forward", type=int, default=1)
    ap.add_argument("--model-dir", default="/tmp/m5-tiled-mlp/tiny-qwen3")
    ap.add_argument("--rdzv", default="/tmp/m5-tiled-mlp/rdzv-smoke")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    from verl.trainer.config import CheckpointConfig
    from verl.utils.device import get_nccl_backend, get_torch_device
    from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig

    result = {
        "strategy": args.strategy,
        "enabled": bool(args.enabled),
        "shards": args.shards,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "use_orig_params": bool(args.use_orig_params),
        "reshard_after_forward": bool(args.reshard_after_forward),
        "ok": False,
    }

    get_torch_device().set_device(0)
    dist.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{args.rdzv}",
        rank=0,
        world_size=1,
    )
    build_model_dir(args.model_dir)

    from verl.workers.engine import EngineRegistry
    from verl.workers.config import ActorConfig
    from verl.workers.utils.losses import ppo_loss

    model_config = HFModelConfig(
        path=args.model_dir,
        load_tokenizer=False,
        override_config={"attn_implementation": "sdpa"},
        use_remove_padding=True,
        tiled_mlp={"enabled": bool(args.enabled), "num_shards": args.shards},
        enable_gradient_checkpointing=bool(args.grad_checkpointing),
    )

    engine_config = FSDPEngineConfig(
        forward_only=False,
        fsdp_size=1,
        strategy=args.strategy,
        ulysses_sequence_parallel_size=1,
        use_remove_padding=True,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=args.batch,
        max_token_len_per_gpu=4096,
        use_torch_compile=False,
        use_orig_params=bool(args.use_orig_params),
        reshard_after_forward=bool(args.reshard_after_forward),
    )
    if args.mp_dtype:
        engine_config = FSDPEngineConfig(
            forward_only=False,
            fsdp_size=1,
            strategy=args.strategy,
            ulysses_sequence_parallel_size=1,
            use_remove_padding=True,
            use_dynamic_bsz=False,
            micro_batch_size_per_gpu=args.batch,
            max_token_len_per_gpu=4096,
            use_torch_compile=False,
            use_orig_params=bool(args.use_orig_params),
            reshard_after_forward=bool(args.reshard_after_forward),
            mixed_precision={
                "param_dtype": args.mp_dtype,
                "reduce_dtype": "fp32",
                "buffer_dtype": "fp32",
            },
        )
    engine = EngineRegistry.new(
        model_type="language_model",
        backend=args.strategy,
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=FSDPOptimizerConfig(lr=1e-3, lr_scheduler_type="constant"),
        checkpoint_config=CheckpointConfig(),
    )

    from verl.models.transformers.tiled_mlp import _MODEL_TYPE_TO_MLP_CLASS

    mlp_module, mlp_class_name = _MODEL_TYPE_TO_MLP_CLASS[model_config.hf_config.model_type]
    import importlib

    mlp_class = getattr(importlib.import_module(mlp_module), mlp_class_name)
    result["mlp_class"] = mlp_class_name
    result["mlp_forward_before_init"] = getattr(mlp_class.forward, "__name__", "?")

    buf = io.StringIO()
    t0 = time.time()
    with redirect_stdout(buf):
        engine.initialize()
    result["init_s"] = round(time.time() - t0, 1)
    result["init_log_tail"] = [
        line
        for line in buf.getvalue().splitlines()
        if "TiledMLP" in line or "monkey patch" in line.lower()
    ]
    result["mlp_forward_after_init"] = getattr(mlp_class.forward, "__name__", "?")
    result["patched"] = result["mlp_forward_after_init"] == "tiled_forward"
    result["param_fingerprint"] = fingerprint(engine.module)
    result["param_numel"] = sum(p.numel() for p in engine.module.parameters())

    actor_config = ActorConfig(strategy=args.strategy, rollout_n=1, ppo_micro_batch_size_per_gpu=-1)
    loss_fn = partial(ppo_loss, config=actor_config)
    batch = make_batch(model_config, engine_config, args.batch, args.seq_len, 1)
    from verl.utils import tensordict_utils as tu

    tu.assign_non_tensor(batch, return_model_output=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        with engine.train_mode(zero_grad_on_exit=False):
            engine.optimizer_zero_grad()
            out = engine.forward_backward_batch(batch, loss_function=loss_fn, forward_only=False)
        result["step_s"] = round(time.time() - t0, 2)
        result["ok"] = True
        result["grad_norms"] = grad_norms(engine.module)
        # Reporting only; a failure here must not hide the gradient evidence above.
        try:
            result["metrics"] = {
                k: round(float(v), 8)
                for k, v in (out.get("metrics") or {}).items()
                if isinstance(v, (int, float))
            }
            result["model_output_keys"] = sorted((out.get("model_output") or {}).keys())
            if out.get("model_output"):
                lp = out["model_output"]["log_probs"]
                flat = lp.to_padded_tensor(0.0) if lp.is_nested else lp
                result["logprobs"] = {
                    "nested": bool(lp.is_nested),
                    "numel": int(flat.numel()),
                    "sum": float(flat.float().sum()),
                    "first": [round(float(v), 6) for v in flat.flatten()[:8]],
                }
        except Exception as exc:  # noqa: BLE001
            result["logprob_error"] = f"{type(exc).__name__}: {exc}"[:300]
    except Exception as exc:  # noqa: BLE001 - the failure mode is the deliverable
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:1200]
        result["traceback_tail"] = traceback.format_exc().splitlines()[-6:]
        result["step_s"] = round(time.time() - t0, 2)
    result["peak_mem_bytes"] = torch.cuda.max_memory_allocated()

    dist.barrier()
    dist.destroy_process_group()
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: result[k] for k in ("strategy", "enabled", "patched", "ok")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
