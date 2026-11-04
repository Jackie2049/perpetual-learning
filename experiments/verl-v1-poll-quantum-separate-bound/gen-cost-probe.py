#!/usr/bin/env python3
"""Engine-level generation-cost probe for the separate_async rollout-bound question.

Why: in separate_async the trainer consumes exactly the trajectories the rollout
produced, so the only thing that can push the trainer into
`ReplayBufferAsync.sample()`'s poll sleep is a larger *per-token* generation cost
than training cost.  The stack itself does not log generation wall time (only the
trainer's wait, `timing_s/gen`), so this probe measures the rollout side directly
on the same GPU / model / vLLM version with the same batch shape.

Scope: engine level only (vLLM offline API).  It is NOT the verl stack and does
not include weight sync, scheduling or agent-loop overhead.  Use it to bound the
ratio, not as a stack measurement.
"""
import argparse
import json
import time

import pandas as pd


def load_prompts(parquet: str, k: int):
    df = pd.read_parquet(parquet)
    col = df["prompt"].tolist()[:k]
    return col


def to_text(messages, tokenizer):
    """verl runs with data.return_raw_chat=True; the parquet holds chat lists."""
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return str(messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/zxw/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--parquet", default="/home/zxw/data/gsm8k/train_sub480.parquet")
    ap.add_argument("--prompts", type=int, default=4, help="prompts per step (=train_batch_size)")
    ap.add_argument("--n", type=int, default=16, help="samples per prompt (=rollout.n)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.6)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    raw = load_prompts(args.parquet, args.prompts)
    prompts = [to_text(p, tok) for p in raw]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        enforce_eager=False,
        disable_log_stats=True,
    )
    sp = SamplingParams(n=args.n, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens)

    rows = []
    for rep in range(args.repeats):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        prompt_tokens = sum(len(o.prompt_token_ids) for o in outs)
        gen_tokens = sum(len(c.token_ids) for o in outs for c in o.outputs)
        n_seqs = sum(len(o.outputs) for o in outs)
        rows.append(
            dict(
                repeat=rep,
                wall_s=round(dt, 4),
                n_seqs=n_seqs,
                prompt_tokens=prompt_tokens,
                gen_tokens=gen_tokens,
                total_tokens=prompt_tokens + gen_tokens,
                gen_tok_per_s=round(gen_tokens / dt, 1),
                total_tok_per_s=round((prompt_tokens + gen_tokens) / dt, 1),
                mean_resp_len=round(gen_tokens / max(n_seqs, 1), 1),
            )
        )
        print(json.dumps(rows[-1]), flush=True)

    out = dict(
        model=args.model,
        vllm=__import__("vllm").__version__,
        prompts=args.prompts,
        n=args.n,
        max_tokens=args.max_tokens,
        rows=rows,
    )
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))


if __name__ == "__main__":
    main()
