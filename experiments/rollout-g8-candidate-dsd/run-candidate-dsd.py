#!/usr/bin/env python
"""Candidate DSD: ngram-spec with dynamic K schedule (vLLM 0.24.0 Dynamic SD).

Usage: run-candidate-dsd.py <input_len> <batch>  input_len: 512|2048  batch: 8|16

Same stack as candidates C/C2 (Qwen2.5-1.5B-Instruct, single 4090, gsm8k
prompts, temp 0.7/top-p 0.9, max_tokens 128). Spec method = ngram proposer
(k=5 default) + num_speculative_tokens_per_batch_size schedule that maps
runtime batch size to draft-token count K:

    (1, 8, 2), (9, 16, 3), (17, 64, 5)

i.e. small batches get fewer draft tokens (K=2 at b8, K=3 at b16, K=5 at
b32+). This directly tests the G3 memo axis-1/axis-3 open item "DSD 动态 K
行为（未测）": whether dynamic-K (vs fixed K=5 from candidate C) changes
the spec/AR throughput ratio and the per-position acceptance curve A(p)
for the ngram weak proposer.

Prints a JSON row to stdout: wall_s, decode_tok_per_s, spec/AR ratio
(against the AR baseline from candidate C summary: in512 b8/b16 =
1405.0/2630.4 tok/s, in2048 b8/b16 = 862.2/1157.2 tok/s -- read from
summary.json at runtime if available), finished_full count.
The engine's "SpecDecoding metrics: ... Per-position acceptance rate:"
stderr line carries A(p) for the max K in the schedule.
"""
import json
import os
import sys
import time

from vllm import LLM, SamplingParams

MODEL = "/home/zxw/models/Qwen2.5-1.5B-Instruct"
MAX_TOK = 128
GPU_MEM = float(os.environ.get("CANDC_GPU_MEM", "0.40"))
MAX_MODEL_LEN = int(os.environ.get("CANDC_MAX_LEN", "4096"))
# Dynamic K schedule: inclusive batch-size ranges -> draft token count.
DYNAMIC_K = [(1, 8, 2), (9, 16, 3), (17, 64, 5)]
SAMPLING = SamplingParams(max_tokens=MAX_TOK, temperature=0.7, top_p=0.9)

# AR decode baseline (tok/s incl. prefill) from candidate C summary.json
# (experiments/rollout-g8-candidate-c/raw/summary.json rows *_ar),
# same stack/stacking; used to compute the spec/AR ratio per combo.
AR_BASELINE = {
    (512, 8): 1294.6,
    (512, 16): 3156.1,
    (2048, 8): 1348.9,
    (2048, 16): 2746.8,
}


def _load_gsm8k_questions():
    import glob

    import pyarrow.parquet as pq

    cache = os.path.expanduser("~/.cache/huggingface/hub")
    pats = glob.glob(
        cache + "/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"
    )
    pats += glob.glob(
        cache + "/datasets--cais--gsm8k/snapshots/*/*/train-00000-of-00001.parquet"
    )
    if not pats:
        print("WARN no gsm8k parquet in cache (fallback random tokens)",
              flush=True)
        return None
    try:
        t = pq.read_table(pats[0])
        col = "question" if "question" in t.column_names else t.column_names[0]
        return [q.as_py() for q in t.column(col)]
    except Exception as exc:
        print("WARN gsm8k parquet read failed: %r (fallback random tokens)"
              % (exc,), flush=True)
        return None


def prompt_ids(llm, n, seed):
    tok = llm.get_tokenizer()
    src = _load_gsm8k_questions()
    if src is not None:
        ids = tok(" ".join(src[:200]))["input_ids"][:n]
        if len(ids) < n:
            ids = (ids * (n // len(ids) + 1))[:n]
        return ids
    import random

    rng = random.Random(seed)
    vocab = tok.vocab_size
    return [rng.randint(4, vocab - 1) for _ in range(n)]


def main():
    il, batch = int(sys.argv[1]), int(sys.argv[2])
    sc = dict(
        method="ngram",
        num_speculative_tokens=5,  # default K; schedule overrides per batch
        prompt_lookup_max=5,
        prompt_lookup_min=1,
        num_speculative_tokens_per_batch_size=DYNAMIC_K,
    )
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=GPU_MEM,
        max_model_len=MAX_MODEL_LEN,
        speculative_config=sc,
        trust_remote_code=True,
        disable_log_stats=False,
        enforce_eager=os.environ.get("CANDC_EAGER") == "1",
    )
    pids = prompt_ids(llm, il, seed=il)
    reqs = [{"prompt_token_ids": pids} for _ in range(batch)]
    t0 = time.monotonic()
    outs = llm.generate(reqs, SAMPLING)
    dt = time.monotonic() - t0
    try:
        llm.llm_engine.do_log_stats()
    except Exception as exc:
        print("WARN do_log_stats failed: %r" % (exc,), flush=True)
    n_ok = sum(1 for o in outs if len(o.outputs[0].token_ids) == MAX_TOK)
    tok_s = round(batch * MAX_TOK / dt, 1)
    ar = AR_BASELINE.get((il, batch))
    row = dict(
        mode="dsd-ngram",
        input_len=il,
        batch=batch,
        max_tokens=MAX_TOK,
        dynamic_k=DYNAMIC_K,
        wall_s=round(dt, 4),
        decode_tok_per_s=tok_s,
        spec_over_ar=round(tok_s / ar, 3) if ar else None,
        finished_full=n_ok,
    )
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
