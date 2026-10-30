#!/usr/bin/env python
"""Candidate C single-combo measurement: AR baseline or ngram-spec decode.

Usage: run-candidate-c.py <mode> <input_len> <batch>
  mode      : ar | spec
  input_len : 512 | 2048   (prompt token count)
  batch     : 8 | 16       (sequences per generate call)

Measure one (mode, input_len, batch) combo with vLLM offline engine
(Qwen2.5-1.5B-Instruct, single 4090). Prints a JSON row to stdout:
wall_s, decode_tok_per_s (inclusive of prefill, direction-only).
The vLLM engine writes its own logs to stderr, including the
"SpecDecoding metrics: ... Per-position acceptance rate: ..." line for
spec mode (engine-internal per-position acceptance curve A(p)).

Anchor: vllm 0.24.0+cu129 wheel (anchor-drift 0.24.0, see anchor.json),
same stack as candidates A/B. Spec method = ngram proposer (no draft
model download, k=5, lookup window 1..5, ratio 0.8).
"""
import json
import os
import sys
import time

from vllm import LLM, SamplingParams

MODEL = "/home/zxw/models/Qwen2.5-1.5B-Instruct"
MAX_TOK = 128
GPU_MEM = float(os.environ.get("CANDC_GPU_MEM", "0.40"))
SPEC_CONFIG = dict(
    method="ngram",
    num_speculative_tokens=5,
    prompt_lookup_max=5,
    prompt_lookup_min=1,
)
# RL-typical sampling (temp 0.7 / top-p 0.9, cf. Heddle temp=1/top-p=.9)
SAMPLING = SamplingParams(max_tokens=MAX_TOK, temperature=0.7, top_p=0.9)


def _load_gsm8k_questions():
    """Distinct natural-text prompts from gsm8k train (offline HF cache).
    Reads the cached parquet directly (datasets.load_dataset would still try
    to fetch the loading script gsm8k.py from the hub when only data is
    cached; pyarrow avoids any network).
    """
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
    """in=512 prompt is a prefix of in=2048 prompt (same content, two lengths);
    b sequences in one call replicate the same prompt (comparable pooling)."""
    tok = llm.get_tokenizer()
    src = _load_gsm8k_questions()
    if src is not None:
        ids = tok(" ".join(src[:200]))["input_ids"][:n]
        if len(ids) < n:
            ids = (ids * (n // len(ids) + 1))[:n]
        return ids
    import random

    rng = random.Random(seed)  # seeded per combo, reproducible fallback
    vocab = tok.vocab_size
    return [rng.randint(4, vocab - 1) for _ in range(n)]


def main():
    mode, il, batch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    sc = None if mode == "ar" else SPEC_CONFIG
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=GPU_MEM,
        max_model_len=4096,
        speculative_config=sc,
        trust_remote_code=True,
        disable_log_stats=False,  # offline LLM defaults to True; needed for
        # the SpecDecoding per-position acceptance flush (metrics.py)
        enforce_eager=os.environ.get("CANDC_EAGER") == "1",
    )
    pids = prompt_ids(llm, il, seed=il)
    reqs = [{"prompt_token_ids": pids} for _ in range(batch)]
    t0 = time.monotonic()
    outs = llm.generate(reqs, SAMPLING)
    dt = time.monotonic() - t0
    # Force a final stats flush so the last "SpecDecoding metrics:" stderr
    # line aggregates this whole run (incl. per-position acceptance A(p)).
    try:
        llm.llm_engine.do_log_stats()
    except Exception as exc:  # flush is best-effort
        print("WARN do_log_stats failed: %r" % (exc,), flush=True)
    n_ok = sum(1 for o in outs if len(o.outputs[0].token_ids) == MAX_TOK)
    row = dict(
        mode=mode,
        input_len=il,
        batch=batch,
        max_tokens=MAX_TOK,
        wall_s=round(dt, 4),
        decode_tok_per_s=round(batch * MAX_TOK / dt, 1),
        finished_full=n_ok,
    )
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
