#!/usr/bin/env python
"""Candidate D single-combo: batch-internal length distribution -> makespan.

Usage: run-candidate-d.py <mode> <comp> <iter>
  mode : ar | spec
  comp : H512 | H2048 | MIX
  iter : 1 | 2     (repetition id; sampling variance note, batch 29/31)

Same stack as candidates C/C2/DSD (vLLM 0.24.0+cu129, Qwen2.5-1.5B-Instruct,
single 4090, gsm8k-derived prompts, temp 0.7/top-p 0.9, max_tokens 128,
batch 8). Fills the G3 memo axis-3 open item "批内轨迹分布无数据": every
prior engine-level measurement used fixed-length x batch (homogeneous
batches). Here each request gets a DISTINCT prompt of a target length:

  H512 : 8 x 512
  H2048: 8 x 2048                       (homogeneous baseline, mean 2048)
  MIX  : [512,512,1024,2048,2048,2048,4096,4096]  (mean 2048, spread 512..4096)
  RX8  : 8 x 2048 with max_tokens [64,64,128,128,128,128,256,256] (mean 128)
         -- response-length heterogeneity: short trajectories finish early,
         active batch shrinks (EfficientRollout claim tested at engine level)

wall_s = batch makespan (all requests finished, incl. prefill). Per-request
e2e latencies (RequestOutput.metrics, best effort) give the T_i distribution:
sync-RL makespan is driven by the upper quantile of T_i, not the mean (G3
memo axis 3); the ratio wall_s / mean(T_i) is an engine-level "bubble
factor" proxy. Spec mode additionally reports the engine's A(p) curve
aggregated over the heterogeneous batch (does prompt-length heterogeneity
shift acceptance?).

Anchor: vllm 0.24.0+cu129 wheel (anchor-drift 0.24.0), ngram proposer k=5.
NOTE: max_model_len=4608 here (MIX contains in=4096 + max_tokens=128);
candidates C/B used 4096 -- within-candidate-D comparisons are consistent,
cross-candidate decode rates are not directly comparable (single points).
"""
import json
import os
import sys
import time

from vllm import LLM, SamplingParams

MODEL = "/home/zxw/models/Qwen2.5-1.5B-Instruct"
MAX_TOK = 128
GPU_MEM = float(os.environ.get("CANDD_GPU_MEM", "0.40"))
MAX_MODEL_LEN = int(os.environ.get("CANDD_MAX_LEN", "4608"))
SPEC_CONFIG = dict(
    method="ngram",
    num_speculative_tokens=5,
    prompt_lookup_max=5,
    prompt_lookup_min=1,
)
SAMPLING = SamplingParams(max_tokens=MAX_TOK, temperature=0.7, top_p=0.9)

# batch = 8 for every composition; H2048 and MIX sum to 16384 input tokens
# (identical total work, identical mean 2048) -- isolates the
# upper-quantile/straggler effect from the amount of work. RX8 keeps the
# input side fixed (8 x 2048) and varies the response budget instead.
COMPS = {
    "H512": ([512] * 8, [128] * 8),
    "H2048": ([2048] * 8, [128] * 8),
    "MIX": ([512, 512, 1024, 2048, 2048, 2048, 4096, 4096], [128] * 8),
    "RX8": ([2048] * 8, [64, 64, 128, 128, 128, 128, 256, 256]),
}


def _load_gsm8k_questions():
    """All gsm8k train questions from the offline HF cache (no network)."""
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


def prompt_windows(llm, lengths, seed):
    """Distinct natural-text prompt windows, one per request.

    Evenly spaced offsets into the concatenated gsm8k train token stream,
    so requests are different questions/contexts (a batch of distinct
    trajectories), unlike candidates C/DSD which replicated one prompt.
    """
    tok = llm.get_tokenizer()
    src = _load_gsm8k_questions()
    if src is not None:
        ids = tok(" ".join(src))["input_ids"]
        total = len(ids)
        step = total // len(lengths)
        out = []
        for k, ln in enumerate(lengths):
            start = (k * step) % max(total - ln, 1)
            win = ids[start:start + ln]
            if len(win) < ln:
                win = (win * (ln // len(win) + 1))[:ln]
            out.append(win)
        return out
    import random

    rng = random.Random(seed)
    vocab = tok.vocab_size
    return [[rng.randint(4, vocab - 1) for _ in range(ln)] for ln in lengths]


def _e2e(o):
    """Best-effort per-request e2e seconds from RequestOutput.metrics."""
    m = getattr(o, "metrics", None)
    if m is None:
        return None
    v = getattr(m, "e2e_latency", None)
    if v is not None:
        return v
    ft = getattr(m, "finished_time", None)
    at = getattr(m, "arrival_time", None)
    if ft is not None and at is not None:
        return ft - at
    return None


def _first(o):
    m = getattr(o, "metrics", None)
    if m is None:
        return None
    v = getattr(m, "first_token_latency", None)
    if v is not None:
        return v
    ft = getattr(m, "first_token_time", None)
    at = getattr(m, "arrival_time", None)
    if ft is not None and at is not None:
        return ft - at
    return None


def main():
    mode, comp, it = sys.argv[1], sys.argv[2], sys.argv[3]
    lengths, max_toks = COMPS[comp]
    sc = None if mode == "ar" else SPEC_CONFIG
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=GPU_MEM,
        max_model_len=MAX_MODEL_LEN,
        speculative_config=sc,
        trust_remote_code=True,
        disable_log_stats=False,  # needed for the A(p) stats flush (metrics.py)
        enforce_eager=os.environ.get("CANDD_EAGER") == "1",
    )
    pids_list = prompt_windows(llm, lengths, seed=len(lengths))
    if comp == "RX8":
        # Per-request response budget (per-request sampling params).
        reqs = [{"prompt_token_ids": p,
                 "sampling_params": SamplingParams(
                     max_tokens=mt, temperature=0.7, top_p=0.9)}
                for p, mt in zip(pids_list, max_toks)]
    else:
        reqs = [{"prompt_token_ids": p} for p in pids_list]
    t0 = time.monotonic()
    outs = llm.generate(reqs, SAMPLING)
    dt = time.monotonic() - t0
    try:
        llm.llm_engine.do_log_stats()
    except Exception as exc:
        print("WARN do_log_stats failed: %r" % (exc,), flush=True)
    n_ok = sum(1 for o, mt in zip(outs, max_toks)
               if len(o.outputs[0].token_ids) == mt)
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    e2e = [round(x, 4) for x in (_e2e(o) for o in outs) if x is not None]
    first = [round(x, 4) for x in (_first(o) for o in outs) if x is not None]
    row = dict(
        mode=mode,
        comp=comp,
        iter=it,
        batch=8,
        max_tokens=max_toks,
        input_lengths=lengths,
        wall_s=round(dt, 4),
        decode_tok_per_s=round(gen_tokens / dt, 1),
        gen_tokens=gen_tokens,
        finished_full=n_ok,
        e2e_s=e2e,
        first_s=first,
    )
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
