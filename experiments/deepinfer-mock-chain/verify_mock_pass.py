#!/usr/bin/env python3
"""Independent PASS verification of the DeepInfer mock weight chain (4090-1, SM89).

Runs the same recipe the parallel M5 instance used (deepinfer-stubs +
deepinfer-mock with vocab_size=130000): LLMEngine init with load_format=dummy
(random-init weights), CUDA graph capture warmup, prefill + decode, then a
small generate() with token ids < vocab_size. Output tokens are meaningless
(random weights); we only verify the chain executes end to end.
"""
import os
import sys
import traceback

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("DEEPINFER_USE_NEXTN_DECODE", "0")
os.environ.setdefault("DEEPINFER_USE_MULTIMODAL_POOL", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MASTER_PORT", "24446")

from deepinfer.entrypoints.llm import LLM  # noqa: E402
from deepinfer.sampling_params import SamplingParams  # noqa: E402


def main():
    print("=== [1] LLMEngine init (load_format=dummy) ===", flush=True)
    llm = LLM(
        model=os.environ.get("MOCK_DIR", "/home/zxw/deepinfer-mock"),
        tokenizer=os.environ.get("MOCK_DIR", "/home/zxw/deepinfer-mock"),
        load_format="dummy",
        max_model_len=256,
        max_num_batched_tokens=2048,
        max_num_seqs=16,
        gpu_memory_utilization=0.4,
    )
    print("=== [2] init OK (incl. warmup graph capture); generate() ===", flush=True)
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    outs = llm.generate(
        prompt_token_ids=[np.asarray([5, 7, 9, 11], dtype=np.int32)],
        sampling_params=sp,
        use_tqdm=False,
    )
    print(f"=== [3] outputs={len(outs)} ===", flush=True)
    for i, out in enumerate(outs):
        print(f"[sample {i}] tokens={len(out)} text={out.text!r}", flush=True)
        print(f"[sample {i}] token_ids={[t.token_id for t in out]}", flush=True)
    print("=== [4] chain PASS ===", flush=True)
    llm.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
