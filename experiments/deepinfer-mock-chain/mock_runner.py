"""DeepInfer mock weight chain runner (civilian port, SM89).

Goal: run LLMEngine init + prefill/decode chain with load_format=dummy
(random-init weights, no real checkpoint download). Output tokens are
meaningless; we only verify the chain executes end to end.
"""
import os
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("DEEPINFER_USE_NEXTN_DECODE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["DEEPINFER_LOGGING_INTERVAL_STEP"] = "5"

from deepinfer.entrypoints.llm import LLM
from deepinfer.sampling_params import SamplingParams

def main():
    print("=== [1] LLMEngine init with load_format=dummy ===", flush=True)
    llm = LLM(
        model="/home/zxw/deepinfer-mock-m5",
        trust_remote_code=True,
        load_format="dummy",
        seed=42,
        expert_parallel_size=1,
        context_parallel_size=1,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=2,
        gpu_memory_utilization=0.9,
        lookahead_tokens=0,
        disable_log_stats=False,
        tokenizer="/home/zxw/deepinfer-mock-m5",
    )
    print("=== [2] init OK; running prefill/decode ===", flush=True)
    params = SamplingParams(max_tokens=8, temperature=0.7, top_p=0.9)
    outputs = llm.generate(prompts=["Hello, this is a mock weight chain test."], sampling_params=params, use_tqdm=False)
    for i, out in enumerate(outputs):
        text = llm.llm_engine.tokenizer.decode([t.token_id for t in out])
        print(f"[sample {i}] tokens={len(out)} text={text!r} first_logp={out[0].token_logprobs if out else None}", flush=True)
    print("=== [3] chain OK ===", flush=True)
    llm.close()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
