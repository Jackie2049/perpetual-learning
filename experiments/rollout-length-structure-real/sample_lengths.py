#!/usr/bin/env python3
"""真实 rollout 的响应长度结构实测（rollout 专项 §13.38 残留，M5 算力侧）。

**为什么有本文件**：§13.38 的「真实长度结构」是构造出来的——`realistic = (cv_group 0.60,
cv_in 0.08)`，含义是"同 prompt 的 n 条响应长度强相关、组内几乎齐平"。这个参数直接决定
该节两个读数（`ldm_over_grouped` 与 `interleaved_over_ldm`）的落点，但**从未在真实模型上量过**。
本脚本用与 RL 训练完全一致的采样参数生成真实 group，供 analyze_lengths.py 离线统计。

**采样口径对齐 verl**（`verl/trainer/config/rollout/rollout.yaml` 与 `vllm_async_server.py`）：
temperature=1.0、top_p=1.0、top_k=-1、max_tokens=max_response_length=256、
贪心关闭、不强制 ignore_eos。prompt 用与 verl 相同的 chat template 渲染。

用法（4090-1）：
    CUDA_VISIBLE_DEVICES=1 /home/zxw/venvs/rollout-g8/bin/python sample_lengths.py \
        --model /home/zxw/models/Qwen2.5-1.5B-Instruct \
        --data /home/zxw/data/gsm8k/train_sub480.parquet \
        --out raw/qwen15b_gsm8k_n8.jsonl.gz --meta raw/qwen15b_gsm8k_n8.meta.json
"""

import argparse
import gzip
import hashlib
import json
import os
import platform
import time


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            h.update(buf)
            if limit and f.tell() >= limit:
                break
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--meta', default=None)
    ap.add_argument('--n-samples', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--max-prompt-tokens', type=int, default=512)
    ap.add_argument('--max-model-len', type=int, default=2048)
    ap.add_argument('--gpu-mem', type=float, default=0.45)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top-p', type=float, default=1.0)
    ap.add_argument('--top-k', type=int, default=-1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 个 prompt（0=全部）')
    args = ap.parse_args()

    import pandas as pd
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.time()
    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts, rows = [], []
    for i, row in df.iterrows():
        msgs = [{'role': str(m['role']), 'content': str(m['content'])} for m in row['prompt']]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        n_prompt = len(tok(text, add_special_tokens=False)['input_ids'])
        prompts.append(text)
        rows.append({'prompt_idx': int(i), 'prompt_tokens': int(n_prompt),
                     'prompt_tail': msgs[-1]['content'][:200]})
    too_long = [r['prompt_idx'] for r in rows if r['prompt_tokens'] > args.max_prompt_tokens]
    if too_long:
        print('WARN prompts over %d tokens: %s' % (args.max_prompt_tokens, too_long[:10]))
    keep = [r['prompt_idx'] for r in rows if r['prompt_tokens'] <= args.max_prompt_tokens]
    rows = [r for r in rows if r['prompt_tokens'] <= args.max_prompt_tokens]
    prompts = [prompts[i] for i in keep]

    llm = LLM(model=args.model, dtype='bfloat16', max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, trust_remote_code=True)
    sp = SamplingParams(n=args.n_samples, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_tokens, seed=args.seed)
    gen_t0 = time.time()
    outputs = llm.generate(prompts, sp)
    gen_sec = time.time() - gen_t0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_rec = 0
    with gzip.open(args.out, 'wt', encoding='utf-8') as f:
        for r, out in zip(rows, outputs):
            for j, o in enumerate(out.outputs):
                rec = dict(r)
                rec.update({
                    'sample_idx': j,
                    'response_tokens': len(o.token_ids),
                    'finish_reason': o.finish_reason,
                    'truncated': len(o.token_ids) >= args.max_tokens,
                    'cum_logprob': float(o.cumulative_logprob) if o.cumulative_logprob is not None else None,
                    'text': o.text,
                })
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                n_rec += 1

    meta = {
        'script': os.path.abspath(__file__),
        'script_sha256': sha256_file(os.path.abspath(__file__)),
        'argv': vars(args),
        'model': args.model,
        'model_files': {p: {'size': os.path.getsize(os.path.join(args.model, p)),
                            'mtime': int(os.path.getmtime(os.path.join(args.model, p)))}
                        for p in sorted(os.listdir(args.model)) if p.endswith(('.safetensors', '.json'))},
        'data': args.data,
        'data_sha256': sha256_file(args.data),
        'vllm': vllm.__version__,
        'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'python': platform.python_version(),
        'host': platform.node(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'n_prompts': len(prompts),
        'n_records': n_rec,
        'gen_seconds': round(gen_sec, 1),
        'wall_seconds': round(time.time() - t0, 1),
        'started_unix': int(t0),
    }
    meta_path = args.meta or (args.out + '.meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: meta[k] for k in ('n_prompts', 'n_records', 'gen_seconds', 'vllm', 'torch')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
