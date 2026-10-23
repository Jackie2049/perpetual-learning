# DeepInfer mock 权重链路（2026-10-23 M5 跑通）

目标（用户 2026-10-23 明确）：demo 级（能正常推理即可），先 mock 权重跑通链路，
调通后再上真实权重调精度。真实权重环节（约 300GB）不归本条目。

## 验证结果

- LLMEngine 初始化 + CUDA graph 捕获 + prefill/decode 全链路 **PASS**
- 环境：conda `deepinfer-smoke` + PYTHONPATH=`deepinfer-stubs` + `deepinfer-civilian`(774ce24)
- 运行命令（GPU 0，服务器）：

  ```bash
  CUDA_VISIBLE_DEVICES=0 DEEPINFER_USE_NEXTN_DECODE=0 DEEPINFER_USE_MULTIMODAL_POOL=0 \
  MASTER_PORT=23555 PYTHONPATH=/home/zxw/deepinfer-stubs:/home/zxw/deepinfer-civilian \
  python - <<'PY'
  import numpy as np
  from deepinfer.entrypoints.llm import LLM
  from deepinfer.sampling_params import SamplingParams
  llm = LLM(model='/home/zxw/deepinfer-mock', tokenizer='/home/zxw/deepinfer-mock',
            load_format='dummy', max_model_len=256, max_num_batched_tokens=2048,
            max_num_seqs=16, gpu_memory_utilization=0.4)
  sp = SamplingParams(max_tokens=8, temperature=0.0)
  outs = llm.generate(prompt_token_ids=[np.asarray([5,7,9,11], dtype=np.int32)],
                      sampling_params=sp, use_tqdm=False)
  print(outs[0])  # 随机文本（mock 权重无意义）
  PY
  ```

- 输出：随机文本（mock 权重无意义，预期行为）；显存 0.79 GiB allocated / 1.32 GiB reserved
- 输入需用 `< vocab_size(1000)` 的 token id（真实 tokenizer 词表 129280 超出 mock vocab）

## 环境改动（相对 774ce24 基线，服务器）

1. **tilelang_ops/cuda/*.py 28 个文件覆盖为 nosync 变体**（stage2 已验证的「删除 pdl_sync」改写），
   原文件备份在 `/home/zxw/deepinfer-cuda-orig/`；变体生成材料备份在 `/home/zxw/tilelang-variants-backup/`
2. **deepseek.py**：dummy 分支 sf dtype 改为 `PLATFORM.fp8_sf_type`(int32)（与真实路径一致）；
   跳过 `process_moe_weight` 的 FP4 打包（其 per_token_cast packed-ABI 与极小尺寸不兼容，
   真实加载路径在 `materialize_hf_parameter` 内部分包）
3. **runtime_config.py**：`set_pdl` 按 compute capability 门控（SM89 关 PDL）
4. **deep_cluster/hardware/device.py**：`get_bf16_tflops_limit` 加 SM89 映射（stage2 已有）
5. **deepinfer-stubs/**：deep_gemm/deep_ep/flash_mla/deep_select/flash_attn 提供 torch 回退
   （含 MLA 稀疏注意力、mega_moe、topk 等 mock 语义，CUDA graph 内避免同步操作）

## mock 模型目录（/home/zxw/deepinfer-mock/）

- `config.json`：伪造 V4.1 极小尺寸（n_layer=2, hidden=128, heads=8, head_dim=512,
  q_lora_rank=128, o_lora_rank=128, kv_lora_rank=448, moe_intermediate=256,
  n_routed=1, nextn=0, engram 关, candidate 关）；KV 布局常量硬编码要求 head_dim=512
- `tokenizer.json` / `tokenizer_config.json`：真实 V4.1（hf-mirror 拉取）
