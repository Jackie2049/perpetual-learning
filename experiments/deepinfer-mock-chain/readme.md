# DeepInfer mock 权重链路跑通（M5 定时任务，2026-10-23）

状态：**✅ 已完成（完整链路 PASS）**——LLMEngine 初始化（含 CUDA graph 捕获）+ prefill/decode 全链路真实执行，`generate` 返回随机文本（mock 权重无意义，预期行为），显存 0.79 GiB allocated / 1.32 GiB reserved，8 token 生成约 0.21s。**最后一次阻塞（warmup graph 捕获的 `convert_kv_index_tl` CUDA_ERROR_ASSERT）已由本会话解决**：并行会话报告时用的是 `deepinfer-stubs-m5`/`deepinfer-mock-m5` 隔离目录，本会话在共享目录 `deepinfer-stubs`/`deepinfer-mock`（最终工作树）上复跑并通过——关键差异是 `DEEPINFER_USE_NEXTN_DECODE=0` 与输入改用 `< vocab_size` 的 token id。最终复现命令与证据见下方「最终复现」。服务器存档 `/home/zxw/deepinfer-mock/README-mock.md`（含环境改动清单与 mock 模型目录说明）。

## 任务来源

用户 2026-10-23 明确 DeepInfer demo 级目标（能正常推理即可）：先 mock 权重跑通链路、再上真实权重调精度；除真实权重下载环节外授权 M5 定时任务推进闭环。交接见 [handoff-m5.md](../../experiments/handoff-m5.md#deepinfer-mock-权重链路跑通2026-10-23-用户授权推进除真实权重环节)。

## 目标机器与环境（4090-1）

- SSH: `4090-1`（219.223.198.62，用户 zxw）
- GPU 0/1/7 空闲（1 MiB / 0%）；GPU 2-6 被他人 VLLM 占用
- conda env `deepinfer-smoke`：Python 3.12.15、torch 2.13.0+cu129、tilelang 0.1.15、tile_kernels（site-packages 版本）
- DeepInfer 源码：`/home/zxw/deepinfer-civilian`（HEAD `774ce24`，全子模块）
- `_C` 二进制已由 stage2 用 gcc13+cu129 编译（`deepinfer/_so/nvidia-x86_64/`），导入需 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib`
- 运行需 `PYTHONPATH=/home/zxw/deepinfer-civilian:/home/zxw/<stubs>`

## 已验证通过的结构性环节

1. **PDL 门控（SM89）**：`configure_worker_kernels` 按 `torch.cuda.get_device_capability(0) >= (9, 0)` 门控 `deep_gemm.set_pdl`/`tile_kernels.set_pdl`（tile_kernels 有真实 PDL 开关，默认 False；kernel 工厂在 `use_pdl=False` 时不发 `pdl_sync/pdl_trigger`）。参考 patch 在 [server-patches/civilian-source-patches.diff](server-patches/civilian-source-patches.diff)。
2. **tilelang_ops 28 个 cuda kernel 应用 nosync 变体**（stage2 产出，删除 pdl_sync / pdl_trigger→pass；工作树 `deepinfer/model_executor/layers/tilelang_ops/cuda/*.py` 已替换，原文件备份 `/home/zxw/tilelang-variants-backup/working-cuda-orig/`）。
3. **单 rank NCCL 健康检查跳过**：`init_device_groups` 对 size-1 组跳过 all_reduce（worker 子进程内 size-1 NCCL all_reduce 报 `cudaErrorInvalidValue`，最小复现无法触发；world_size=1 + engram 关闭时该检查无必要）。此 patch 也解决了并行会话遇到的同一错误。
4. **多模态池必须关闭**：`DEEPINFER_USE_MULTIMODAL_POOL=0`（8GB pinned 池 + 子进程干扰 worker 子进程 CUDA 上下文，导致模型构建期 CUDA invalid argument）。
5. **mock config 字段约束**（逐项按 `load_hf_model_config`/`_assert_supported`/builder 校验补齐，见 [mock-config/config.json](mock-config/config.json)）：
   - `model_type=deepseek_v41` + `architectures=[DeepseekV41ForCausalLM]` + fp8/fp4 quantization（强制项）
   - `hidden_act=silu`、`attention_bias=false`、`attention_dropout=0`、`num_key_value_heads=1`、`dtype=bfloat16`（`_assert_supported`）
   - `qk_rope_head_dim=64`（MQAIndexer `rope_dim==64` 断言）
   - `q_lora_rank` 为 128 的倍数（fp8 权重 merge 的 `%128==0` 断言）
   - `index_head_dim=128`（sf pack 的 `groups % itemsize == 0`，64 会失败）
   - `n_shared_experts=0`（绕开 shared 专家融合路径）
   - 极小尺寸：`num_hidden_layers=2`、`hidden_size=128`、`head_dim=512`（并行会话调大以适配 kernel 约束）、`vocab_size=1000`、`o_lora_rank=128`
6. **stub 设计**（`/home/zxw/deepinfer-stubs-m5/`，本会话工作副本；并行会话的 `/home/zxw/deepinfer-stubs/` 是另一份演进）：
   - `deep_gemm`：torch fallback——bf16/fp8 gemm（NT 布局）、`tf32_hc_prenorm_gemm`（split-K prenorm + sqr_sum）、`fp8_einsum`、mega_moe（gate + swiglu + down，向量化、CUDA-graph-safe）、mqa_logits 系（head 加权 p_sum，形状鲁棒）、paged_mqa（字节级解码 gathered blocks，避免全缓存物化）、`SymmBuffer`（x/x_sf/topk_idx/topk_weights 真实 CUDA 张量视图，x_sf 为 int32 packed ue8m0 布局）、`set_pdl`/`use_deterministic_algorithms` no-op
   - `deep_ep`：导入存活 stub（EP=1 单卡下前向不触达）
   - `deep_select`：`topk` torch fallback（end 掩码、offset、OOB -1 填充、索引钳制到 end 内；始终返回 (values, indices) 元组——调用方 `topk(...)[1]`）
   - `flash_mla`：`flash_mla_sparse_fwd` / `flash_mla_with_kvcache`（gather + 缩放点积 + sink + softmax + lse，逐 head 循环控显存，CUDA-graph-safe）
7. **运行时约束**：`DEEPINFER_USE_NEXTN_DECODE=0`、`MASTER_PORT` 自定义（避免与并行会话默认端口 23283 冲突）、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 当前阻塞（最后一个）

~~warmup 阶段 `capture_model`（CUDA graph 捕获）中，真实 tilelang kernel `convert_kv_index_tl`（`tilelang_ops/cuda/flash_mla_utils.py`）消费 mock fallback 产出的 topk 索引时触发 `CUDA_ERROR_ASSERT`（sticky 越界）。索引单位错配：fallback 的 p_sum 宽度是 raw KV block 位置，而 kernel 按 compressed 索引映射 `block_tables[bid, x // block_size]`。已加 `end` 钳制仍复现，疑与 warmup 空序列/lens=0 的边界有关。~~ **已解决**：最终工作树（共享 `deepinfer-stubs`/`deepinfer-mock` + `DEEPINFER_USE_NEXTN_DECODE=0` + token id 输入 < 1000）通过 graph 捕获并完成真实 generate。

## 最终复现（2026-10-23 本会话 PASS）

```bash
ssh 4090-1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate deepinfer-smoke
cd /home/zxw/deepinfer-civilian
export PYTHONPATH=/home/zxw/deepinfer-civilian:/home/zxw/deepinfer-stubs
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 DEEPINFER_USE_NEXTN_DECODE=0 DEEPINFER_USE_MULTIMODAL_POOL=0 \
  MASTER_PORT=23555 python - <<'PY'
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

关键环境变量：
- `DEEPINFER_USE_NEXTN_DECODE=0`（默认 5 会要求 MTP 层数匹配，mock config `num_nextn_predict_layers=0`）
- `DEEPINFER_USE_MULTIMODAL_POOL=0`（多模态池导致 worker 子进程 CUDA 上下文异常，见上）
- `MASTER_PORT` 自定义（避开并行会话默认端口）
- 输入 token id 必须 `< vocab_size(1000)`（真实 tokenizer 词表 129280 超出 mock vocab，超限请求被引擎 drop）

## 独立复验（13:07 轮，文本提示路径）

另一条等价 PASS：mock config `text_config.vocab_size` 调至 **130000**（覆盖真实 tokenizer 词表 54994 与 image_token_id 129264），**直接用真实 tokenizer 的文本提示**跑通（无需手造 token id）。脚本 [run_mock_chain.py](run_mock_chain.py)（43 kernel nosync monkeypatch + `load_format=dummy` + `DEEPINFER_USE_NEXTN_DECODE=0`），完整日志 [mock_chain_run23.log](mock_chain_run23.log)：`LLMEngine constructed OK` → `GENERATED OUTPUT 'unch旅游景区 Ballet sama imaginationcheid谧-random'` → `MOCK CHAIN COMPLETED OK`（EXIT=0）。stub 差异点：`_dq`/`_dequant_v41` 加 `torch.nan_to_num`（随机 e4m3 权重的 NaN 码型会进 logits，触发 fused_sample「Invalid logits」设备断言——这是本会话卡住的最后一个 CUDA_ERROR_ASSERT 根因）；`mega_moe_compute` 向量化（einsum，兼容 [hidden, 2*inter] 转置权重布局）；`flash_mla_with_kvcache` 按 decode 调用方契约返回 `[M,1,H,Dv]`/`[M,H,1]`；`paged_mqa_logits` 先 gather 后解包（避免全缓存物化 OOM）。GPU 7 空闲运行，结束后回到 1 MiB。

## 并行会话说明

本任务触发后约 30 分钟另有 M5 定时实例同时在服务器上推进同一链路（创建 mock 目录、改写 stub、调整 config、给源码打同类补丁）。两实例共享 `/home/zxw/deepinfer-civilian` 工作树与 `/home/zxw/deepinfer-mock`，曾互相覆盖 stub/config；本会话改用隔离目录 `deepinfer-stubs-m5`/`deepinfer-mock-m5` 与自定义 `MASTER_PORT` 规避冲突。双方补丁同向（PDL 门控、dummy sf dtype、FP4 packing 跳过等），最终状态以服务器工作树 + 后续提交为准。

## 边界

- mock 输出无意义（随机初始化权重 + torch fallback 数值），仅验证「链路能跑」。
- 真实权重环节（约 300GB、磁盘剩 109G、hf-mirror 慢）不归本条目，待用户决策。
- 性能、吞吐、数值精度均不在验证范围。
