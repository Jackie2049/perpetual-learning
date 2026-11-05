# FastVideo 扩池第八批：65 个开关 / 8 条候选，逐条查证后真缺陷 0 条

对 [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo) `36dc971d`（2026-11-04，
[#2001](https://github.com/hao-ai-lab/FastVideo/pull/2001)）跑一遍 pyflag。

前七批：SGLang / vLLM / LMDeploy / TensorRT-LLM / llama.cpp（C++）/ MLC-LLM / AReaL（训练侧）。
本批换**视频生成**——前六批都在做 LLM 推理、第七批在 RL 训练，这是生态里另一个大块，
而且形状不同：一张 `FastVideoArgs` 配置面 + 一堆各 backend 的 impl 类。

- 证据页：[open-source/fastvideo/evidence/pyflag-2026-11.md](../../open-source/fastvideo/evidence/pyflag-2026-11.md)
- 扫描归档：[scan-fastvideo-36dc971d.json](../../tools/pyflag/scan-fastvideo-36dc971d.json)
- 上游状态：**不对外**（真缺陷 0 条）

## 读数

| 项 | 值 |
|---|---|
| 快照 | `36dc971dc6a87dd1d40a97204908e7b524d40310`（2026-11-04 17:24 -0800，`git status` 干净） |
| `.py` 文件 | 2118（含独立的 `fastvideo-kernel/` 子工程） |
| 发现开关 | **65** |
| `silently_inert` | **8** |
| 真缺陷 | **0** |

## 8 条候选的归因

| 开关 | `mode` | 档位 | 事实（行号对 `36dc971d`） |
|---|---|---|---|
| `use_bf16` | always | **R3 内部默认** | `attention/backends/sla.py:202`／`:404` 两处形参，`:217`／`:423` 直接 `self.dtype = torch.bfloat16 if use_bf16 else torch.float16`，`:303-319`／`:506` 真用它做 cast 与 autocast。构造走 `attention/layer.py:83-90` 的通用 `**extra_impl_args`，**仓库内调用方一处都不传** ⇒ 恒 `True`。bf16 是 fp16 之上的常规选择，属设计意图 |
| `use_128x4_sf_layout` | always | **R3 内部默认** | `layers/fp4linear.py:36` 形参，`:58` 与 `:68` 真按它选 `SfLayout.layout_128x4` / `layout_8x4`。三处调用点（`fp4linear.py:113`、`quantization/nvfp4_qat_train_config.py:50`、测试 `tests/ops/quantization/test_nvfp4_spark_fence.py:89/:94`）全部**显式传 `True`** ⇒ 值不是死的，只是仓库里没人传 `False` |
| `use_queue_mode` | always | **R3 内部默认** | `entrypoints/streaming_generator.py:81` 形参，`:82-86` 真分支（给 `FastVideoArgs` 开 IPC 队列 / 先拷贝 args）。生产构造走 `:104-110` 的 `from_fastvideo_args` 不传 ⇒ 恒 `True`；唯一的 `False` 生产者是测试 `tests/worker/test_multiproc_executor.py:431-445` 的参数化 |
| `use_nvfp4` | mixed | **R3 内部默认（转发型）** | `layers/quantization/nvfp4_config.py:306` 的 `_mm_fp4_op` 形参，`:325` 原样转给 `mm_fp4`；调用端在 `:416` 用 `kwargs.pop("use_nvfp4", True)` 取默认、`:424` 传下去 ⇒ 仓库内**有生产者但值恒为默认** |
| `use_high_prec_o` | never | **R5 子工程内部** | `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_train.py:1230` 的 `forward` |
| `use_qat_qkv_backward` | always | **R5 同上** | 同一个 `forward`（`:1230`），默认 `True`，子工程内部调用方不传 |
| `enable_smoothing_k` | never | **R5 同上** | `fastvideo-kernel/attn_qat_infer/api.py:88` 的 `preprocess_qkv` |
| `enable_smoothing_q` | never | **R5 同上** | 同上（同一个函数的第二个参数） |

档位分布 **R3 4 / R5 4**（`use_nvfp4` 按"有生产者、值恒默认"归 R3，不是 R2 环境探测）。
`classify.py` 全覆盖，漏一条 `exit 1`；`verify_findings.py` 33 条判据钉到具体行，`--selftest` 3 条注入。

## 本批的形状观察（换池的价值在这里）

1. **"一堆 backend impl 类"的框架里，形参默认值高发、但绝大多数是内部默认。** 8 条里 4 条落在
   `AttentionImpl.__init__` / `StreamingVideoGenerator.__init__` / `_mm_fp4_op` 这种"工厂按后端
   实例化、调用方只传公共参数"的形状上：接收端有 `**extra_impl_args` / `**kwargs`，发送端不填。
   这是第七类易错点的镜像（`**kwargs` 兜底的接收侧），**不该按"失效"处理**。
2. **多包仓库要按包分组再归因。** `fastvideo-kernel/` 是独立子工程（自带 pyproject 与调用约定），
   4 条候选全落在那里；与主包混在一起看，"零生产者"会把子工程内部的 partial／包装算漏。
3. **`always` 不等于"没人传"。** `use_128x4_sf_layout` 三个调用点都写死 `True`——生产者存在、
   只是值不变。读数里 `mode=always` 回答的是"值会不会变"，与 `producers_total` 回答的不是同一件事。

## 脚本

| 文件 | 作用 | 怎么跑 |
|---|---|---|
| `classify.py` | 8 条候选逐条给档位与理由，漏一条 `exit 1` | `python3 classify.py raw/scan-fastvideo.json` |
| `verify_findings.py` | 33 条判据钉到文件行；`--selftest` 3 条注入 | `python3 verify_findings.py <FastVideo 根>` |

## 复现

```bash
git clone --filter=blob:none https://github.com/hao-ai-lab/FastVideo /tmp/fastvideo
git -C /tmp/fastvideo checkout 36dc971dc6a87dd1d40a97204908e7b524d40310
cd <repo>
python3 tools/pyflag/scan_unwired_flags.py /tmp/fastvideo --json tools/pyflag/scan-fastvideo-36dc971d.json
python3 experiments/fastvideo-pyflag-triage/classify.py experiments/fastvideo-pyflag-triage/raw/scan-fastvideo.json
python3 experiments/fastvideo-pyflag-triage/verify_findings.py /tmp/fastvideo
```

## 边界

纯静态 AST + 全仓扫描，**未装 fastvideo、未跑推理、未上 GPU/NPU、未触上游**。
"内部默认"只说明"仓库内没人改它"，不说明"这个默认值一定合适"。
