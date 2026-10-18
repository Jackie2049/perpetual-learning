# Candidate: Fused Megatron LM-head 遗留 AllGather pending (#8028)

> Issue: [verl-project/verl#8028](https://github.com/verl-project/verl/issues/8028)
> 创建 2026-09-27；0 评论；查重 2026-10-19：0 个引用 PR（无人认领）
> 状态：**analysis-done + patch applied**（DDP overlap 语义分析完成，patch 已应用本地 fork；e2e 验证需 GPU + Megatron DDP overlap 环境）

## 0. 为什么值得做

| 维度 | 评估 |
|---|---|
| I | 4 — `ddp.overlap_param_gather=True` + fused LM-head 时第二个 batch 必挂（AssertionError，无可用 workaround 组合说明），SFT/训练主路径 |
| C | 2 — 修复涉及 DDP 内部 bucket 语义，验证需 Megatron + 多卡 + overlap 配置（重） |
| A | 4 — 明确 bugfix、0 竞争、reporter 给出完整机制链 |
| F | 4 — Megatron DDP/参数通信是我们的分布式深耕区 |
| 分 | **21** |

## 1. Bug

`use_fused_kernels=True` + `ddp.overlap_param_gather=True`：第一个 batch 正常，**第二个 batch 在 decoder prefetch 时参数 gather 断言失败**：

```text
megatron/core/distributed/param_and_grad_buffer.py, start_param_sync
    assert self.param_gather_handle is None
AssertionError
```

## 2. 根因机制（源码确认，本地 Megatron-LM + verl）

1. DDP overlap 模式下（distributed_data_parallel.py:378-393），每个模块注册 forward pre-hook：对模块直属参数调 `bucket_group.finish_param_sync(...)`（:431-447），完成 gather 并 dispatch 下一 bucket（prefetch 链）。
2. verl 的 fused forward（`_fused_GPTModel_forward`，model_forward_fused.py:301-308）**直接读 `model.output_layer.weight`** 传给 `linear_cross_entropy`，跳过 `output_layer.forward` → 跳过其 pre-hook → output-weight bucket 的 gather handle 滞留（`param_gather_dispatched=True`）。
3. batch 边界 `finish_grad_sync` 只重置 flag（False），**不清 handle**；下一 batch decoder prefetch 再次 dispatch 该 bucket → `start_param_sync` 的 `assert param_gather_handle is None` 失败。
4. `patch_fused_forward`（:52）在 patch 时持有 DDP 包装（引擎 `self.module` 是 DDP-wrapped chunks），但 `_get_patching_model` unwrap 后**丢失 DDP 引用**——这是可修复的接入点。

## 3. Patch（已应用本地 fork，model_forward_fused.py）

**① patch 时保存 DDP 引用**（duck-typed，不依赖 MCore 版本）：

```python
ddp = model if hasattr(model, "param_to_bucket_group") else None
model = _get_patching_model(model)
if model is not None:
    if ddp is not None:
        model._verl_ddp = ddp
    ...
```

**② fused forward 消费权重前完成 pending gather**（镜像 pre-hook 语义）：

```python
ddp = getattr(model, "_verl_ddp", None)
if ddp is not None and ddp.ddp_config.overlap_param_gather:
    bucket_group = ddp.param_to_bucket_group.get(output_weight)
    if bucket_group is not None and bucket_group.param_gather_dispatched:
        bucket_group.finish_param_sync(
            skip_next_bucket_dispatch=(
                ddp.ddp_config.align_param_gather or ddp.overlap_param_gather_with_optimizer_step
            )
        )
```

**设计论证**：
- `param_gather_dispatched` 守卫：只完成**确有 pending** 的 gather（bug 场景：decoder prefetch 已 dispatch output bucket）；tied-embedding 情形（权重=word_embeddings，早已被 embedding hook 完成）不触发 → 无多余 gather。
- `finish_param_sync` 签名（param_and_grad_buffer.py:494）：assert overlap on；未 dispatch 时自己补 dispatch——守卫保证只在已 dispatch 时调用，不会重复 dispatch。
- `skip_next_bucket_dispatch` 镜像 pre-hook 的 `align_param_gather or overlap_param_gather_with_optimizer_step`（:447）。
- 非 overlap / 无 DDP（推理、评估）：`_verl_ddp` 为 None 或配置关 → 完全跳过，行为不变。

## 4. 验证

- 代码级：语法通过；守卫逻辑覆盖 4 种场景（overlap+dispatched=修复目标 / tied-embedding=跳过 / 非 overlap=跳过 / 无 DDP=跳过）。
- GPU 级（待做）：`overlap_param_gather=True` + fused kernels 跑 ≥2 batch（reporter 配置：TP1/PP1/CP4/EP8 太重，可用单机小规模等价组合：TP1/PP1/CP1 + overlap 参数），确认第二个 batch 不再断言；再跑 overlap=False 回归。

## 5. 备注

- `model_forward_1f1b_overlap.py:194` 有同样的直读 `self.output_layer.weight` 模式，但该文件在当前代码中未被引用（实验路径），未纳入本次 patch；PR 中可提及作为 follow-up。
- legacy fused forward（issue 提到）即 model_forward_fused 的历史版本，本次修复覆盖。
