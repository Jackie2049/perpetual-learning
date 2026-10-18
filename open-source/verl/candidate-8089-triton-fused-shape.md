# Candidate: Triton fused kernels 在 use_remove_padding=False 崩溃 (#8089)

> Issue: [verl-project/verl#8089](https://github.com/verl-project/verl/issues/8089)
> 创建 2026-10-01；0 评论；查重 2026-10-19：0 个引用 PR（无人认领）
> 状态：~~**analysis-done + patch applied**~~ → **2026-10-19 复查重纠正：reporter 次日已开 PR #8090（同类 reshape 方案）**。本地补丁已还原，不提交。

## 0. 为什么值得做

| 维度 | 评估 |
|---|---|
| I | 5 — 官方示例脚本直接崩（`use_fused_kernels=True` + triton + `use_remove_padding=False`），首步训练前 IndexError |
| C | 3 — 5 文件 × 2 行机械补丁，但 e2e 验证需 GPU（triton kernel） |
| A | 5 — 明确 bugfix、无人认领、reporter 给出期望行为（对齐 torch backend） |
| F | 3 — triton kernel 形状契约分析，中等契合 |
| 分 | **25** |

## 1. Bug

`use_fused_kernels=True` + `impl_backend=triton` + `use_remove_padding=False`：
- Triton backend 返回 `log_probs`/`entropy` 为扁平 `[B*S]`（1D）
- 引擎 `FSDPEngineWithLMHead.prepare_model_outputs`（transformer_impl.py:1244）对 dense 输出执行 `torch.arange(output.log_probs.shape[1])` → 1D tensor 上取 `shape[1]` → `IndexError: tuple index out of range`
- Torch backend（`FusedLinearForPPO`）返回 `[B,S]`（2D）→ 不崩

## 2. 根因（本地源码确认）

`LinearCrossEntropy.forward`（verl/utils/kernel/linear_cross_entropy.py:73）把 hidden 拍平 `[B,S,H] → [B*S,H]`，**forward 输出 logprobs/entropy 保持扁平 `[B*S]`**；`original_hidden_shape` 只在 backward 里恢复 d_hidden（backward 依赖扁平约定，不能在 kernel 层改返回形状）。

5 个 triton forward 包装（dense_common.py、qwen2_vl.py、qwen3_vl.py、glm4v.py、qwen3_5.py）把 `linear_cross_entropy` 输出**直接**塞进 `CausalLMOutputForPPO`，没有恢复 dense 形状 → 与 torch backend 的模型级契约 `[B,S]` 不一致。

rmpad 路径恰好工作：引擎 rmpad 分支 `output.log_probs.squeeze(0)` 对 `[1,total_nnz]`/`[total_nnz]` 都得到 `(total_nnz,)`——所以 bug 只在非 rmpad 暴露（形状分裂）。

## 3. Patch（已应用本地 fork 5 文件）

每个 triton forward 在 `linear_cross_entropy(...)` 后恢复 dense 形状：

```python
log_probs, entropy = linear_cross_entropy(...)
# linear_cross_entropy returns flattened (B*S,) outputs; restore the dense
# (B, S) shape to match the torch fused backend contract (issue #8089).
log_probs = log_probs.reshape(hidden_states.shape[:-1])
entropy = entropy.reshape(hidden_states.shape[:-1])
return <Model>CausalLMOutputForPPO(log_probs=log_probs, entropy=entropy, ...)
```

**兼容性论证**（不破坏现有路径）：
- 非 rmpad dense：hidden `[B,S,H]` → `[B,S]` → 引擎 :1244 正常工作（= 修复目标）
- rmpad：hidden `[1,total_nnz,H]` → `[1,total_nnz]` → 引擎 squeeze(0) → `(total_nnz,)` 不变
- 2D hidden（Megatron 等直接传 total_nnz）：reshape 为 `(total_nnz,)` 不变
- backward：kernel 内部仍收到扁平 dlogprobs（autograd 自动按 forward 形状还原梯度），`ctx.original_hidden_shape` 逻辑不受影响

## 4. 验证

- 代码级：5 文件语法通过；形状推理覆盖 3 条引擎路径（rmpad / 非 rmpad / 2D hidden）。
- GPU 级（待做）：`run_qwen3_8b_fsdp.sh` + triton + use_remove_padding=False 复现崩溃 → 打 patch → 首步训练通过；同时跑 use_remove_padding=True 回归确认无变化。

## 5. 待办

- [ ] 查重复检（提交前）
- [ ] GPU e2e 验证（当前无确认可用 GPU 服务器）
- [ ] 用户授权后 fork 开 PR（5 文件，可附单元测试：mock linear_cross_entropy 断言 forward 输出形状 == hidden.shape[:-1]）
