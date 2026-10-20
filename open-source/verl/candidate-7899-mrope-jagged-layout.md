# verl：多模态位置编码在等长批次中的布局错误

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（后训练框架）

**【贡献点】** 多模态位置编码在等长批次中的布局错误。[Issue #7899](https://github.com/verl-project/verl/issues/7899) · [PR #7901](https://github.com/verl-project/verl/pull/7901)

**【类型】** 漏洞修复

**【价值评估】** 4 / 10 — 技术分析仍可复用，但截至记录核验已有他人方案覆盖，新增贡献空间有限。

**【进展】** 停止重复投入（2026-10-19 核验）：已有 [PR #7901](https://github.com/verl-project/verl/pull/7901) 覆盖；不是我方已合并成果，当前上游状态未重新查询。

**【大白话讲明白】**

多模态模型为图像和文本保存多组位置信息。等长样本让某种布局判断失效，后续代码就把维度看反。已有其他贡献者处理同一问题，我们保留分析供学习，不再重复提交。

**【What–Why–How】**

- **What：** 3D mRoPE position_ids 的 jagged 布局与等长输入。
- **Why：** 按形状推测布局会在特殊尺寸下产生歧义。
- **How：** 重建并验证布局；本地材料仅作参考，推进前应查看已有修复。
<!-- contribution-summary:end -->

## 详细记录

> 以下保留原始分析与历史进展；当前结论见上方速览。

> Issue: [verl-project/verl#7899](https://github.com/verl-project/verl/issues/7899)
> 创建 2026-09-17；0 评论；查重 2026-10-19：0 个引用 PR。关联 #6851（同崩溃，batch=1）、#7767（closed-unmerged，检测用 3/4 行启发式有缺陷）
> 状态：~~**verified-works**~~ → **2026-10-19 复查重纠正：同日已开 PR #7901（iceflysnow，rebuild jagged layout）**。验证材料保留，不提交。

## 0. 为什么值得做

| 维度 | 评估 |
|---|---|
| 影响 | 4 — 训练中途崩溃（等长 batch 必现，含 batch=1 平凡情形），错误归因极难（写侧产出后多步才炸，易被误判为通信故障）；多个用户中招（#6851 佐证） |
| 成本 | 4 — 消费侧 choke point 单函数修复，复用现有 `nested_tensor_from_tensor_list`；CPU 可验证 |
| 接受度 | 4 — 明确 bugfix、0 竞争；#7767 被拒给了改进空间（我们的方案无组件数启发式） |
| 契合度 | 3 — nested tensor 布局语义，中等契合 |
| 分 | **16** |

## 1. Bug

V1 trainer 的 3D mRoPE `position_ids`（每样本 `(C, seq_len)`）以 jagged NestedTensor 流转。当 batch 内所有序列**等长**（含 batch_size=1）时：

1. **写侧**：`torch.nested.as_nested_tensor(list_of_2d, layout=torch.jagged)` 把 component 维误当 ragged 维 → 产出 ragged@1 布局 `(B, j, D)`（values `(B·C, L)`，offsets 步长 C）而非 canonical ragged@2 `(B, C, j)`。
2. **消费侧**：`maybe_fix_3d_position_ids`（tensordict_utils.py:910）**无条件** `_ragged_idx = 2` → 元数据不一致 → 下一个 `unbind()` 崩（2.13: `split_with_sizes` 错误；2.2.2: "unbind only supported when jagged dimension is 1"）。

真实触发：35B MoE 多轮 RL，策略退化成"重复至截断"→ 全 batch 等长 → 第 82 步炸。**延迟引爆**模式（写侧产出多步后远处炸）使其极难归因。

## 2. 根因（本地复现确认，torch 2.2.2）

```text
等长: (128, j0, 34135) _ragged_idx=1 values=(512, 34135) offsets=[0,4,8,...]   ← 怪布局
变长: (3, 4, j2)       _ragged_idx=2 values=(4, 600)                            ← canonical
盲标 _ragged_idx=2 后 unbind(0) → RuntimeError
```

## 3. Patch（已应用本地 fork，tensordict_utils.py）

`maybe_fix_3d_position_ids` 从盲标改为**布局签名校验 + 重建**：

```python
nt = data["position_ids"]
if nt._ragged_idx == 2:
    return  # canonical
# ragged@1 3D：as_nested_tensor 把 component 维当 ragged（等长时必现）。
# 校验等长不变量 → 按 (B, C, L) 视图切回每样本 (C, L) → 用 verl 助手
# 以显式 ragged_idx=2 重建（sequence 维 ragged，component 维保留）。
values = nt.values()
offsets = nt.offsets()
lengths = offsets.diff()
assert lengths.min() == lengths.max(), "3D ragged@1 必须来自等长样本"
num_components = lengths[0].item()
batch = nt.shape[0]
per_sample = values.view(batch, num_components, values.shape[1])
data["position_ids"] = nested_tensor_from_tensor_list(
    [per_sample[i] for i in range(batch)], ragged_idx=2
)
```

- 复用同文件现有 `nested_tensor_from_tensor_list(ragged_idx=2)`（写侧本来就在用，生产环境 torch≥2.4 的 `nested_tensor_from_jagged` 路径）。
- **无组件数启发式**（#7767 缺陷）：不假设 C∈{3,4}，靠 offsets 步长推导。
- 等长不变量 assert：ragged@1 3D 只可能由等长样本产生（as_nested_tensor 语义），非等长说明另有未知路径 → 显式失败而非静默错。
- canonical 输入走 early return，零开销、零副作用。

## 4. 验证（CPU，torch 2.2.2 + 独立提取逐字逻辑）

| 检查 | 结果 |
|---|---|
| 怪布局复现（等长 16×4×128） | `(16, j0, 128)` ragged@1，offsets 步长 4 ✓ |
| 重建 values == `cat(per_sample, dim=1)` | True |
| 重建 offsets == `[0,128,256,...]`（cumsum 等长） | True |
| 每样本内容保真（vs 原始切片） | True |
| canonical（ragged@2）early return 身份保持 | True（同一对象） |
| 修复前盲标行为 | unbind CRASH ✓（bug 复现） |

限制：torch 2.2.2 的 jagged 系统不支持 ragged@2 的 unbind（`nested_tensor_from_jagged` 为 2.4+ API）；完整链路（canonical ragged@2 unbind → 128×(4, 34135)）由 issue 的 torch 2.13 CPU 复现 + 生产写侧既用同一助手佐证。

## 5. 分层防御（PR 可选跟进）

- 写侧（padding.py left_right_2_no_padding / seqlen_balancing.py restore_dynamic_batch / protocol.py serialize+deserialize）显式 `ragged_idx=2` 构建 + serialize 记录 `_ragged_idx`——治本；本 patch 是消费侧 choke point，先保证不炸。
- 本次 patch 单点覆盖全部写侧门路（base.py/engine_workers 都经 maybe_fix_3d_position_ids）。
