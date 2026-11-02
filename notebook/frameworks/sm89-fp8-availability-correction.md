# 订正：RTX 4090（SM89）并非"无 FP8"，出错的是推断而非硬件

<!-- contribution-summary:start -->
**【仓库】** 本仓库（perpetual-learning）｜影响 [rollout 专项](notebook/rollout-performance-research.md) 的 P-2 可行性判断

**【贡献点】** 订正一条**跨 ~30 个文件**的过期判断：2026-06 的笔记把「PyTorch #184119 在某评审环节停滞」推断成「SM89 FP8 训练不可用，RTX 4090 不能用 FP8」，并据此把 P-2 记为"硬件门槛未评估"。核实上游后发现**推断错了**：#184119 是**安全防护**（避免 pre-sm90 上的 fp8→bf16 prologue fusion）而非"FP8 不可用"；vLLM 的 Triton FP8 路径在 SM89 上**本来就是允许的**。

**【类型】** 结论订正（硬件可行性判断）

**【价值评估】** 8 / 10 — 直接改变 P-2 的可行性判断：从"硬件门槛未评估"变成"推理侧 FP8 可用，训练侧需实测"。但**不能据此宣布 P-2 可做**，训练侧仍在本次核实范围之外。

**【进展】** 上游核实完成（3 条 PR/issue），本仓库根因已定位；**正在同步受影响文件**。未发布任何上游内容。

**【大白话讲明白】**

2026 年 6 月的笔记里写过一句「RTX 4090 不支持 FP8 训练」。这条判断后来被反复引用，成了"P-2 做不了"的主要理由。

本轮去查原文，发现当时的推理链有问题：真正的上游 PR（#184119）说的是"在较老的 GPU 上，编译器不应把 FP8 转 BF16 的转换融合进矩阵乘法，那样生成的代码会出错"——这是一条**安全护栏**，恰恰说明 FP8 在这些卡上是**能用**的，只是某个优化不能开。而 vLLM 那边把 SM89（4090 所在架构）设成了 Triton FP8 路径的**允许下界**，不是禁止线。

所以准确的结论是：4090 上 FP8 **推理侧可用**（Triton 路径），训练侧还没人跑过——那是"未测"，不是"不能"。

**【What–Why–How】**

- **What：** 订正"RTX 4090 无 FP8 训练 / SM89 FP8 Blocked"这一判断，及其对 P-2 可行性评估的影响。
- **Why：** 根因是**把安全护栏读成了能力缺失**。PyTorch PR #184119 标题即「Avoid fp8 bf16 prologue fusion on pre-sm90」，作用是阻止编译器在 pre-sm90 上生成有问题的融合代码；它 closed 且 **merged=False**。与此同时 vLLM #43914「Triton KV dtype: enforce fp8>=SM89」与 #40127「SM>=89 guard for Triton block FP8」把 SM89 设为**允许下界**——方向与"不可用"相反。真正不支持的是 FlashInfer backend 与 compressed-tensors override（#44879），属**特定后端**限制。
- **How：** 用 GitHub API 核对三条 PR 的标题、状态与合并情况；回读本仓库自己的 [vllm-v0.23-rtx4090-impact-reading.md](vllm-v0.23-rtx4090-impact-reading.md) §4，该节当时就已写下「之前的理解（错误）：'SM89 不支持任何 FP8' → 过于简化」，但后续文件仍沿用了简化版。**边界**：本次只核实**推理侧** FP8 路径与 #184119 的语义；**训练侧（TE／FP8 优化器／梯度）未核实**，P-2 仍需实测才能定论。
<!-- contribution-summary:end -->

## 一、证据链

| 项 | 核实结果 |
|---|---|
| PyTorch #184119 标题 | 「Avoid fp8 bf16 prologue fusion on pre-sm90」 |
| 该 PR 的作用 | **阻止**编译器在 pre-sm90 上把 fp8 读入融合进 bf16 点积——**安全护栏** |
| 合并状态 | closed，**merged: false**（2026-09-19 最后更新） |
| vLLM #43914 | 「Triton KV dtype: enforce **fp8>=SM89**」→ SM89 是**允许下界** |
| vLLM #40127 | 「**SM>=89 guard** for Triton block FP8」→ SM89 允许 |
| 真正不可用的 | FlashInfer backend；compressed-tensors override（#44879）；NVFP4 需 SM100+ |

**关键点**：`fp8>=SM89` 的意思是"FP8 从 SM89 起可用"，不是"SM89 被排除"。

## 二、根因

1. 2026-06-20 的 [7-framework-status-update](7-framework-status-update-2026-06-20.md):372 写了
   「SM89 FP8 | Blocked (#184119) | **FP8 training not available on RTX 4090**」。
2. 该表的依据是"#184119 停滞"，而 #184119 是**防护类** PR——它停滞不等于能力缺失。
3. 这个简化结论随后被 ~30 个文件引用，包括 P-2 的可行性判断。

**更值得注意的是**：本仓库 [vllm-v0.23-rtx4090-impact-reading.md](vllm-v0.23-rtx4090-impact-reading.md) §4 当时**已经纠过一次**
（"之前的理解（错误）：SM89 不支持任何 FP8 → 过于简化 → 不完全正确"），但那份更准确的表述没有传播开，
简化版继续被引用。**订正传播不出去，等于没订正。**

## 三、对 P-2 的影响（谨慎表述）

| 之前 | 现在 |
|---|---|
| "需 FP8 训练栈；当前硬件条件未评估" | **推理侧** FP8 在 4090 上可用（Triton 路径）；**训练侧未测** |
| 隐含"硬件门槛" | 门槛可能不存在，但**未验证** |

**不改变**：P-2 仍未完成；仍需一次真实 FP8 训练实验才能定论。
**改变**：把它从"硬件不可行"改成"**未测**"——这两个说法在排期和优先级上完全不同。

## 四、边界

- 只核实**推理侧** SM89 FP8 路径与 #184119 的语义；**训练侧未核**（TE、FP8 optimizer、梯度缩放）。
- 上表三条 PR 状态取 2026-11-02 的 GitHub API 快照，会随上游变化。
- **未运行任何 GPU 实验**；本节不主张"4090 能做 FP8 训练"，只主张"原判断的推理链是错的"。
