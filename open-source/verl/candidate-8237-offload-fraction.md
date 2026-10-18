# Candidate: megatron optimizer_offload_fraction < 1 崩溃 (#8237)

> Issue: [verl-project/verl#8237](https://github.com/verl-project/verl/issues/8237)
> 标签：bug；创建 2026-10-14；0 评论；**0 个引用 PR（2026-10-19 复查重确认：可解析 total_count=0，真实无人认领）**
> 状态：`analysis-done`（根因确认 + **代码级验证完成**，patch 方案已拟，待 GPU 验证 + 用户授权）

## 0. 为什么值得做

| 维度 | 评估 |
|---|---|
| I | 5 — `optimizer_offload_fraction<1` + `optimizer_offload=True` 时首轮 update 必崩（0.0 Python 崩溃 / 0.5 原生崩溃无 traceback），多节点复现（8 节点 & 64 节点 DeepSeek-V4-Pro） |
| C | 3 — 改动 5 行（offload 设备分类），但验证需 Megatron 环境；GPU 验证成本高 |
| A | 5 — 明确 bugfix、无人认领、reporter 给了两个可接受方向 |
| F | 5 — offload 是我们的深耕区（#6690 pin_memory 同族） |
| 分 | **42**（本轮扫描最高） |

## 1. Bug

`optimizer_offload_fraction<1`（GPU/CPU 混合存放 Adam 状态）+ `optimizer_offload=True` 时：
- fraction=0.0：`offload_megatron_optimizer` 里 `v.to("cpu")` 报 `CUDA error: invalid argument`（异步错误，真实触发点待 logs）
- fraction=0.5：首个 `update_actor` 期间 WorkerDict 进程原生崩溃（无 Python traceback，Ray SYSTEM_ERROR）
- fraction=1.0（默认，全 CPU）：正常

## 2. 根因（本地源码确认，2026-10-19）

**verl 侧** `verl/utils/megatron_utils.py:678` `offload_megatron_optimizer`：
```python
hdo = _opt.optimizer
if all(hasattr(hdo, attr) for attr in ("sub_optimizers", "inner_param_to_orig_param", "state")):
    for optimizer in hdo.sub_optimizers:
        for param, state in optimizer.state.items():
            for k, v in state.items():
                if not isinstance(v, torch.Tensor):
                    continue
                orig_param = hdo.inner_param_to_orig_param.get(param, param)
                hdo.state[orig_param][k] = state[k] = v.to("cpu")   # ← 无条件搬 CPU
```
→ 把 **GPU 子优化器（FusedAdam）的状态也搬成 CPU tensor**，FusedAdam step 在 CPU tensor 上 → 原生崩溃。

**Megatron-LM 侧** `megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py:357` `_move_new_state_to_right_device` 用子优化器类别决定状态去向：
```python
if isinstance(optimizer, self.defaults["cpu_optimizer_cls"]):
    self.state[orig_param][k] = state[k] = v.to("cpu")
else:
    self.state[orig_param][k] = state[k] = v.to("cuda")
```
→ **verl 的 offload 绕过了 HDO 自己的设备分类**，直接把设备语义打乱。load 侧 `_move_new_state_to_right_device` 是对的，问题只在 offload。

## 3. Patch（已拟，未验证）

镜像 HDO 的设备分类，只搬 CPU 子优化器的状态；GPU 子优化器状态保持不动（它本来就该在 GPU）：

```python
for optimizer in hdo.sub_optimizers:
    # Only CPU-resident sub-optimizer state should be moved: moving the
    # GPU sub-optimizer's (FusedAdam) state to CPU corrupts its device
    # assumption and crashes the next fused step (issue #8237). Mirror
    # HybridDeviceOptimizer._move_new_state_to_right_device's device
    # classification.
    is_cpu_opt = isinstance(optimizer, hdo.defaults.get("cpu_optimizer_cls", ()))
    if not is_cpu_opt:
        continue
    for param, state in optimizer.state.items():
        ...
```

注：`hdo.defaults` 存在性需在环境确认（`defaults` 是 HDO 构造属性，源码 `hybrid_optimizer.py` 可见）。`master_param`（use_precision_aware_optimizer）随同处理：只搬 CPU 子优化器自己的 master_param。

## 4. 验证

- **代码级（2026-10-19 完成，本地 Megatron-LM hybrid_optimizer.py 逐行核对）**：
  - `sub_optimizers` 为属性（:466）：`cpu_optimizers + [gpu_optimizer]`（gpu 为 None 时仅 cpu 列表）。
  - fraction<1 时参组划分（`_get_sub_optimizer_param_groups`：按 numel 顺序 offload 前 fraction 到 CPU pin_memory 副本）→ **cpu_optimizers 与 gpu_optimizer（GPUAdam/FusedAdam）各有参**，二者并存。✓ patch 的 `is_cpu_opt` 判定有效。
  - fraction=1.0：gpu_optimizer=None → sub_optimizers 全为 CPU 优化器 → 现代码 no-op 移动，正常（与 issue 观察一致）。
  - fraction=0.0：offload_threshold=0 → 全部留在 GPU，cpu_optimizers 空 → 现代码把所有 GPU 状态搬 CPU → 崩溃（0.0 具体报错点需 reporter logs；patch 侧绕开：**不搬任何状态**，ckpt save 直接处理 GPU 驻留状态——非 offload 训练本就如此，合法）。
  - `defaults["cpu_optimizer_cls"]` 存在（:59-62 构造时写入）。✓
  - overlap_cpu_optimizer_d2h_h2d=True 时 cpu_optimizers 为**列表**（build_cpu_optimizer_list）→ 循环语义正确。✓
- GPU 级（待做）：有 GPU 服务器时用 verl megatron 单卡/小模型复现 fraction=0.5 崩溃 → 打 patch → 跑 2-3 步验证不再崩。当前 GPU 条件未确认，暂缓。

## 5. 备选方案（reporter 提出）

- 方案 B：config validation 拒绝 `optimizer_offload_fraction<1` + `optimizer_offload=True`（安全但砍功能，64 节点 905/952GB host 满的用户场景是合理诉求）。
- 方案 A（本候选采用）：offload 跳过 GPU 子优化器状态。若 HDO 自有 offload 方法存在（verl 代码 TODO 也提到"should be a method in Megatron-LM"），可改调 HDO 原生接口，但需 megatron-core 版本确认。

## 6. 待办

- [ ] GPU 环境验证（当前无确认可用的 GPU 服务器）
- [ ] 用户授权后：查重（提交前）+ fork 开 PR（PR 描述按 verl CLAUDE.md：为何不重复/测试/声明 AI 协助）
