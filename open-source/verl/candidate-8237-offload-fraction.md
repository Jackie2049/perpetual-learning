# Candidate: megatron optimizer_offload_fraction < 1 崩溃 (#8237)

> Issue: [verl-project/verl#8237](https://github.com/verl-project/verl/issues/8237)
> 标签：bug；创建 2026-10-14；**2026-10-19 已回复 + PR #8286 已发 ready-for-review**
> 状态：**CLOSED**（根因 100% 实锤 + 修复方案 2 已 PR + issue 已回复，详见 §7 收官）

## 0. 为什么值得做

| 维度 | 评估 |
|---|---|
| 影响 | 5 — `optimizer_offload_fraction<1` + `optimizer_offload=True` 时首轮 update 必崩（0.0 Python 崩溃 / 0.5 原生崩溃无 traceback），多节点复现（8 节点 & 64 节点 DeepSeek-V4-Pro） |
| 成本 | 3 — 改动 5 行（offload 设备分类），但验证需 Megatron 环境；GPU 验证成本高 |
| 接受度 | 5 — 明确 bugfix、无人认领、reporter 给了两个可接受方向 |
| 契合度 | 5 — offload 是我们的深耕区（#6690 pin_memory 同族） |
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

- [x] GPU 环境验证（4090 完成：机制对照/单测/e2e 全过，见 §7）
- [x] 用户授权后：查重（无重复，与 #8235 resume 修复/#5651 feature 差异已说明）+ fork 开 PR（#8286）

## 7. 收官（2026-10-19）

**根因 100% 实锤**（三路证据互锁）：
- **代码实锤**：`offload_megatron_optimizer` HDO 分支（`verl/utils/megatron_utils.py:698`）对 `hdo.sub_optimizers` 无差别 `v.to("cpu")`，把 GPU 常驻的 FusedAdam 状态搬成 CPU。load 侧（`_move_new_state_to_right_device`，两版 megatron 逐字相同）会把 GPU 半修回 cuda，所以 verl080+bf16 端到端不崩（污染瞬时）。
- **4090 机制实证**：P1（纯 torch，双版本）证明"view 的底层存储被 `storage().resize_(0)` 释放 → `.to("cpu")` 报 `invalid argument`"；P2a（verl 真实函数 + 真实 HDO + FusedAdam + 悬空 fp32-view master_param）在 offload 行报 `CUDA error: invalid argument`（reporter 签名），修复后 PASS。污染态未经 load 进 fused step → SIGABRT/illegal memory access（独立脚本 3/3）。
- **模型条件实证**：DeepSeek-V4-Pro 公开（HF `deepseek-ai/DeepSeek-V4-Pro`），megatron 43e45e13 多处 `mark_keep_in_fp32`（`csa.py:957` ape、`:1621` attn_sink、`hyper_connection`、`mtp.*`）；`Float16Module` 用 `convert_module_to_dtype_except_fp32_marked` 跳过它们 → fp32 参数存活 → DistributedOptimizer 对 fp32 参数走 **view shard**（distrib_optimizer.py:474）→ 悬空 → 698 行 invalid argument。bf16 参数走 clone（:453）安全。fp8/fp4 量化权重 dequantize 成 bf16，不是触发源。

**修复（PR #8286，方案 2，commit df1435f）**：HDO 状态在 offload 中**完全不动**——HDO 已管理放置（fraction 部分住 CPU、GPU 半必须住 GPU），verl 只对非 HDO 优化器 offload `exp_avg/exp_avg_sq/master_param`；`offload_megatron_copy_params`（params/fp32 主 shard）与 `load_megatron_optimizer` 不变。回归测试：`test_optimizer_offload_and_load.py` 新增 HDO 用例（fraction 0.5/0.0，全 bf16 模型绕开 megatron 0.16.1 non-leaf 构造 bug，断言只盯 exp_avg/exp_avg_sq）。

**验证全绿（4090，verl080_back1 + megatron 0.16.1 + TE 2.16.1）**：
- 机制对照：buggy → offload 行 `CUDA error: invalid argument`；修复 → PASS（GPU 侧 exp_avg/exp_avg_sq 对象不变）
- 单测：`-k "hdo or precision_aware"` → **2 passed**（非 HDO 无回归）
- e2e 冒烟：fraction 1.0/0.5/0.0 全 exit 0（2 step）
- pre-commit 受影响文件全绿

**发布**：issue #8237 回复已发（根因 + PR，无 AI 标注，用户要求 PR/issue 均不标 AI）；PR #8286 ready-for-review。**两个环境前置 bug 与 #8237 无关**：verl_cdd9014f 丢 param_data guard 回归；megatron 0.16.1 `Float16Optimizer` KeyError + non-leaf 构造 bug。`patch-8237-offload-fraction.diff`（早期 8 行方案 1）已被实际 PR 的方案 2 取代。
