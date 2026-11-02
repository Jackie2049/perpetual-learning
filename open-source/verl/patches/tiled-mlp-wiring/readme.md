# 把 `tiled_mlp` 配置接到 FSDP 引擎（本地补丁，未提交上游）

<!-- contribution-summary:start -->
**【仓库】** [verl-project/verl](https://github.com/verl-project/verl)（LLM 后训练 / RL 训练框架，Apache-2.0）

**【贡献点】** `actor_rollout_ref.model.tiled_mlp` 有配置项与官方文档，但 FSDP 引擎调用 `apply_monkey_patch` 时从不传 `use_tiled_mlp`，配置完全无效。本补丁把配置接到调用点（3 行逻辑）。**本地补丁，未提交、未开 PR**——证据与验证见 [verify_patch.py](verify_patch.py)、问题取证见 [../../evidence/tiled-mlp-unwired/readme.md](../../evidence/tiled-mlp-unwired/readme.md)

**【类型】** 漏洞修复（配置项未接线）

**【价值评估】** 7 / 10 — 修复面小、语义清晰、边界可测（默认关／空配置／None 均安全回退）；补丁后已在 FSDP2 真机验证接线与数值（见 §三）。**但选项自身的收益被高估**：verl 默认 `enable_gradient_checkpointing=True` 时 TiledMLP 不省显存（5.46 GB 不变），只有关掉检查点才从 16.39 GB 降到 8.74 GB——因此这是「配置项名不副实」的证据，不是一条能带来默认收益的优化。

**【进展】** 补丁可干净应用（`patch -p1` 干跑与应用均 exit 0），修复前后调用参数已用脚本对比确认；**M5 侧 4090 真机 FSDP2 冒烟已完成（2026-11-02）**：打补丁后 `enabled=True` 真实 patch 到 MLP 类并可跑通前反向，不打补丁时配置逐位无效；**FSDP1 下开启会硬报错**；**verl 默认梯度检查点开启时无显存收益**。详见 [真机冒烟记录](../../../../experiments/verl-tiled-mlp-fsdp-smoke/readme.md)。**未提交上游**，需用户逐项授权。

**【大白话讲明白】**

开源框架 verl 提供了一个省显存的开关 `tiled_mlp`：把 MLP 拆成几片算。文档写了怎么开、`num_shards` 是什么意思，配置里也有正式位置。

问题是这个开关从来没被接到真正干活的地方：引擎创建模型时调用 `apply_monkey_patch`，但没把 `tiled_mlp` 配置传进去。于是用户打开它，显存一点没变，也没有任何提示。

这个补丁就是把配置读出来、传进去，总共 3 行逻辑。修复后默认行为完全不变（默认关闭），只有显式打开才会生效。

真机验证后还要补一句实话：把这个开关接通以后，它**只在关掉梯度检查点时才真的省显存**（0.5B 模型 8k token 下 16.4 GB → 8.7 GB）；verl 默认是开启梯度检查点的，那时两者机制重叠，显存一点不变。所以这个问题的价值在于「配置项名不副实」，不在于「开了就能赚」。另外在 FSDP1 下打开会直接报错，提 PR 时应一并加校验。

**【What–Why–How】**

- **What：** 把 `ModelConfig.tiled_mlp`（`{"enabled": bool, "num_shards": int}`）接到 `verl/workers/engine/fsdp/transformer_impl.py` 的 `apply_monkey_patch` 调用点。
- **Why：** `apply_monkey_patch` 早有 `use_tiled_mlp` / `tiled_mlp_shards` 两个形参并在体内分支（`monkey_patch.py:321`），但全仓唯一生产调用点从不传值；`verl/workers/engine/` 下 10 个引擎目录 grep `tiled_mlp` 均为 0。配置与文档都在，只差这一处转发。
- **How：** 用 `getattr(self.model_config, "tiled_mlp", None) or {}` 取配置（对 `None`/缺失安全），转成 `bool`/`int` 传入。验证：`patch -p1` 干跑与应用均 exit 0；`verify_patch.py` 对比修复前后调用点的关键字实参（前 5 个 → 后 7 个）；语法检查两版均通过；配置取值语义覆盖默认／开 4 片／开 8 片／空配置／`None` 五档。**真机（4090）**：FSDP2 下配置真实生效、前反向可跑通、fp64 数值精确等价；FSDP1 下硬报错。**边界**：只覆盖 FSDP 引擎；单卡单机、无 optimizer step，不验证多 rank 下的梯度 reduce。
<!-- contribution-summary:end -->

## 一、补丁内容

改动 `verl/workers/engine/fsdp/transformer_impl.py` 的 `_build_module`：

```python
  use_fused_kernels = self.model_config.use_fused_kernels
+ _tiled_mlp = getattr(self.model_config, "tiled_mlp", None) or {}
  apply_monkey_patch(
      model=module,
      use_remove_padding=self.use_remove_padding,
      ulysses_sp_size=self.ulysses_sequence_parallel_size,
      use_fused_kernels=use_fused_kernels,
      fused_kernels_backend=fused_kernels_backend,
+     use_tiled_mlp=bool(_tiled_mlp.get("enabled", False)),
+     tiled_mlp_shards=int(_tiled_mlp.get("num_shards", 4)),
  )
```

完整 diff：[tiled-mlp-wiring.patch](tiled-mlp-wiring.patch)。

## 二、验证

```bash
# 1. 干跑
patch -p1 --dry-run -d <unpacked-verl> < tiled-mlp-wiring.patch

# 2. 修复前：应报"未接线"
python3 verify_patch.py <unpacked-verl>

# 3. 修复后：应报"已接线"
patch -p1 -d <unpacked-verl> < tiled-mlp-wiring.patch
python3 verify_patch.py <unpacked-verl> --expect-wired
```

实测输出（2026-11-02，源 = main `6879e1ab07`）：

| | 调用点关键字实参 | 判定 |
|---|---|---|
| 修复前 | `model, use_remove_padding, ulysses_sp_size, use_fused_kernels, fused_kernels_backend` | PASS（确认未接线） |
| 修复后 | 上述 5 个 + `use_tiled_mlp, tiled_mlp_shards` | PASS（配置已接线） |

配置取值语义（五档）：

| 配置 | `use_tiled_mlp` | `tiled_mlp_shards` |
|---|---|---|
| 默认 `{"enabled": False, "num_shards": 4}` | False | 4 |
| `{"enabled": True, "num_shards": 4}` | True | 4 |
| `{"enabled": True, "num_shards": 8}` | True | 8 |
| `{}` | False | 4 |
| `None` | False | 4 |

**默认行为不变**——只有显式 `enabled: True` 才会生效。

## 三、4090 真机冒烟（M5，2026-11-02）

在 `6879e1ab07` 上构造真实 `FSDPEngine`（`EngineRegistry.new` → `engine.initialize()` → `forward_backward_batch` + `ppo_loss`），单卡 4090。完整记录、原始 JSON/日志与复现命令见 [实验归档](../../../../experiments/verl-tiled-mlp-fsdp-smoke/readme.md)。

| 问题 | 结果 |
|---|---|
| 补丁后 `enabled=True` 是否真的生效（FSDP2） | 是。`Qwen3MLP.forward` / `Qwen2MLP.forward` 变为 `tiled_forward`，日志出现 `TiledMLP monkey patch applied to: ... (shards=N)` |
| 不打补丁时 | 与 `enabled=False` **逐位相同**（梯度相对差 0.00e+00），配置被静默吞掉 |
| 数值是否一致（FSDP2） | fp64 CPU 对照差异 1e-15（浮点舍入级）；bf16 下 1e-3 ~ 1e-5，换成 fp32 混合精度后降到 1e-7 且 logprob 逐位相同 → 偏差来自 bf16 GEMM 形状舍入，不是逻辑错误 |
| FSDP1（`strategy=fsdp`）开启 | **硬报错**：`RuntimeError: setStorage ... out of bounds for storage of size 0`（默认与 `reshard_after_forward=False`）、`expected mat1 and mat2 to have the same dtype`（`use_orig_params=True`）。机制为 FSDP1 前向结束后已 reshard，TiledMLP 的反向重算读到已释放存储 |
| 显存收益（Qwen2.5-0.5B，8192 token） | 梯度检查点**关**：16.39 → 8.74 GB（−46.7%，4 片即饱和）；检查点**开**（verl 默认）：5.46 → 5.46 GB，**±0** |
| 分片数 > token 数的边界 | `torch.chunk` 只切出更少的块，`GradientAccumulator` 的「最后一片」分支不触发，但 fp64 对照仍精确 → 不影响正确性 |

**对提交方案的影响**：补丁把 FSDP1 上「静默无效」变成「第一个反向崩」，所以建议 PR 里同时加一条配置期校验（`tiled_mlp.enabled=True` 且 `strategy != "fsdp2"` 时抛明确错误），并避免在描述里写「省显存」而不带前提。

## 四、为什么没写测试

上游对 `transformer_impl.py` 的测试需要真实模型与 FSDP2 环境。本补丁只加了一次配置转发，最合适的回归测试是端到端跑一次 `tiled_mlp.enabled=True` 的训练并在日志里看到 `apply_tiled_mlp_monkey_patch` 生效——这一步已由 M5 在 4090 上完成（§三），但目前是外部实验脚本，不是仓库内测试。

若上游接受，建议在 PR 里附带一个 CPU 可跑的单元测试：mock `apply_monkey_patch`，断言它收到 `use_tiled_mlp=True`。

## 五、边界与未核项

- 真机验证范围：单卡、单机、只做前反向（无 optimizer step / 多 rank 梯度 reduce）；0.5B 显存结论是单模型单形状，更大模型下 MLP 激活占比不同。
- 只覆盖 FSDP 引擎；`veomni`／`megatron` 等引擎是否也有同样问题未核（grep 为 0 但没有逐个读代码），因此即使合并，其他引擎仍是静默无效。
- **FSDP1 下开启已实测为硬报错**（§三），错误信息是 storage 越界，看不出根因。补丁本身没有加校验，所以提 PR 时**建议同时加配置期校验**，否则等于把静默无效换成难懂的运行期崩溃。
- **未提交上游**；推送分支、开 PR、评论均需用户逐项授权。

## 六、下一个动作（待授权）

1. 若要提交：在 fork 上建分支 → 应用补丁（可加上文建议的 fsdp1 配置校验）→ 再决定是否开 PR。真机冒烟已完成，见 [实验归档](../../../../experiments/verl-tiled-mlp-fsdp-smoke/readme.md)。
2. 若不想提交：本补丁仍可作为与上游讨论时的具体材料——它把"接线缺失"从描述变成了可应用的 3 行，并附上了「默认配置下不省显存」这一反直觉证据。
