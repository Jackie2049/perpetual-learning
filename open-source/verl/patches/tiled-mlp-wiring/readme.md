# 把 `tiled_mlp` 配置接到 FSDP 引擎（本地补丁，未提交上游）

<!-- contribution-summary:start -->
**【仓库】** [verl-project/verl](https://github.com/verl-project/verl)（LLM 后训练 / RL 训练框架，Apache-2.0）

**【贡献点】** `actor_rollout_ref.model.tiled_mlp` 有配置项与官方文档，但 FSDP 引擎调用 `apply_monkey_patch` 时从不传 `use_tiled_mlp`，配置完全无效。本补丁把配置接到调用点（3 行逻辑）。**本地补丁，未提交、未开 PR**——证据与验证见 [verify_patch.py](verify_patch.py)、问题取证见 [../../evidence/tiled-mlp-unwired/readme.md](../../evidence/tiled-mlp-unwired/readme.md)

**【类型】** 漏洞修复（配置项未接线）

**【价值评估】** 7 / 10 — 修复面小、语义清晰、边界可测（默认关／空配置／None 均安全回退）；但**未在真实 FSDP2 训练上验证** TiledMLP 本身的正确性与显存收益，只有接线层面的验证。

**【进展】** 补丁可干净应用（`patch -p1` 干跑与应用均 exit 0），修复前后调用参数已用脚本对比确认；**未提交上游**，需用户逐项授权。

**【大白话讲明白】**

开源框架 verl 提供了一个省显存的开关 `tiled_mlp`：把 MLP 拆成几片算。文档写了怎么开、`num_shards` 是什么意思，配置里也有正式位置。

问题是这个开关从来没被接到真正干活的地方：引擎创建模型时调用 `apply_monkey_patch`，但没把 `tiled_mlp` 配置传进去。于是用户打开它，显存一点没变，也没有任何提示。

这个补丁就是把配置读出来、传进去，总共 3 行逻辑。修复后默认行为完全不变（默认关闭），只有显式打开才会生效。

**【What–Why–How】**

- **What：** 把 `ModelConfig.tiled_mlp`（`{"enabled": bool, "num_shards": int}`）接到 `verl/workers/engine/fsdp/transformer_impl.py` 的 `apply_monkey_patch` 调用点。
- **Why：** `apply_monkey_patch` 早有 `use_tiled_mlp` / `tiled_mlp_shards` 两个形参并在体内分支（`monkey_patch.py:321`），但全仓唯一生产调用点从不传值；`verl/workers/engine/` 下 10 个引擎目录 grep `tiled_mlp` 均为 0。配置与文档都在，只差这一处转发。
- **How：** 用 `getattr(self.model_config, "tiled_mlp", None) or {}` 取配置（对 `None`/缺失安全），转成 `bool`/`int` 传入。验证：`patch -p1` 干跑与应用均 exit 0；`verify_patch.py` 对比修复前后调用点的关键字实参（前 5 个 → 后 7 个）；语法检查两版均通过；配置取值语义覆盖默认／开 4 片／开 8 片／空配置／`None` 五档。**边界**：只验证"配置是否传到"，**未运行 verl、未上 GPU**，不验证 TiledMLP 的正确性或显存收益；只覆盖 FSDP 引擎。
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

## 三、为什么没写测试

上游对 `transformer_impl.py` 的测试需要真实模型与 FSDP2 环境。本补丁只加了一次配置转发，**最合适的回归测试是端到端跑一次 `tiled_mlp.enabled=True` 的训练并在日志里看到 `apply_tiled_mlp_monkey_patch` 生效**——那需要 GPU，本机不具备。

因此这里只提供接线层验证。若上游接受，建议在 PR 里附带一个 CPU 可跑的单元测试：mock `apply_monkey_patch`，断言它收到 `use_tiled_mlp=True`。

## 四、边界与未核项

- **未运行 verl、未上 GPU**；不验证 TiledMLP 的正确性、显存收益或性能影响。
- 只覆盖 FSDP 引擎；`veomni`／`megatron` 等引擎是否也有同样问题未核（grep 为 0 但没有逐个读代码）。
- 文档说 TiledMLP 只在 FSDP2 下兼容——补丁没有加这条校验，若在 fsdp1 下开启，行为取决于 `apply_tiled_mlp_monkey_patch` 自身是否报错。**这一点需要在提 PR 前确认**，否则可能引入新的静默错误。
- **未提交上游**；推送分支、开 PR、评论均需用户逐项授权。

## 五、下一个动作（待授权）

1. 若要提交：请在 fork 上建分支 → 应用补丁 → 跑一次 FSDP2 小模型冒烟（需 M5 侧 GPU）→ 再决定是否开 PR。
2. 若不想提交：本补丁仍可作为与上游讨论时的具体材料——它把"接线缺失"从描述变成了可应用的 3 行。
