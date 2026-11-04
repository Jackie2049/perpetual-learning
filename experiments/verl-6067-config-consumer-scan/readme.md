# verl #6067 重构的配置消费点扫描（实验）

**目的**：把 TM-006 的 `tiled_mlp` 单条回归扩成"一类回归"——查 #6067（`044bbba2`，"deprecate workers, migrate to engines"）删掉的两个 worker 文件里，还有哪些配置读取点在新引擎侧没有消费者。

**结论（详见 [证据文档](../../open-source/verl/evidence/engine-refactor-config-consumers/readme.md)）**：102 个读取路径 → 文本候选 30 条 → **真失效 3 条**（`tiled_mlp`、`freeze_vision_tower`、`use_prefix_grouper`），26 条改名/换父配置后照常生效，1 条搬进 `HFModelConfig.__post_init__`。**`freeze_vision_tower` 为本次新发现。**

## 文件

| 文件 | 作用 |
|---|---|
| `scan_consumers.py` | 从 `044bbba2^` 的两个 worker 文件抽 `config.*` 读取点，在当前树查叶子名是否被读，输出候选清单 |
| `verify_findings.py` | 对候选逐条写死判定（旧位置 / 当前消费证据 / 期望结论），跑一遍必须 0 不一致；`--selftest` 注入读取点验证判据会翻转 |
| `scan-result.json` | 本轮 `scan_consumers.py` 输出（30 条候选） |
| `findings.json` | 本轮 `verify_findings.py` 输出（7 条判定，0 不一致） |

## 复现

```bash
# 需要一台有 verl 仓库的机器（本目录脚本只读该仓库，不改它）
python3 scan_consumers.py --repo <verl 仓库> --json scan-result.json
python3 verify_findings.py --repo <verl 仓库> --json findings.json   # 期望 exit 0
python3 verify_findings.py --selftest                                # 期望"自检通过"
```

本轮实测环境：verl main `1a9e711e`（2026-11-04 检出），父提交 `044bbba2^`，Python 3.12.13，macOS arm64，未跑 verl、未上 GPU。

## 脚本自身的两个坑（都已修，写在这里免得下次重犯）

1. **`lost` 判据的搜索范围不能包含 `verl/workers/config/`**：字段声明行本身就是一次文本出现，会让"没人读"自证成"有人读"。现在读取搜索排除该前缀，另用 `config_declarations` 独立确认字段仍在。
2. **不能用 `\b<key>\b` 匹配**：`_` 是 Python 的单词字符，`\btiled_mlp\b` 匹配不上 `use_tiled_mlp`、`tiled_mlp_config`。改成子串匹配，宁可多报（误报由 `verify_findings.py` 的逐条判定兜底）。

## 已知未做

- 没有跑 GPU，因此"开 `freeze_vision_tower` 到底省不省显存"没有实测（属 M5 真机范围）。
- 只覆盖被删的两个 worker 文件的读取点，**不覆盖** #6067 同时修改的另外 71 个文件的语义变化。MTP 那条（`raise NotImplementedError` → 静默清零）是人工读 diff 才发现的，扫描脚本看不见这种行为契约变化。
- Megatron 侧 26 条"改名/生效"的判定只抽查了 3 条的证据链（`mtp_loss_scaling_factor`、`log_prob_micro_batch_size`、`fsdp_size`）。
