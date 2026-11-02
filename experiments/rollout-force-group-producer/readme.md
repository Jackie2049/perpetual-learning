# `force_group_size` 有没有生产者：§13.35 残留的全仓判定

支撑 [主文档 §13.37](../../notebook/rollout-performance-research.md)。

## 要解决的问题

§13.35 判决一写的是：`force_group_size` 在引擎入口能核到的所有 TRAIN 路径上恒为默认 1，**残留**是
「接线在 ray trainer 侧，而 ray trainer 不在该 checkout 的 `enable` 流程里，本批只核到 engine 入口为止，
不作全局保证」。

这句话读起来像「可能还有一条没核到的路径把开关打开了」。本批检验一个更强的命题：**整个 checkout 的
非测试代码里没有任何写入方**——即「读不到」不是某条路径没核到，而是这个键从来没有人写。

## 做法

静态读源码，不执行 verl、不给任何耗时数字。三层证据：

1. **写入方计数**：全仓扫 `assign_non_tensor(_data)?` 且键名是 `force_group_size` 的行（三种写法：
   位置参数、`key="..."` 关键字、直接 kwargs）。生产代码 **0 处**，测试 **1 处**
   （`tests/utils/test_prepare_micro_batches_with_group_size.py:64`，构造假输入用）。
2. **三条可能旁路逐条排除**：
   - *默认键注入*：`train_batch`（`engine_workers.py:335`）、`infer_batch`（`:389`）、tinker（`:134`）
     三处 `default_keys` 都是静态字面量，五个键为 `use_remove_padding / use_dynamic_bsz /
     max_token_len_per_gpu / micro_batch_size_per_gpu / use_fused_kernels`，都不含目标键。
   - *TransferQueue 回写*：`tqbridge` 会把输出的 `NonTensorData` 写入 `extra_info` 再回注 batch
     （`transferqueue_utils.py:135-152` 与 `111-132`，`@register` 对所有方法生效），理论上是一条
     能把键带进后续 batch 的通道；但 `train_mini_batch` 只回 `metrics`
     （`engine_workers.py:315-320`），目标键不进输出，通道断在源头。
   - *配置层*：`EngineConfig`（`verl/workers/config/engine.py:77-107`）没有该字段，用户在 yaml 里
     无处可配；`docs/`、`recipe/`、`examples/` 零引用。
3. **能力确实存在**：`rearrange_micro_batches` 里 `if force_group_size > 1:` 的真实分支在
   （`seqlen_balancing.py:415`），所以这不是「功能被删了」，而是「功能在、没人喂」。

## 结果

| 判决 | 结果 |
|---|---|
| 生产代码无写入方 | 是（0 处） |
| 唯一写入方是测试 | 是 |
| 四个源码锚点仍成立 | 是 |
| 三处 default_keys 都不注入它 | 是 |
| TransferQueue 回写带不走它 | 是 |
| **恒为默认 1 是代码结构结论**（非「某路径没核到」） | **是** |

引入史解释了为什么会这样：`80eb57ea`（#5438，2026-03-10）原本在 `use_dynamic_bsz=True` 时用
`assert force_group_size == 1` **禁止**使用它，那次改动删掉了断言并把参数透传进
`rearrange_micro_batches`——**只加了能力，没有加生产者**。与之相邻的 `c3fc222a`（#5424）在
`EngineConfig` 里加的是 `forward_only` / `param_offload`，与分组无关。

## 对 §13.35 的影响

只影响**定性**，不动任何数字：

- §13.35 判决一的「恒为默认 1」从**engine 入口级**升级为**全仓级**，残留可以划掉；
- §13.35 判决二（可达上限 0.47 倍）与判决三（vPP 结构账）**完全不变**——它们本来算的就是
  「接线之后能拿回多少、要付什么」，与当前有没有接线是两件事；
- §13.34 的 81.3% 切开率与 3.62× 前缀开销同样不变。

一句话：**0.47 倍是「如果有人把生产者写上」的收益上限，而本快照连生产者都没有**。

## 复现

```bash
cd experiments/rollout-force-group-producer
VERL_ROOT=../../verl python3 check_producers.py          # 覆盖 summary.json
VERL_ROOT=../../verl python3 check_producers.py --verify  # 与归档逐字段比对
python3 negative_control.py                              # 正则的负向对照 11/11
python3 -m unittest test_check_producers -v              # 7 passed
```

## 边界

- **静态源码核查，不执行 verl**。脚本只核「证据是否仍命中」（锚点文本在不在、写入方计数是否为 0），
  「没有生产者」这个判定本身是人读代码得出的。
- 只覆盖 `1a9e711e`（2026-07-14）这一个 checkout。上游若已加写入方，需重核（重跑脚本即可看到翻转）。
- 不涉及任何性能结论，也无 GPU 参与。

## 自查修正（三处，均实际发生）

1. 写入方正则第一版漏掉 `assign_non_tensor(td, force_group_size=4)` 这种 kwargs 写法——恰是
   `engine_workers.py:345` 默认键注入的同形；若上游真在那里加了键，本脚本会漏报。由
   `negative_control.py` 抓出。
2. `scan()` 第一版挂了 `lru_cache`，同一进程内注入写入方后第二次 `build()` 仍返回旧的 0，正向可行性
   检验**假通过**。已移除缓存，并把「注入后必须翻成 false」固化为单测。
3. `build()` 对不存在的 root 曾可能静默返回空集合，把「没扫到」读成「没有」。已改为显式
   `verl_missing=True` 且判决强制 false。
