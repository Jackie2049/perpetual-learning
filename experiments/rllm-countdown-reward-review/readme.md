# rLLM 倒计时奖励复核：复现 issue #945，评估 PR #946

<!-- contribution-summary:start -->
**【仓库】** [perpetual-learning](https://github.com/Jackie2049/perpetual-learning)（AI 技术探索与贡献引擎）

**【贡献点】** 在 rLLM main `4e828c8` 的真实奖励函数上复现 issue #945 的四类"白拿满分"答案，并对修它的 PR #946 跑 22639 条合法答案的回归搜索。[复核结论](../open-source/rllm/countdown-reward-review-945.md) · [脚本](repro.py)

**【类型】** 实验记录（他人提交的独立复核）

**【价值评估】** 3 / 10 — 证据干净可复现，但问题已被认领、我方无可独立提交物。

**【进展】** 2026-11-04 完成；`repro.py` 退出码 0。未对外发布。

**【大白话讲明白】**

同目录的复核文档已解释背景；这里只记"怎么跑的、跑出什么数、哪里没测"。

**【What–Why–How】**

- **What：** `repro.py` 与上游文件快照 `countdown_reward_main_4e828c8.py`。
- **Why：** 需要一份不依赖网络、不装 rllm 就能重跑的证据，否则结论只能靠转述。
- **How：** 复现四类漏洞 + 指数链耗时 + 补丁回归搜索（含"拒绝一切"的负向对照）。见 [run-info.json](run-info.json)。
<!-- contribution-summary:end -->

## 怎么跑

```bash
python repro.py          # 退出码 0 = 结论成立（也可 `python experiments/rllm-countdown-reward-review/repro.py`，快照按脚本位置解析）
python repro.py --json
```

Python 3.12；不装 rllm，两个 `rllm` 导入用最小桩替代，`compute_score` 照跑。

## 读数

| 项 | 值 |
|---|---|
| issue #945 的四个例子在 main 上的得分 | 全 1.0（合法对照也 1.0） |
| `44**19**2` / `44**19**5` 耗时 | 约 0 秒 / **2.1 秒** |
| 补丁前后比较的合法答案 | **22639** 条（main 上得 1.0 的） |
| 补丁造成的回归 | **0** |
| 负向对照（补丁拒绝一切）报出的回归 | **159** 条 |

## 没做的

- 未跑真实训练，**这处漏洞是否被模型学会、影响多大无数据**；
- 未跑 rLLM 测试套件；
- 未在 fork 建分支、未开 PR、未评论。
