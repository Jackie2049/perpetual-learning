# 第五十七批补记（`max_off_policy_strategy=wait`）独立复核

复核者：M5 第二会话（2026-11-03 05:3x CST，与 wait A/B 执行会话并行）。
对象：[补测 README](README.md#补测max_off_policy_strategywaitdropless2026-11-03m5-第二会话)、`summary-wait.json` 与四份原始日志。
方式：只读；另写 [`verify-wait.py`](verify-wait.py)（逐 token 按 `" - "` 切分、`np.*` 包装单独解包、bootstrap 按跑分层 4000 次），**不 import** `make-summary-wait.py`。`verify-wait.py` exit 0。

## 一、逐跑与池化对账

| 运行 | gen（复核） | 整步（复核） | 生成组/步 | 逐出 |
|---|---|---|---|---|
| swt200 | 3.070086 | 22.118764 | 32 | 无 |
| swt200b | 3.231741 | 22.527012 | 32 | 无 |
| swt010 | 2.173703 | 21.064877 | 32 | 无 |
| swt010b | 2.179587 | 21.094060 | 32 | 无 |
| 池化 2.0 | 3.150913 | 22.322888 | 32 | 无 |
| 池化 0.1 | 2.176645 | 21.079468 | 32 | 无 |

差值：gen **−0.974268s（−30.92%）**、整步 **−1.243420s（−5.57%）**；与 `summary-wait.json` 的逐跑/池化 gen、step 在 **1e-9** 内一致（该文件另有一处 `generated_mean=32` 与 `spans/staleness=1.0`，同）。

## 二、数据完整性

- `md5sum -c md5sums.txt`：**12/12 OK**；服务器 `/tmp/rollout-poll-stale-swt{200,010,200b,010b}.log` 与本地 `raw/*.txt` **4/4 字节一致**。
- 配置解析：四跑分别为 `'poll_interval': 2.0/0.1/2.0/0.1`、`'off_policy_threshold': 1`、`'max_off_policy_strategy': 'wait'`。
- 步数：四跑均 15/15；窗口 steps 3–14（n=12/跑）。

## 三、代码层结构：日志之外的一层独立证据

这两条是从 `replay_buffer.py` 读出来的，不是从数据反推：

1. **wait 永不逐出**。`ReplayBufferAsync._stale_terminal_keys` 首行 `if partition_id == "val" or self.max_off_policy_strategy != "drop": return set()`；`_terminal_eviction_reasons` 随之给出空 stale 集，`_evict_terminal_groups` 不会发出 `off_policy/evicted_samples`。日志里该键**整体缺失**（不是值为 0），与此一致。
2. **threshold=1 的 wait 是完整屏障**。`_has_enough_samples` 对 `pending_keys | running_keys` 用 `(global_steps - prompt_global_steps + 1) >= threshold` 判定：阈值 1 时任何在飞批次都满足，采样必须等它变 terminal。因此 `trajectory_spans == 1` 是**结构必然**，四跑 12/12 步恒 1.00 不构成经验证据，只验证代码按预期跑。

含义：本实验测的是**最保守的 dropless 设置**（严格按步对齐、零丢弃）。`threshold=2` 的 wait（允许一步重叠）已由同一实验补测，见 [README 补测二](README.md#补测二wait--threshold2允许一步重叠轮询收益缩到噪声内但整步比-t1-屏障低约-14s2026-11-03m5-第三会话) 与 [verification-wait-t2.md](verification-wait-t2.md)；`threshold≥3` 结构上等同默认 async。

## 四、口径提示

- **跨策略表（wait vs drop）不能当单变量对照**：除阻塞/逐出策略外，两臂的每步生成量也不同（32 vs 64 组），且来自相邻窗口而非同窗口交错。README 已标"只作方向性证据"，保持该口径即可。
- **−1.243s 的可重复性**：2.0 臂两跑差 0.408s、0.1 臂差 0.029s，差值约为前者 3 倍，方向上强于 drop 补记的 ≈1 SE；但每臂只有两跑，**没有可报告的 run 级 CI**。本复核给出的 bootstrap CI（step [−1.786, −0.724]、gen [−1.392, −0.566]）是**步级、按跑分层**区间，只说明窗口内差异稳定，不等于 run 级标定。
- **整步差几乎全部来自 gen**：池化 `update_actor` 的 −0.199s 与单跑 sd 0.53–0.69 同量级，不作机制项。

## 五、边界

只读复核，未修改原实验任何文件，未重跑 GPU，不产生新的计时读数，未做任何上游写操作。所有比较以本地归档与服务器 `/tmp` 原件为准。
