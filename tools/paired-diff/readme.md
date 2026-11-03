# paired-diff — 配对差判读的共享判据

解决什么问题
---

"臂 A 比基线好/坏多少"这类读数，只报一个均值会说谎：同一个均值既可能来自"每个种子都同向"，也可能来自"一半正一半负正好抵消"；而一个 1.6% 的效应可能比基线自己换个输入顺序就能动的 11.7pp 还小。

这个工具把判读那一套判据收在一处，供各实验脚本 import。此前同一族判据在四个脚本里各写一遍（容差、字段名、舍入都各写各的），改正一处不会带动另一处——这正是 2026-11-03 第四十二批起一直没做的那件事。

用法
---

作为库：

```python
import paired_diff as pd

pd.sign_counts(diffs)            # {'n_positive', 'n_negative', 'n_identical'}
pd.same_sign_fraction(diffs)     # 多数方向占比——引用前先看 n_identical
pd.summarize_paired(diffs, scale=100)   # 上面这些 + sd/min/max
pd.exceeds_band(effect, band)    # 判读门槛：|效应| > 基线自身抖动？
pd.unresolved(diffs)             # 这一格是不是根本没分辨率
pd.span(values)                  # 极差：换阈值/换顺序差多少
pd.sign_flip_around(values, 1.0) # 比率读数是否穿过 1
```

自检：`python3 tools/paired-diff/paired_diff.py --selftest`

判读三步（次序不能换）
---

1. **`n_identical` 有没有吃掉大半种子** —— 两臂读数完全相同不是"一致"，是这一格没有分辨率（低积压时 `R` 本来就接近 0，两臂一起落在 0 上）。
2. **`same_sign_fraction`** —— 均值是不是靠少数种子撑起来的。
3. **`|效应| > 带宽`** —— 效应是否大到无法用基线自身抖动解释。

三步任一不过就只能写"方向未定"。`exceeds_band` 是判读门槛，**不是显著性检验**：它没做 CI 或置换。

两个口径要点
---

**容差是绝对容差，且必须在调用时解析。** `TOL = 1e-12`，不是相对容差——R 在 `[0,1]`、步数在 `[0,300]` 时它等价于"精确相等"。所有函数签名写 `tol=None` 并在函数体内解析，不能写 `def f(tol=TOL)`：后者会在定义那一刻把值绑进默认参数，之后改 `TOL` 不生效，也无法从外部收紧。抽工具时实测到过——注入 `TOL=1e-3` 后 `sign_counts([1e-6])` 仍记正。

**比率的"翻号"参考点是 1，不是 0。** `steps_ratio`、`wall_ratio` 都 > 0，把它们交给相对零的 `sign_flip` 会一律返回 False。这个语义差点在抽工具时丢掉，结果把 p3 的 `A` 翻号读没了（0.978 → 1.005 明明翻了）。用 `sign_flip_around(values, 1.0)`。

验证范围与已知边界
---

- **数值口径不变**：四个脚本改接后，`ledger.json`、`bucketing-probe.json`、`real-lengths-summary.json`、`summary.json`、`legacy-group-arm-check.json` 五份归档的 `--verify` 全部逐字段相同；测试由 113 增至 121 passed。
- **注入负向对照四组全部被测红**：`TOL` 1e-12 → 1e-3（补测试后抓到）、`same_sign_fraction` 恒 1.0、`exceeds_band` 恒 True、`sign_flip_around` 只认零。
- **一个已知缺口**：容差注入在补测试前**113 个测试全绿**——既有测试的配对差一律取 0 或明显非零，从没造出"小于 1e-3 但大于 1e-12"的差值。补的 `TestSharedTolerance` 三条规定住了这条边界。
- 不提供显著性检验（CI／置换）。需要 CI 的场合另用各脚本自带的 bootstrap。

跨脚本一致性（2026-11-03 第二批）
---

**归档 `--verify` 相同不等于两版实现相同。** 它只说明"既有数据没落进两版判据的分歧区"。要回答"四个脚本对同一列输入是否给同一个答案"，得**同一列输入分别喂旧版手写实现与共享库、逐值比对**。已建 [compare_old_new_impl.py](../../experiments/rollout-c-class-ledger/compare_old_new_impl.py) 与 [test_cross_script_consistency.py](../../experiments/rollout-c-class-ledger/test_cross_script_consistency.py)：

- 真实输入列从源 summary **逐种子重算**（`wall-threshold-band.json`／`ledger.json` 只留计数与汇总量，反解不出逐种子差值）——**96 列／2480 个差值／332 个 identical 种子，零分歧**；
- 210 组合成边界列 + 200 组极差／翻号，全部零分歧；
- 对照自身做过注入验证（把正号计数偏 +1 会报 FAIL，清除后恢复 PASS）。

**重建口径上自己踩的两个坑**：① 第一版从归档反解，读出"0 列、PASS"——**没扫到东西的 PASS 什么也没证明**，已加"真实列不足 40 即 FAIL"的覆盖下限；② p3 的基线是 `R_nofilter` 不是第一个条目，第一版拿 `R` 自己比自己，读出"30/30 identical"的假象。

**三条跨脚本的已知差异（不是 bug，但键名不能混用）**：

- **比率必须平移后再交给 `sign_counts`**。`sign_counts` 相对零分类，直接喂 0.978 会记成"正"，而它的含义是"比 1 小"。三个比率型调用点都显式写 `v - 1.0`。
- **距 1 不到 `TOL` 的比率会被重新定基吞掉**（`1.0+1e-13` 减出来是 9.99e-14 < 1e-12），记成 identical。臂真的只差 1e-15 时也没有分辨率；既有归档的 353 个比率里没有一个落进这条盲区（3 个恰好等于 1.0，都正确记为 identical）。
- **整数步号用 `d == 0`，浮点比率用 `|d| <= TOL`**。整数输入下两者等价；非整数时"恰好差 5e-13 步"在步号口径记非并列、在比率口径记相同——都对，但不能共用一个键名。

使用它的地方
---

| 脚本 | 用的判据 |
|---|---|
| `experiments/rollout-group-split-tradeoff/bucketing_probe.py` | `sign_counts`、`same_sign_fraction`、`sd_pp`、`exceeds_band` |
| `experiments/rollout-group-split-tradeoff/real_lengths_policy.py` | 同上（真实负载侧） |
| `experiments/rollout-c-class-ledger/ledger.py` | `sign_counts`、`span`、`sign_flip_around(…, 1.0)`、`sign_flip`（步号，相对零） |
| `experiments/rollout-c-class-ledger/wall_threshold_band.py` | `span`、`sign_flip_around`、`sign_counts` |

舍入口径与刻度（2026-11-03 第三批）
---

第七批的逐值对照只覆盖**计数**。`same_sign_fraction` 与 `sd_pp` 是写进正文的读数（同号占比、种子间标准差），此前没有逐值对照，也不知道舍入口径。本批补了两条声明和一条实测：

**一、第 3 位小数是 `round(x, 3)`，对精确平局取偶。** 内建 `round` 对 `x.xxx5` 取偶：`0.3125 → 0.312`、`0.0625 → 0.062`，而"保留三位小数"的人类读法是 `0.313`／`0.063`。**归档里出现平局时，第 3 位取决于用的是哪一种**——这一点必须写明，不能靠"我们一直用 round"默认。

**二、真实输入里不存在平局。** 差值来自浮点除法与求和，末位不会恰好落在 5 上。96 条真实输入列 + 8 条构造平局列实测：真实列 `sd_pp` 落在舍入平局上的个数为 **0**。⇒ 上述取偶口径目前不影响任何已发布读数；`rounding_cases()` 把构造平局与两种读法的差都留成可复现的样例。

**三、`TOL` 是绝对容差，必须和刻度一起引用。** 比率型读数在不同刻度上的"零"不一样：

| 调用点 | 刻度 | `TOL` 相当于容忍多大的原始偏差 |
|---|---|---|
| `real_lengths_policy`／`bucketing_probe`（`v - 1`） | 原量纲 | 1e-12 |
| `steps_ratio`／`wall_ratio` 摆幅（`100 * (v - 1)`） | pp | **1e-14** |

两者相差 100 倍。同一个"0 个 identical"在原量纲上成立，不代表在 pp 刻度上也成立。`across_scale(samples, ref=1.0)` 一次给出两个刻度下的计数，用于说明差异（不是用来统一口径——各脚本各自归档、各自 `--verify`，口径锁定不改）。

**盲区扫描扩到六份归档。** 第七批只核了 `ledger.json`。本批扫 `ledger.json`、`wall-threshold-band.json`、`resolution.json`、`bucketing-probe.json`、`real-lengths-summary.json`、`legacy-group-arm-check.json` 的每一个 float（共 1907 个），对 ref=0 与 ref=1 各查 `0 < |v-ref| < TOL`：**全部 0 命中**。

**旧版实现改为从 git 取。** 第七批把旧版手写实现嵌在对照脚本里，另留了三条 `/tmp/old_*.py` 常量——那三个文件是临时产物，换台机器就不存在。现在 `old_source_at(relpath)` 直接 `git show <TOOL_COMMIT>^:<path>`，核对不再依赖任何人先跑过某个命令。
