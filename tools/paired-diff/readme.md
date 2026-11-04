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
pd.sign_regime(diffs)            # 四档：stable／mixed／consistent_but_thin／no_resolution
pd.summarize_paired(diffs, scale=100)   # 上面这些 + sd/min/max
pd.exceeds_band(effect, band)    # 判读门槛：|效应| > 基线自身抖动？
pd.unresolved(diffs)             # 这一格是不是根本没分辨率
pd.span(values)                  # 极差：换阈值/换顺序差多少
pd.sign_flip_around(values, 1.0) # 比率读数是否穿过 1
```

自检：`python3 tools/paired-diff/paired_diff.py --selftest`

判读三步（次序不能换）
---

1. **`n_identical` 有没有吃掉大半种子** —— 两臂读数完全相同不是"一致"，是这一格可能没有分辨率（低积压时 `R` 本来就接近 0，两臂一起落在 0 上）。
2. **`same_sign_fraction`** —— 均值是不是靠少数种子撑起来的。
3. **`|效应| > 带宽`** —— 效应是否大到无法用基线自身抖动解释。

三步任一不过就只能写"方向未定"。`exceeds_band` 是判读门槛，**不是显著性检验**：它没做 CI 或置换。

### 第 1 步的一处过度概括（2026-11-04 第十批订正）

原来的第 1 步把"重复比例高"直接判成"没分辨率"，且只给"方向未定"一个出口。这不成立：**表态的种子若方向一致，高重复说明的是"方向稳定、分辨率薄"，不是"方向混杂"**——两者都不能主张效应量，但写成同一句话会读错机制。

拆开要靠两个量（`summarize_paired` 已一并给出）：

| 量 | 含义 |
|---|---|
| `resolve_active_fraction` | 有多少比例的种子**在表态**（非 identical），即 `n_active / n` |
| `same_sign_fraction_active` | **只在表态种子**里的多数方向占比；没有种子表态时返回 `None` |
| `same_sign_fraction`（全列） | 全列做分母，恒 **≤** 表态口径——identical 只会把读数往下压 |

`sign_regime(diffs)` 把两者压成一个档名，已直接写进两份探针归档：

| 档 | 条件 | 能说什么 |
|---|---|---|
| `stable` | 表态占比 ≥ 0.5 且表态种子里同号 | 可主张方向（效应量仍要过带宽） |
| `mixed` | 表态占比 ≥ 0.5 但表态种子里不同号 | 真正的方向混杂 |
| `consistent_but_thin` | 表态占比 < 0.5 但表态种子里同号 | **分辨率不足**；不是"混杂"，也不是"不稳" |
| `no_resolution` | 没有种子表态，或表态的自己也混杂 | 什么也不能说 |

门槛 `min_active=0.5`、`min_same_active=0.8` 是**判读门槛不是显著性检验**，与 `exceeds_band` 同性质。

**真实负载上的差别不是修辞**（2026-11-04 第十批，数字取自两份归档，可直接复核）：

- `real-lengths-summary.json`（30 种子，§13.30 第五批那张 16 格表）：带重复（`n_identical ≥ 1`）的是 **8 格**（`stable` 3／`mixed` 2／`consistent_but_thin` 2／`no_resolution` 1），无重复的 8 格是 `stable` 6／`mixed` 2。按旧口径只看全列 ssf，那 8 格会被一律读成"符号混杂"。
- `bucketing-probe.json`（20 种子，16 个真实格）：`stable` **8**／`mixed` **5**／`consistent_but_thin` **2**／`no_resolution` **1**；带重复的 6 格里表态种子同号的是 3 格。

最典型的是 `qwen15b_cap256` 积压 2（30 个种子里 27 个重复、表态的 3 个**全同向**）与 `qwen05b_cap256` 积压 4（19 个重复、表态的 11 个里 10 正 1 负），它们都属**分辨率不足**而不是"方向混杂"。**结论本身不变**——这些格子要么效应量落在带宽内、要么带宽退化为 0，仍不可引用；变的是**判词**，不能拿它们当"符号不稳"的证据。

**第二个要防的坑**：带宽为 `0.0` 不一定是"基线很稳"。`qwen15b_cap256` 积压 2 的 `fifo_band` 中位／均值／最大全是 0，因为低积压时该格 `R` 本就恒为 0——**噪声为零的带宽不构成门槛**，`|效应| > 0` 在这里没有判读意义。三档诊断（`no_reading`／`degenerate`／`usable`）在 [band_floor.py](../../experiments/rollout-group-split-tradeoff/band_floor.py)，真实档 16 格里 2 格 `no_reading`、1 格 `degenerate`、13 格 `usable`。

**这个坑已经变成代码拒绝（2026-11-04 第十四批）**：`exceeds_band(effect, 0.0)` 现在**抛 `ValueError`**，不再恒真。带宽为 0 有两种成因——`baseline_stable`（抽样里确实有非零读数，只是每次换序结果相同：这时"效应 > 0"可以说）与 `no_reading`（每个种子的读数都恒为 0，样本结构使其必然如此：这一格问不出来）——判词相反，而 `exceeds_band` 只看得到那个 0。要断言一个为 0 的带宽，用 `band_floor_pass(effect, band, zero_kind=...)`：

| 调用 | 返回值 |
|---|---|
| `band_floor_pass(0.02, 0.0, zero_kind='baseline_stable')` | `True` |
| `band_floor_pass(0.02, 0.0, zero_kind='no_reading')` | `False`（问不出来不是证据） |
| `band_floor_pass(0.02, 0.0)`（未声明） | `ValueError` |
| `band_floor_pass(1.5, 1.0, ...)` | 与 `exceeds_band(1.5, 1.0)` 逐位相同 |

为什么不在工具里自动判断：成因只有**产生带宽的那一层**知道（要数逐种子的读数是不是全零），工具收到的只是一个数。所以工具这边能做的是**拒绝猜测**——这是"判据留在调用方"与"工具层不静默"之间唯一能同时满足的写法。

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
| `experiments/rollout-group-split-tradeoff/band_floor.py`／`sampling_reach.py` | `band_floor_rule`（`degenerate` 的过半判据，与 `fifo_band_floor` 同一份） |
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

> **⚠ 2026-11-04 第十六批（C 类账记作第十三批）：`band_floor` 那一侧的"过半"同样随单位数移动，而且**不单调**。**
>
> 3 个种子时门槛取 2，2/3 与 3/3 都在过半数侧且两侧都算 ⇒ 掷硬币输入 **100%** 判 `degenerate`，退到 `usable` 只需 1 个种子不同号；4 个种子 62.5%、5 个又回 100%、30 个是 **85.6%**。⇒ **"种子越多判据越稳"在过半判据上是错的**；单位数小到门槛恒被越过时，档名的误判率比档名本身更有信息。`no_reading` 一侧与单位数无关（n 个种子全零 ⇒ `2^-n`）。
>
> 同批把那个门槛的**四处抄本收成一份**：`fifo_band_floor`（归档用）、`band_floor.py` 的 `band_floor_verdict`（自检）、`sampling_reach.py` 的副本、`calibrate_thresholds.py` 的合成场景此前各写一遍 `tally * 2 > n`，于是 `band_floor.py` 的 `crosscheck` 比的是两份抄本、改一处不改另一处照报 `ok`。现在统一到：
>
> ```python
> pd.band_floor_rule('degenerate')(16, 30)   # True  —— 与原式 tally*2 > n 逐点相同
> pd.band_floor_rule('usable')(15, 30)       # True  —— 同一 passes 取反，不是另写一份
> pd.band_floor_rule('no_reading')           # ValueError —— 那一档不是"过半"，不在这里
> pd.falsy_rate(3, passes=pd.band_floor_rule('degenerate'), two_sided=True)['consistency_rate']  # 1.0
> ```
>
> 四处改接后三份归档 `--verify` 与 16/16 `crosscheck` 逐字段不变。详见 [C 类账 readme 第十三批](../../experiments/rollout-c-class-ledger/readme.md)。

**带宽为 0 不等于"基线很稳"（2026-11-04 第十一批，落在实验层不在本工具）**
---

`exceeds_band(|效应|, 0)` 对所有非零效应一律返回 True，所以带宽为 0 的格子最容易被读成"效应显著超出基线抖动"。而 0 有两种成因，判词相反：

- **抽样结构性无读数**：这一格所有种子的基线读数都恰好相同（例如负载被截断压到同一个上限），打乱输入顺序也不会变。这不是"基线稳"，是**这一格问不出来**。
- **部分种子无读数**：过半种子的基线带宽为 0，有效带宽由剩下的少数种子撑起。

这两个量 `paired_diff` 自己看不见——带宽是调用方算出来的，工具只收到一个数。所以判据留在**产生带宽的那一层**（[`band_floor.py`](../../experiments/rollout-group-split-tradeoff/band_floor.py)，档名 `no_reading`／`degenerate`／`usable` 写进 `real-lengths-summary.json`）。本工具这边要守的是：**调用方传入 `band=0.0` 时必须同时说明这个 0 是哪一种**，否则"效应超出带宽"这句话没有内容。

**这三个档名的代价位也量了**（`calibrate_thresholds.py` 的 `band_floor` 一节）：`no_reading` 要求**全部** 30 个种子的 `R` 都为 0，只要有一个种子抽到非零就翻档——所以它很严，`usable` 很松。真实 16 格里 `usable` 的格子零种子计数从 0/30 到 4/30 不等，**引用时必须同时给 `n_r_identically_zero`，不能只报档名**。

**门槛到底误判多少（2026-11-04 第十批）**
---

前面几节说 `min_active=0.5`、`min_same_active=0.8`"是判读门槛不是显著性检验"。这句只交代了性质，没交代**代价位**。现在量了，见 [`calibrate_thresholds.py`](../../experiments/rollout-c-class-ledger/calibrate_thresholds.py)（`--json` 写 `threshold-calibration.json`，13 条守护测试在 `test_threshold_calibration.py`）：

| 已知真相（每格 4000 列 × 20 种子） | 判出 `stable` | 判出 `mixed` | 判出 `thin` | 判出 `no_resolution` |
|---|---|---|---|---|
| 每种子同向 | 1.000 | **0.000** | 0 | 0 |
| 明显混杂 | 0.018 | 0.982 | 0 | 0 |
| 表态少但同向 | 0 | **0.000** | 1.000 | 0 |
| 纯掷硬币（无信息） | **0.011** | 0.989 | 0 | 0 |
| 全零 | 0 | 0 | 0 | 1.000 |

**假阳为 0**：真同向／真 `thin` 的列从不被报成 `mixed`，所以把 `consistent_but_thin` 读成"方向混杂"的责任在引用方，不在门槛。**唯一的假阴是 1.1%**，而且它就是二项分布本身：`stable` 只要求表态种子里 ≥80% 同号，20 个硬币双侧 ≥16 个同号的概率是 1.18%，实测 1.1% 与之吻合。⇒ 这是「20 个种子」这个样本量给的分辨力上限，**压下去只能加种子，不能挪门槛**。

> **⚠ 2026-11-04 第十三批：那个「假阳为 0」有适用条件，别当成普适结论。** 上表的真值都设成"每个种子都明显同向"，效应约是噪声的 6 倍——**任何门槛都判得对**。补扫「真方向已知、真效应 = ratio × 带宽」后，`stable` 的判出率在 0.20／0.25／0.30／0.40／0.50 倍带宽处依次为 **0.09／0.15／0.37／0.82／0.99**：**效应不到带宽的 1/3 时，多数列会被报 `mixed`，而真相是方向全程一致**。真实归档的 |效应|／带宽 中位只有 0.76–1.17，正落在这段里。⇒ 读 `mixed` 前先看效应是不是小于带宽；两个量要**分开写**（方向看 `sign_regime`、量级看 `effect_exceeds_band`）。详见 [C 类账 readme 第十一批](../../experiments/rollout-c-class-ledger/readme.md)。

另注意：`min_same_active` 从 0.8 调到 0.55 时，那个假阴从 1.1% 跳到 **82.2%**——门槛对这类输入的敏感度是断崖式的，不是线性的。改动它之前先跑一次 `calibrate_thresholds.py` 看表。

> **⚠ 2026-11-04 第十五批（C 类账记作第十二批）：上面那 1.1% 是「20 个种子**全部表态**」的数字，而判据实际被用在 3–27 个种子上。**
>
> 上表六个场景全是 20 个种子，`stable` 的假阴 1.1% 因此只是**表态种子数 = 20** 那一个点。判据真实的单位数逐格不同：真实 16 格里表态种子只有 **3–27** 个（1.5B cap256 积压 2 只有 3 个）。同一个 0.8 门槛在 3 个种子上要"三个全中"，在 27 个上只要"≤5 个反"。把掷硬币输入喂进去，判 `stable` 的概率是：
>
> | 表态种子数 | 3 | 5 | 19 | 20 | 27 | 30 |
> |---|---|---|---|---|---|---|
> | 掷硬币判 `stable` | **0.250** | 0.375 | 0.0044 | 0.0118 | **0.0015** | 0.0014 |
>
> 两个数量级的差，还带**锯齿**（19 个要 16 个 = 0.842，18 个只要 15 个 = 0.833）。⇒ 报 `stable` 的可靠性必须同时报**表态种子数**；「假阳为 0」只在足够多的表态种子上成立。
>
> **这条已落成工具函数**：`falsy_rate(n_truth, passes=..., two_sided=...)` 用**调用方自己的判据**精确枚举判真概率（判据不在这里复写一份，否则会像 `band_floor`／`fifo_reorder_band` 各写一份种子式子那样漂）。`passes` 在 n 个单位上恒假时**抛 `ValueError`**——那说明问错了问题，给 0 会被读成"门槛很严"。
>
> ```python
> pd.falsy_rate(3,  passes=lambda t, n: t / n >= 0.8, two_sided=True)['consistency_rate']  # 0.25
> pd.falsy_rate(27, passes=lambda t, n: t / n >= 0.8, two_sided=True)['consistency_rate']  # 0.001514
> ```
>
> `two_sided` 必须由调用方声明：`stable` 的"同号占比"和 `band_floor` 的"过半"两侧都算，`no_reading` 的"全部为零"只算一侧。**本函数第一版想自动判双侧，结果 n=20 时镜像侧恰好差一个种子不通、判成单侧，把误判率少算了一半**（精确 0.0059 vs 蒙特卡洛 0.0117）——蒙特卡洛复核抓到的，现在 0.0118 vs 0.0114 对上了。详见 [C 类账 readme 第十二批（全文批号第十五批）](../../experiments/rollout-c-class-ledger/readme.md)。

