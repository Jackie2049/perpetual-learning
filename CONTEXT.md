更新时间：2026-11-04（M1 **第十一批**：给 `fifo_band` 加三档准入——**带宽为 0 有两种成因**，此前归档里分不出来。第五批只记了一句「噪声为零的带宽不构成门槛」，没有变成判据，于是 `fifo_band = 0` 的格子看起来和「基线换序完全不动」一样，而后者恰是最强的「效应显著」信号。新增 `no_reading`（每个种子 `R` 都恒为 0 ⇒ 问不出来）／`degenerate`（过半种子带宽为 0）／`usable`，16 格实测 **13／1／2**：`no_reading` 是两个模型 cap256 积压 2，`degenerate` 是 0.5B cap256 积压 4（22/30 带宽为 0）。机制是**截断贴顶**——cap256 组最大 **86.0%（1.5B）／94.2%（0.5B）** 恰等于上限（cap1024 只 0.2%／4.8%）。**不改任何已发布读数**，唯一收窄是 0.5B cap256 积压 4 的 `stable` 建在 `degenerate` 带宽上。旧字段相对原始 HEAD **0 变化 0 删除**、新增 96 个。另记 `exceeds_band(|效应|, 0.0)` 恒真、而工具层分不出是哪种 0 ⇒ 判据只能留在产生带宽那一层。M1 03:1x 第十批：同号占比分母拆开、四档 `sign_regime` 进两份归档，并修掉第九批 `compare_old_new_impl.py` 的假正 PASS（收尾处 `KeyError`，那句 PASS 从没打印出来）。）

## 当前现场

> **只读最上面一节 `### M1 本轮` 即可恢复现场。** 下面 20+ 个 `###` 是历史批次叙述
> （按时间倒序）——**它们已在 `diary/` 与专项文档留档，此处仅为交接连续性的冗余副本**。
> 判断"现在该做什么"看 `## 未完成操作与并行工作` 与两个队列，不要从头读到尾。

### M1 本轮（2026-11-04 04:1x）：第十一批——带宽为 0 有两种成因，`fifo_band` 加三档准入

**一、为什么补这一层。** 第五批给出带宽判读口径 `|效应| > 带宽` 后，自己留了一句"噪声为零的带宽不构成门槛"——但那只是提醒，没有变成字段。归档里 `fifo_band = 0` 的格子于是和"基线换序完全不动"**长得一样**，而后者恰是最强的"效应显著"信号。新增 [`band_floor.py`](experiments/rollout-group-split-tradeoff/band_floor.py) 三档：`no_reading`（**每个种子**的 FIFO `R` 都恒为 0 ⇒ 这一格问不出来）／`degenerate`（过半种子带宽为 0）／`usable`。

**二、16 格实测 13／1／2（按归档逐格核过，与独立脚本 16/16 一致）。**

| 负载 | 2 批 | 4 批 | 8 批 | 16 批 |
|---|---|---|---|---|
| 1.5B cap256 | **`no_reading`** | usable（4/30 带宽为 0） | usable | usable |
| 0.5B cap256 | **`no_reading`** | **`degenerate`**（22/30） | usable | usable |
| 1.5B／0.5B cap1024 | usable | usable | usable | usable |

**三、机制是截断贴顶，不是"基线稳"。** cap256 的组最大 **86.0%（1.5B）／94.2%（0.5B）** 恰好等于 `max_response`（cap1024 只 0.2%／4.8%）⇒ 抽 16 组时组最大全等的概率约 0.09／0.38 ⇒ 组间齐平、任何顺序都无事可排，`R ≡ 0`。这是第三十九批"组最大 cv 仅 0.07、组级排队无事可排"的同一个几何事实的另一个后果。

**四、不改读数，改理由。** cap256 六格仍无一格可主张方向（第五批结论与第十批判词都不变）；唯一收窄是 **0.5B cap256 积压 4 的 `stable` 建在 `degenerate` 带宽上**——方向稳定性可引用，效应量不可引用。

**五、判据归属。** `exceeds_band(|效应|, 0.0)` 对所有非零效应返回 True，而带宽是调用方算出的一个数，`paired_diff` 收到时已分不出这个 0 是哪种 ⇒ 判据只能留在产生带宽的那一层；工具 readme 只记一条要求"传 `band=0.0` 时必须说明是哪种 0"。

**核对产物**：`real-lengths-summary.json` 相对原始 HEAD **旧字段 0 变化 0 删除**、新增 96 个；`band_floor.py` exit 0（16/16 `crosscheck: ok`）；负向对照三次注入——压成恒 `usable` ⇒ 4 条红、`n_r_zero` 恒真 ⇒ 3 条红、**过半边界 `>`→`>=` ⇒ 5 条全绿（当时无边界用例）**，同批已由并行会话补 `test_exactly_half_zero_band_is_still_usable`／`test_one_more_zero_band_flips_to_degenerate` 两侧钉住。只读仿真重算、未跑 GPU、未触上游。

**推送**：`f685d58f9` 已推，`origin/main..main` 归零（直连一次成功）。

**留给下一轮**：① `no_reading` 只说"这一格问不出来"，**未回答"换一种抽样（放宽 `n_groups`／换负载档）能否问得出来"**——要答需另立实验；② 四档 `sign_regime` 门槛（0.5／0.8）与三档 `band_floor` 门槛（0.5）都未做假阳率模拟校准，都是判读门槛不是显著性检验；③ 并发重算前先 `ps`（三进程并发把 10 分钟拖到 48 分钟）；④ 语义面剩余缺口仍是长度预测误差／分桶／迁移／抢占／KV 搬运，P-1 跨系统需用户立项。

入口：[实验 readme 第十一批](experiments/rollout-group-split-tradeoff/readme.md) · [band_floor.py](experiments/rollout-group-split-tradeoff/band_floor.py) · [§13.30 第五批](notebook/rollout-performance-research.md) · [paired-diff](tools/paired-diff/readme.md) · [TS-001](queue_tech-secretary.md) · [TM-011](queue_tech-miner.md) · [diary](diary/2026-11-04.md)。

### M1 上一轮（2026-11-04 03:1x）：第十批——同号占比分母拆开、四档 `sign_regime`；并修好上一批遗留的崩溃与三处计数错

**一、开工时工作区有一份未提交、无日志的改动**（`paired_diff` 新增 `resolve_active_fraction`／`same_sign_fraction_active`，两个探针脚本接上），先核实再留用。核实成立（当时 163 测试全过），但它依据的读法**过度概括**：`same_sign_fraction` 分母里混着 identical 种子，"低 ssf ⇒ 方向混杂"只对一半——同一个低值也可能是"方向一致、只是没分辨率"。

**二、四档 `sign_regime()` 进归档。** `stable`／`mixed`／`consistent_but_thin`／`no_resolution`，门槛 `min_active=0.5`、`min_same_active=0.8`（**判读门槛，非显著性检验**）。每格 3 个新键：`bucketing-probe.json` **+60 字段**（20 格）、`real-lengths-summary.json` **+48 字段**（16 格），**两份旧字段 0 变化 0 删除**，两份 `--verify` exit 0。

**三、按归档实测的分布（引数时照抄这里，不要手拼计数）。**

| 归档 | 种子数 | 分档 |
|---|---|---|
| `real-lengths-summary.json` | 30 | 16 格：`stable` 9／`mixed` 4／`consistent_but_thin` 2／`no_resolution` 1；带重复（`n_identical ≥ 1`）**8 格**（`stable` 3／`mixed` 2／`thin` 2／`no_resolution` 1） |
| `bucketing-probe.json` | 20 | 16 个真实格：`stable` 8／`mixed` 5／`consistent_but_thin` 2／`no_resolution` 1；带重复 6 格中 3 格表态种子同号 |
| 其中 cap256 八格 | 30 | `stable` 3／`mixed` 2／`thin` 2／`no_resolution` 1 |

**四、§13.30 第五批的结论不变、判词改了。** cap256 八格效应量全部落在带宽内或带宽退化为 0，**无一格可主张方向**；但不能再把这些格子当"符号不稳／方向混杂"的证据——`qwen15b_cap256` 积压 2（27/30 个种子两臂相等、表态的 3 个**全同向**）与 `qwen05b_cap256` 积压 4（19 个相等、表态 11 个里 10 正 1 负）属**分辨率不足**。cap1024 六格（28–30/30 同号、方向 `R` 变大）不受影响。

**五、修掉上一批遗留的三处错。** ① **已提交的 `compare_old_new_impl.py` 在收尾处 `KeyError` 崩掉**——第九批那句"PASS"从没打印出来（`identical` 两版都有，出处不同：旧版 `summarize` 自己数、新版取容差档；`seed_sign_stability` 的返回值里没有这个键）。已修并把"两版相等"纳入 PASS/FAIL，注入 off-by-one 实测 FAIL。② 三处文档（本条头、TS-001、日志）把 30 种子与 20 种子两个口径的"几个格子"混写，已按上表改写。③ `bucketing_probe.py` 的 `regime` 曾是手写一行式（档名 `matched` 不在词表里、且只看全列 ssf），已统一走 `paired_diff.sign_regime()`。

**六、一条口径（第十批新增）**：带宽为 `0.0` 不一定是"基线很稳"。低积压时该格 `R` 恒为 0，`exceeds_band` 对所有非零效应一律通过。并列会话已建 `band_floor.py`（`no_reading`／`degenerate`／`usable` 三档），**已于上一轮（第十一批）审过判据并接进归档**——见最上面一节。

**核对产物**：两目录 **171 passed**；两份归档 `--verify` exit 0；`paired_diff --selftest` 通过；`compare_old_new_impl.py` PASS。只读仿真重算、未跑 GPU、未触上游。

**留给下一轮**：① ~~`band_floor.py` 的三档门槛未审~~ **已于第十一批审过并入档**（见最上面一节）；② 四档门槛 `0.5`／`0.8` 未做假阳率模拟校准；③ 三个 `real_lengths_policy.py` 进程曾同时重算（本会话与并行会话各起一到两个），把单次约 10 分钟拖到近一小时——**并发重算前先 `ps` 看一眼**。

**本轮已补上并发遗留**：`fifo_band_floor` 接进源码后归档没跟着重算（源码与归档不同步），本会话已于 04:1x 重算完成——`real_arms.*.sign_stability.*.band_floor` 现在在归档里，16 格分布为 `usable` 13／`no_reading` 2／`degenerate` 1，**旧字段相对原始 HEAD 0 变化 0 删除**。

入口：[paired-diff](tools/paired-diff/readme.md) · [实验 readme](experiments/rollout-group-split-tradeoff/readme.md) · [§13.30](notebook/rollout-performance-research.md) · [TM-011](queue_tech-miner.md) · [TS-001](queue_tech-secretary.md) · [diary](diary/2026-11-04.md)。

### M1 上上轮（2026-11-03 22:2x）：推送恢复；补齐非计数输出的逐值对照与盲区扫描

**一、推送恢复。** 开工时 `git fetch`／`git push` 直连均通，积压的 **16 个提交**一次推完（远端 main `878600d56`），`git log origin/main..main` 归零。此前 `## 未完成操作` 里的"两条通道都不通"已作废。

**二、第七批遗留三条全部做完。** ① `same_sign_fraction`／`sd_pp` 逐值对照：真实 96 + 合成 210 + 8 条构造舍入列零分歧，两次注入（`sd_pp` scale 100→10 → 192 处分歧；`ssf` 恒 1.0 → 45 处）被测红；② 盲区扫描扩到**六份归档、1907 个 float**，ref=0 与 ref=1 各查 `0 < |v-ref| < TOL`，**0 命中**，负向对照（种进 0.9999999999995）立即报出；③ 旧版实现改为 `git show 038fc2e3b^:<path>`，去掉三条 `/tmp/old_*.py` 常量。

**三、量出两条口径。** 第 3 位小数用内建 `round`，对精确平局**取偶**（`0.3125 → 0.312`，人类读法是 0.313）——真实输入落在平局上的个数实测为 **0**，所以不影响任何已发布读数，但已写成断言；`TOL` 是绝对容差，同一列比率在原量纲与 pp 刻度上容忍的偏差**差 100 倍**，两刻度的 `n_identical` 不能互相解释（`across_scale()`）。

**四、本批的自我纠正（写 readme 时才发现的）**：想写"96 列真实输入零分歧"，核了一下其中 **40 列是整数步号**——旧版 `d == 0` 与新版 `|d| <= TOL` 在整数输入下精确等价，**那 40 列根本没检验容差**。真正检验容差的只有 56 列浮点比率，相对 1 翻号的只有 2 列。已把覆盖面直接打进脚本输出。

**核对产物**：`test_cross_script_consistency` 13 → **23 passed**；两目录合计 **144 passed**；`paired_diff --selftest` 通过；六份归档 `--verify`（含约 10 分钟真实负载重算）**逐字段一致**。只读归档、未跑 GPU、未触上游。

**留给下一轮的三条**：① 真实输入"无舍入平局"只实测了 96 列 + 8 条构造列，**新数据出现平局时第 3 位取决于取偶还是四舍五入**，建议在写进正文的读数旁标注；② 两刻度差异在既有数据上全为 0，**只说明口径存在、不说明已发生**，若将来某份归档出现 `raw` 与 `pp` 计数不同，需要先判断是哪一侧写错了刻度；③ `compare_old_new_impl` 的真实列仍只覆盖 p3／p4 的 C 类账，**分桶探针与真实负载脚本的真实输入列仍未纳入逐值对照**（第七批记为遗留③的另一半，本批只做了"从 git 取"这部分）。

入口：[两版对照](experiments/rollout-c-class-ledger/compare_old_new_impl.py) · [跨脚本测试](experiments/rollout-c-class-ledger/test_cross_script_consistency.py) · [readme 第八批](experiments/rollout-c-class-ledger/readme.md) · [paired-diff](tools/paired-diff/readme.md) · [TS-001](queue_tech-secretary.md) · [diary](diary/2026-11-03.md)。

### M1 上一轮（2026-11-04 02:0x–02:5x）：先核实工作区那份未提交的拆分，再修掉判据里的过度概括（第十批前半）
**一、开工时的处境。** 工作区有 8 个文件未提交（`paired_diff` 新增 `resolve_active_fraction`／`same_sign_fraction_active`，两个探针脚本接上、两份归档新增字段），**无提交、无 CONTEXT、无日志**。按规则先核实能否成立再决定留用或回退，不接着做未核实的东西。

**二、核实成立，但读法过度概括。** 163 测试（分桶 69＋C 类 94）与 `--selftest` 全过。但原读法「全列 ssf 低 ⇒ 方向混杂」只对一半：分母里混着 identical 种子，**"方向一致、只是没分辨率"会得到同样的低分**。实测两份归档 36 格：真实档 16 格里 10 格带重复，**7 格的表态种子其实同号**（`qwen15b_cap256` 积压 2：30 个种子里 27 个相等、表态 3 个全同向；`qwen05b_cap256` 积压 4：19 个相等、表态 11 个里 10 正 1 负），只有 3 格真混杂。

**三、四档 `sign_regime()` 进归档。** `stable`／`mixed`／`consistent_but_thin`／`no_resolution`，门槛 `min_active=0.5`、`min_same_active=0.8`（判读门槛，非显著性检验）。`bucketing-probe.json` **+20 字段**、`real-lengths-summary.json` **+16 字段**，两份都 **0 改动 0 删除**，`--verify` 逐字段相同（后者含约 10 分钟真实负载重算）。

**四、结论不变、判词要改。** cap256 六格是 2 格 `consistent_but_thin`、2 格 `mixed`、2 格 `stable`、另 1 格 `no_resolution`（0.5B 积压 2）；**效应量全部落在带宽内或带宽退化为 0，没有一格能主张方向**——但**不能把前四格当"符号不稳"的证据**。cap1024 六格（28–30/30 同号、方向 `R` 变大）不受影响。

**五、顺手抓到的坑。** `qwen15b_cap256` 积压 2 的 `fifo_band` 中位／均值／最大**全是 0**（低积压时该格 `R` 恒为 0）⇒ **噪声为零的带宽不构成门槛**，只看 `exceeds_band` 的布尔值会把这一格读成"效应超带宽"。

**核对产物**：四处注入全部被测红后恢复（含 `compare_old_new_impl` 的 off-by-one：把 `identical` 判据改成恒真即 FAIL；该脚本此前在收尾处取 `seed_sign_stability()["n_identical"]` 会 `KeyError`，第九批的 PASS 是在那行之前打印的，`identical` 从未进入判定——第十批改成两版各按自己的判据数并纳入 PASS/FAIL）（`sign_regime` 档名、探针档名换成"全列 ssf<0.5 即 mixed"）；写测试时自己踩了一次边界（样本表态占比 0.533 被正确判成 `mixed`，是**断言写错不是实现错**）。`test_group_split_tradeoff` 54 passed、`test_real_lengths_policy` 18 passed、C 类 94 passed、`verify_anchors --verify` 34 一致／0 不一致、`paired_diff --selftest` 通过。只读仿真重算、未跑 GPU、未触上游。

**留给下一轮**：① 四档门槛（0.5／0.8）是判读门槛，**没有用已知真相的模拟校准过假阳率**——若将来要写进 survey 正文，需要先量"在真实同号率下这个门槛误判多少"；② `real_lengths_policy.policy_sign_stability` 的旧版签名仍喂不了任意列（第九批遗留①），本批又给它加了一个字段，**逐值对照的口子又宽了一点**；③ `compare_old_new_impl` 的覆盖面下限（真实列不足 40 即 FAIL）**仍只对 C 类账成立**，探针那一侧没有下限检查（第九批遗留②）。

入口：[paired-diff](tools/paired-diff/readme.md) · [sign_regime](tools/paired-diff/paired_diff.py) · [§13.30 第五批订正](notebook/rollout-performance-research.md) · [实验 readme](experiments/rollout-group-split-tradeoff/readme.md) · [TM-011](queue_tech-miner.md) · [diary](diary/2026-11-04.md)。

### M1 上一轮（2026-11-04 00:0x）：分桶探针与真实负载纳入逐值对照；两个探针脚本的容差接进共享工具

技术秘书侧无新用户任务；TM 系列对外动作仍全部待用户逐项授权。本轮接第八批自己写下的「留给下一轮」①②：分桶探针与真实负载脚本的真实输入列没有逐值对照。**不新开实验、不改任何测量值。**

**一、先让输入列能被取到。** `bucketing_probe` 原先把"产生逐种子行 → `summarize`"写死在 `synthetic_block()`／`real_block()` 里，归档只留汇总量。抽出 `iter_rows()` 生成器后两个 block 改为消费它，**`bucketing-probe.json` 前后逐字段相同**（`--verify` 通过）。对照脚本据此拿到 20 格／400 个种子对的配对差。

**二、两个探针也对照完了。** `bucketing_probe` 20 格 + `real_lengths_policy.fifo_reorder_band` 8 格，新旧两版**全部零分歧**；每格 9 个非计数键（`identical`／`abs_pp_mean`／`median`／同号占比／sd／极值）一致。**`identical` 两版都是 60，必须相等**——旧版由 `summarize` 自己数（`abs(d) < 1e-12`），新版取 `paired_diff` 的容差档；旧版 `seed_sign_stability` **内部**没有这个键，但那不等于旧版没有这一档（第十批核出并订正，此前写成"旧版没有、新版补了"是错的）。

**三、抓到一个真缺口：两个探针脚本里还有硬写的容差。** `bucketing_probe.summarize` 的 `abs(r['abs']) < 1e-12`、`real_lengths_policy.arrival_assumption_shift` 的两列同侧判定都是字面量 ⇒ **把 `TOL` 注入成 1e-3，这两个脚本的读数根本不动**。第六批那条"注入被测红"只覆盖了直接调用共享函数的那一层。两处已改为 `paired_diff.TOL`，补 `TestToleranceSourceNotHardcoded`（源码无裸 `1e-12` + 模块级 `TOL` 改松后 `summarize` 的 `identical` 跟着变）。

**四、本批的错误都在对照脚本自己身上，前三次被"零分歧"掩盖**：① `n_pairs` 恒 1（整列塞进一条曲线）；② 曲线首元素取 0.0 ⇒ `first_cross` 恒返回 0，正负全被压平，而测试读的是 `n_ties`（两侧都是 0 ⇒ 自己和自己比）；③ `old_sign_counts_real` 把 `n_identical` 写死 0，导致每个含并列的列都被报成分歧（34 列）；④ `probe=True` 分支一开始直接 `return []`，那一侧的"零分歧"没有内容。⇒ 判据写进 readme：**对照脚本必须先用"已知答案"的输入自证**（`[0,1,-1,0,2]` 必须给 `n_pairs=5`、`n_ties=2`、正号占比 0.4），再谈真实数据。

**核对产物**：`test_cross_script_consistency` 23 → **35 passed**；六份归档 `--verify` 全部逐字段一致（含 `real_lengths_policy --verify` 约 10 分钟真实负载重算）；`compare_old_new_impl.py` **PASS**。只读重算、未跑 GPU、未触上游。

**留给下一轮**：① `real_lengths_policy.policy_sign_stability` 旧版签名要排程参数、喂不了任意列，本批用"同一段计数逻辑"间接覆盖而未逐值；② 对照脚本已有自证用例，但**覆盖面下限（真实列不足 40 即 FAIL）只对 C 类账成立**，探针那一侧还没有下限检查；③ M1 这条线（纯归档重算 + 判据一致性）已到边际，语义面剩余缺口仍是长度预测误差／分桶／迁移／抢占／KV 搬运，P-1 跨系统需用户立项。

入口：[readme 第九批](experiments/rollout-c-class-ledger/readme.md) · [两版对照](experiments/rollout-c-class-ledger/compare_old_new_impl.py) · [跨脚本测试](experiments/rollout-c-class-ledger/test_cross_script_consistency.py) · [bucketing_probe](experiments/rollout-group-split-tradeoff/bucketing_probe.py) · [TS-001](queue_tech-secretary.md) · [diary](diary/2026-11-04.md)。

### M1 上一轮（2026-11-03 21:0x）：归档不变 ≠ 两版实现相同——补做逐值对照与跨脚本一致性

技术秘书侧无新用户任务；TM 系列对外动作仍全部待用户逐项授权。本轮接第六批自己写下的两条「留给下一轮」：① 共享判据缺**跨脚本一致性**测试；③「抽完没有回归」只有归档比对这一条证据，**两版实现的逐值对照还没做**。两条都是纯 CPU 重算，不新开实验、不改任何测量值。

**一、为什么归档不变不够。** 五份归档 `--verify` 逐字段相同，只说明「既有数据没落进两版判据的分歧区」；若某个脚本在容差边界上本来就会与旧版分岔、而既有数据恰好没落进那条缝，归档比对看不出来。这正是「改正一处能带动另一处」这个初衷所要求的证据缺口。

**二、两版实现逐值对照（`compare_old_new_impl.py`）。** 同一列输入分别喂旧版手写实现与共享库。真实列**从源 summary 逐种子重算**（`wall-threshold-band.json`／`ledger.json` 只留计数与汇总量，反解不出逐种子差值）：**96 列／2480 个差值／332 个 identical 种子，零分歧**；合成边界 210 列、极差与翻号 200 组，同样零分歧。对照自身做过注入（正号计数偏 +1 → FAIL，清除后 PASS）。

**三、跨脚本一致性测试（`test_cross_script_consistency.py`，13 条）。** 钉住：同一列差在各调用点计数逐字段相同；**比率必须平移后再交给 `sign_counts`**（相对零的分类，直接喂 0.978 会记成「正」）；整数步号的 `d == 0` 与浮点比率的 `|d| <= TOL` 等价但不可共用键名。另记一条盲区：距 1 不到 `TOL` 的比率在重新定基时被吞成 0、记成 identical——**既有归档 353 个比率无一落进盲区**（3 个恰为 1.0，都正确记为 identical）。

**四、本批两个错误都在对照脚本自己身上**：① 第一版从归档反解真实列，读出「**0 列、PASS**」——没扫到东西的 PASS 什么也没证明，已加「真实列不足 40 即 FAIL」的覆盖下限；② **p3 的基线是 `R_nofilter` 不是第一个条目**，第一版拿 `R` 当基线等于自己比自己，五档全读出 `30/30 identical` 的假象，对照 `ledger.py:259` 才修好。第三处：注入脚本又一次改错行（`str.replace("TOL = 1e-12")` 命中 docstring 那句），改成按 `line.startswith("TOL = ")` 定位真实赋值行后才抓到——**这是同一坑第三次**。

**核对产物**：两文件 **134 passed**（本文件新增 13 条）；`ledger.json`／`wall-threshold-band.json` 两处 `--verify` 逐字段一致；`paired_diff.py --selftest` 通过。**零归档改动**。只读既有归档与源 summary、未跑 GPU、未触上游。

**⚠ 推送仍未恢复**：本轮开工时 `git ls-remote` 仍报 `SSL certificate problem: self signed certificate`，`gh api` 报 `tls: handshake failure`——**两条通道都不通**。按规则不用 `--insecure` 绕过、不反复重试。暂存前积压 13 个提交，本轮提交后为 **14 个**（`git log origin/main..main` 实测）。

**留给下一轮的三条**：① 跨脚本一致性目前只覆盖「计数」这一层，**`same_sign_fraction` 与 `sd_pp` 的舍入**还没做逐值对照；② 盲区那条（`|v-1| < TOL` 被吞）只在 `ledger.json` 上核过，**另外三份归档还没扫**；③ `compare_old_new_impl.py` 只对照了 p3／p4 的 C 类账，**分桶探针与真实负载脚本的判据没有逐值对照**，且旧版实现是硬编码 `/tmp/old_*.py` 临时路径，应改成从 git 直接取。

入口：[两版对照](experiments/rollout-c-class-ledger/compare_old_new_impl.py) · [跨脚本测试](experiments/rollout-c-class-ledger/test_cross_script_consistency.py) · [readme 第七批](experiments/rollout-c-class-ledger/readme.md) · [paired-diff](tools/paired-diff/readme.md) · [TS-001](queue_tech-secretary.md) · [diary](diary/2026-11-03.md)。

### M1 上一轮（2026-11-03 20:0x–20:3x）：判据抽成共享工具；`wall_ratio` 也建在同一个 0.9 上

技术秘书侧无新用户任务；TM 系列对外动作仍全部待用户逐项授权。本轮接上一轮自己写下的两条"留给下一轮"：② 同一族判据在四个脚本各写一遍、该抽共享工具（第四十二批起一直在记）；③ `wall_ratio` 的同水平拆解未做。两条都是纯 CPU 重算，不新开实验、不改任何测量值。

**一、抽出 `tools/paired-diff`。** `effect_exceeds_band`、阈值摆幅、`n_identical`、`same_sign_fraction` 四族判据收成共享库，五个脚本改接。**验收线是归档逐位不变**：`ledger.json`、`bucketing-probe.json`、`real-lengths-summary.json`、`summary.json`、`legacy-group-arm-check.json` 五份 `--verify` 全部逐字段相同；测试 113 → **121 passed**。

**二、`wall_ratio` 的阈值拆解（`wall_threshold_band.py`）。** 先做自校验——t=0.90 上重算必须逐位复现 `ledger.json` 的 `wall_ratio`（8 臂 |差| < 1e-12），过了才读五档。八臂摆幅 **2.2–9.5pp**（`A vs R` 9.5、`freq` 9.3、`cancel` 8.2、`B vs R` 7.9、`C vs B` 7.4、`C vs R` 4.5、`cap` 3.9、`replay` 2.2）。**总量比穿过 1.0 的有两臂**（`replay` 1.0085→0.9913、`C vs B` 1.0436→0.9701），但**逐种子看只有 `C vs B` 方向真翻**（0.85 上 20 慢/9 快 → 0.95 上 13 慢/17 快）；`replay` 逐种子 18–20/30 同向，方向稳、只是被个别种子的窗口长度放大；`cap` 有 **14–23/30** 种子两臂窗口成本恰好相等 ⇒ 无分辨率，不是不稳。`gap` 摆幅全部来自 `wall_ratio`（`cost_ratio` 与阈值无关），`freq` 最大 **0.335**。

**三、两个错误（都已修，判据已写进文档）**：① **抽工具不是零风险搬迁**——`threshold_band` 的翻号参考点是 **1 不是 0**，机械换成对零的 `sign_flip` 直接把 p3 `A` 的翻号读丢（0.978→1.005 对零同侧），被**现有**测试 `test_only_p3_A_flips_sign` 拦下。② **容差挪了 10 个数量级、113 个测试全绿**——注入 `TOL=1e-3` 无任何失败，因为既有测试的配对差一律取 0 或明显非零，从没造出"小于 1e-3、大于 1e-12"的差值；补 `TestSharedTolerance` 三条边界测试后才被测红。顺带查出 `def f(tol=TOL)` 把值绑进默认参数、改 `TOL` 不生效，已改成调用时解析。另有一次**注入脚本自身的问题**：`str.replace(...,1)` 改的是文档字符串里相同的那句话、常量没动，导致"注入没抓到"是假结论。

**核对产物**：**121 passed**（新增 `TestSharedTolerance` 3 条 + `test_wall_threshold_band` 5 条）；共享工具 4 组注入、wall band 3 组注入**全部被测红**，恢复后全过；五份归档 `--verify` 逐字段相同；`wall_threshold_band.py --verify` 8/8 参考阈值复现。只读既有归档、未跑 GPU、未触上游。

**⚠ 推送仍未恢复**：20:20 实测 `git ls-remote` 仍报 `SSL certificate problem: self signed certificate`（中间设备替换证书），按规则不用 `--insecure` 绕过、不反复重试。本地积压 **13 个提交**（含并行会话的 `60fbf6939`）。

**留给下一轮的三条**：① 共享判据还缺一条**跨脚本一致性**测试——四个脚本各自调 `paired_diff`，但没有测试断言"同一列差喂给两个脚本得到相同计数"，改一处口径不会立刻被另一处发现，与抽工具的初衷正好相反；② `level_matched_wall` 还只报了个中位数比，**墙钟差里"步数贡献／每步成本贡献"在阈值维上没拆开**；③ 五份归档字段全部不变是好事，但"抽完没有回归"目前只有归档比对这一条证据，**两版实现的逐值对照还没做**。

入口：[§13.47.2](notebook/rollout-performance-research.md) · [readme 第六批](experiments/rollout-c-class-ledger/readme.md) · [wall_threshold_band.py](experiments/rollout-c-class-ledger/wall_threshold_band.py) · [归档](experiments/rollout-c-class-ledger/wall-threshold-band.json) · [test_wall_threshold_band.py](experiments/rollout-c-class-ledger/test_wall_threshold_band.py) · [paired-diff](tools/paired-diff/readme.md) · [TS-001](queue_tech-secretary.md) · [diary](diary/2026-11-03.md)。

### M1 上一轮（2026-11-03 16:4x–19:0x）：上一轮那个"未定位的失败"是判据错了；阈值摆幅拆开后，p4 的形状只值 0.3–3.1 步

技术秘书侧无新用户任务；TM 系列对外动作仍全部待用户逐项授权。本轮接的是上一轮自己写下的"**下轮必看**：先做最小复现"。结论是：**代码没错，判据错了**；顺带把阈值这一列的读数拆成两半，并把它正式接进 `ledger.py`。

**一、107.005 不是 bug，是那条判据的答案。** `first_cross` 找"**首次**达到阈值"的步号。曲线 0.80→0.88→**0.91**→0.96→0.99 在 0.9 上第 2 步到点；把到点之后压平到 0.91，此后任何更高阈值都再也跨不过去。压平按"到点那一步的值"取值，等于抹掉此后所有新高——**"压平后摆幅应为 0"在数学上不成立**。上一轮记的"同一段代码单独 `exec` 行为不一致"是排查时抄漏了输入变换。这条错误判据已用单测钉住。

**二、替代读数：同一水平上的步数差。** 逐种子算"到阈值步数(臂) − 到阈值步数(基线)"，就是 `steps_ratio` 分子分母真正比较的量。八臂五档：**p4 四臂只摆 0.3–3.1 步**（`freq` +0.30→+3.43、`cancel` −0.97→−0.57、`cap` −0.43→−0.73、`replay` −0.23→−0.50），而比率摆 1.7–18.2pp ⇒ `freq` 的"慢 2.4%→20.6%"里形状只值 0.3→3.4 步，其余是阈值挪动本身。p3 的量级大得多（`R` −6.1→−17.2 步、`C` −6.1→−10.9、`A` −1.3→+11.3）。

**三、方向由形状定、量级由阈值定。** 八臂同水平差的符号五档都不翻；**唯一翻号的 p3 `A` 在两种量上都翻**（−1.3→+11.3 步，0.85 上 15 负/10 正/5 平、0.95 上 27 正/3 负）⇒ 阈值挪动不制造方向，只放大已有方向。

**四、`steps_ratio` 比的不是同一水平。** 它在同一阈值上比两臂步数，而"臂到 0.9"与"基线到 0.9"是两条曲线上两次取值：p4 两臂到点准确率几乎相同（差 −0.005～0.000），p3 的 `R`／`C` 已差 0.02–0.03。归档新增 `band_accuracy` 记每个阈值的两臂到点准确率区间。

**改了什么**：`ledger.py` 新增 `threshold_band()`／`level_matched_steps()`／`threshold_transfer()`／`curve_tail()` 并接入 `account_checks` 第五项；`ledger.json` 新增 608 字段、**旧字段 0 变化 0 删除**（扁平比对＋`--verify` 逐字段一致）；`test_ledger.py` 新增 `TestThresholdBand`（6 条，含两条"记录错误判据"的负向对照）与 1 条注入测试；§13.47.1 增"补遗"小节、§11.4 第四实例与 §18.9 7.6 行同步补"方向由形状定、量级由阈值定"；实验 readme 第五批补遗一节（含八臂表）；TS-001 条文补遗。

**核对产物**：两文件 **54 passed**（`test_ledger` 30→37）；四条注入（`first_cross` 未达记 0 步／`sign_flip` 恒 False／`level_matched` 拿臂自比／`span_pp` 恒 0）**全部被测红**，恢复后全过。只读既有归档、未跑 GPU、未触上游。

**留给下一轮的三条**：① **负向对照的预期值本身也要先验**——"理论值应为 0"如果不成立，检查会永远红，而这次被记成"代码 bug"查了一整轮；② 阈值摆幅与基线顺序带宽是**同一族问题**（"设置项比效应还宽"），判据已在四处各写一遍（`effect_exceeds_band`、`threshold_band`、`n_identical`、`same_sign_fraction`），**该抽到共享工具了**（第四十二批起就在记，仍未做）；③ `wall_ratio` 依赖同一个阈值，本轮只核了 `steps_ratio`，`wall_ratio` 的同水平拆解未做。

入口：[§13.47.1 补遗](notebook/rollout-performance-research.md) · [readme 第五批补遗](experiments/rollout-c-class-ledger/readme.md) · [ledger.py](experiments/rollout-c-class-ledger/ledger.py) · [归档](experiments/rollout-c-class-ledger/ledger.json) · [test_ledger.py](experiments/rollout-c-class-ledger/test_ledger.py) · [TS-001](queue_tech-secretary.md) · [diary](diary/2026-11-03.md)。

### M1 上一轮（2026-11-03 16:0x–16:3x）：C 类账的"收敛快慢"还带着一个阈值——0.85–0.95 之间摆 1.7–18.2pp，p3 的 `A` 会翻号

技术秘书侧无新用户任务；TM 系列对外动作仍全部待用户逐项授权。上一轮确立的口径是"**配对差必须同时报符号分布与基线抖动**"。本轮把同一类问题搬到另一条线上：§13.43／§13.47 的 C 类机制账**整个 L3 收敛层建在一个 0.9 阈值上**，而 §18.9 7.6 正要把"步数"这条读数写进 survey 正文。第四批判定三说"到点余量 2–4 格 ⇒ 阈值更稳"——那条量的是**到点那一步离隔壁多远**，本轮量的是**整列读数会不会随阈值漂**。

**一、`steps_ratio` 的摆幅是 1.7–18.2pp。** 阈值挪到 0.85／0.87／0.9／0.93／0.95：`freq` **1.024→1.206**（"慢 2.4%"→"慢 20.6%"）、p3 `A` 0.978→1.127、`R` 0.898→0.806、`B` 0.912→0.945、`C` 0.898→0.877；`cancel`／`cap`／`replay` 只摆 1.7–4.2pp。⇒ §13.43 ③ 的 1.070 只能读作"**在 0.9 上**"。

**二、八臂里只有 p3 的 `A` 会翻号。** 0.87 上它已从 0.978 跨到 1.005——"只降方差让收敛变慢 5.6%"在 ±3pp 阈值内不成立。其余七臂方向稳定 ⇒ **不能说这一列"稳"或"不稳"，要逐臂读**；四类收益来源的归类不受影响（归类读的是 `cost_ratio`／`traj_ratio`／`wall_ratio` 的方向，没翻）。

**三、贴着阈值 ≠ 读数会翻。** `threshold_near.py` 量同一对种子上两臂是否同时不动：±0.001 时 p4 仅 3–13/30 对不动、p3 0–1/30；`cap` 最稳（±0.005 时 12/30）、`freq` 最不稳（3/30）。**脆弱性来自跨过阈值后曲线还在往上爬**——正是"余量 2–4 格"覆盖不到的方向。

**改了什么**：新增 §13.47.1（五档表＋三条读数＋三条引用规则）；§13.47 判定三加收窄指向；§18.9 7.6 补"步数这一列必须带阈值"；§11.4「同一句话差一个量级」补第四个实例（阈值也是基线要素）；实验 readme 增第五批一节；两个独立脚本入库。

**⚠ 一个没定位的失败（已于 19:0x 定位：判据错了，见本轮）**：本想把上面两条做成 `ledger.py` 的 `account_checks` 第五项（`threshold_band()`／`threshold_shift()`）并配负向对照。**失败**——把曲线在到点之后压平（理论上摆幅应为 0）后，函数仍报 `span_pp=107.005`。已排除：数据本身（函数内部 `num`／`den` 求和正确，`0`／`31`）、源码与 AST 一致、无重复定义、无变量遮蔽、工作区无 `__pycache__`（缓存在 `~/Library/Caches/com.apple.python`）；把同一段代码单独 `exec` 出来跑，行为又不一致。**未定位**。已把 `ledger.py`／`ledger.json`／`test_ledger.py` 全部 `git checkout` 回退（回退后 `--verify` exit 0、`test_ledger` 30 passed、`test_resolution` 17 passed），结论只由**不 import `ledger` 的两个独立脚本**产出。**下次要做成正式检查，先做最小复现。**

入口：[§13.47.1](notebook/rollout-performance-research.md) · [readme 第五批](experiments/rollout-c-class-ledger/readme.md) · [threshold_probe.py](experiments/rollout-c-class-ledger/threshold_probe.py) · [threshold_near.py](experiments/rollout-c-class-ledger/threshold_near.py) · [TS-001](queue_tech-secretary.md) · [diary](diary/2026-11-03.md)。只读归档、未跑 GPU、未触上游。

### M1 上一轮（2026-11-03 15:0x–16:0x）：真实负载上的策略位移也要分档——cap256 那两格"反号"落在基线自身抖动里

技术秘书侧无新用户任务；TM 系列对外动作仍全部待用户逐项授权。接上一轮的线索：分桶探针第三版刚刚确立"均值不能当效应"，但那条判据只补在**构造负载**的探针上，**真实 rollout 负载那一侧（§13.30 第四批）仍然只有均值**——而它正是要写进主文档的那批读数。本轮**不新开实验、不改任何测量值**，只给 `real_lengths_policy.py` 补三条判读量再分档。

**一、真实档 FIFO 基线自己就能动 0.7–11.7pp。** 组集合固定、只打乱传入顺序：4 批档中位 2.83／0.00／3.83／4.02pp，p90 最大 11.65pp。**第四批报的 +1.99／−0.95／+0.44／−0.39pp 全部落在各自带宽内或带边。**

**二、符号稳定性把 16 格劈成两半。** cap1024 六格 **28–30/30 同号**（方向"长度优先让 `R` 变大"），4 批两档效应（+5.66／+8.15pp）还超出带宽；**cap256 各格要么符号混杂（10/17、10/9），要么位移小于带宽（−0.95 对 1.25pp、−0.39 对 1.29pp），要么 11–19/30 个种子两臂读数完全相等**（低积压 `R` 本身接近 0，无分辨率）。

**三、"到达顺序"这个假设不是原因。** 把 FIFO 的到达序从 `prompt_idx` 序换成随机序，配对差几乎不动（+1.99→+2.06pp、−0.95→−1.19pp，逐种子同号 23–30/30）⇒ 该钉的不是"到达序选错"，而是**这类负载上配对差本身就有这么厚一层随机性**（来自抽到哪 16 组，不来自 16 组怎么排）。

**⇒ 订正**：第四批"更强调度抬税在真实负载上**反号**"——**cap256 那一半不可引用，只能说方向未定**；可引用的只有不截断档（cv 0.30–0.36）一侧。**`R` 的量级读数（2.7%／11.4%）不涉及两臂配对，不受影响。**

**改了什么**：`real_lengths_policy.py` 新增 `policy_sign_stability()`（含 `n_identical` 分辨率刹车）／`fifo_reorder_band()`／`arrival_assumption_shift()`；归档新增 256 字段、**旧字段 369 个 0 变化 0 删除**（扁平比对 + `--verify` 逐字段相同）；`test_real_lengths_policy.py` 新增 5 条 `TestSignStabilityDiagnostics`；§13.30 新增第五批、第四批标题加订正框、§11.4 第 3 条与语义面表行、§11.5 缺口表与 ⚠ 框、实验 readme 第四批三处 + 新增第五批节、bucketing-probe 交叉引用、TM-011 第四十二批条文与停止条件重写、run-info 新增 `sign_stability_checks`。

**核对产物**：两文件 **59 passed**（本文件 15 个）；`real_lengths_policy.py --verify` 逐字段相同；**两次注入均被测红**（`d > 1e-12` → `d >= 0` 时两臂恒等工作为 0 被误报；删掉 `shuffle` 后带宽恒 0 被抓到），恢复后全过。只读仿真、未跑 GPU、未触上游。

**留给下一轮的三条**：① 带宽是**判读门槛不是显著性检验**，未做 CI／置换，引用时要守住措辞；② 真实负载的"到达顺序"仍没有独立于样本的物理含义，本轮只把假设的价格量出来，未解决；③ "只报中心量不报分布"今天在两个代码路径上各犯一次，判据应进共享工具而非各脚本各写（未做）。

入口：[§13.30 第五批](notebook/rollout-performance-research.md) · [real_lengths_policy.py](experiments/rollout-group-split-tradeoff/real_lengths_policy.py) · [归档](experiments/rollout-group-split-tradeoff/real-lengths-summary.json) · [实验 readme](experiments/rollout-group-split-tradeoff/readme.md) · [TM-011](queue_tech-miner.md) · [diary](diary/2026-11-03.md)。

### M1 上一轮（2026-11-03 14:0x–14:3x）：分桶探针的均值读数分档——只有 cap1024 ≥4 批站得住，并核出 TM-011 立项前置是改造排程器

技术秘书侧无新用户任务；TM 系列对外动作仍全部待用户逐项授权。本轮接 TM-011 的备料，**不新开实验**，只给已有探针补两条判读量，然后重新分档。

**一、FIFO 基线自己就依赖"到达顺序"。** `FIFO` 的定义是"按到达顺序"，而在本排程器里到达顺序**就是传入列表的顺序**；真实负载每个种子抽到不同的 16 组，于是种子间的 FIFO 基线混了"抽到哪些组"和"谁先到"两个来源。把后者单独量出来（组集合固定、只打乱顺序）：同批组只换顺序，`R` 就动 **1.4–5.5pp**（1.5B cap1024 4 批最大 5.52pp）。

**二、符号稳定性把 16 格劈成两半。** 用 `same_sign_fraction` 与种子间 sd 判读：**只有 cap1024 且积压 ≥4 批的六格是 19–20/20 同号**（方向"长度优先让 `R` 变大"）；**cap256（RL 实际配置）各档符号在种子间翻转**（3/13、13/7、10/8、14/5、10/10）；两档 cap1024 2 批 sd 是均值的两倍以上（6.27／6.62）。⇒ 上一版正文的"−1.97／+0.54／+1.23pp"是**把均值当效应**，这些格子不支持方向性结论。主导结论"只排序有害"**只在 cap1024 ≥4 批成立**。

**三、TM-011 立项前置是改造排程器，不是加分桶逻辑。** 核了三条硬约束：① `cap = n_groups * n_per` 恒等于单批轨迹总数；② 组到达 `arrival = b * batch_span`（同步到达）；③ 单批内组臂一次放下全部组（乱序 10 次 `group_makespan` 全同）。⇒ **"哪些组进同一批"在本模型里不是可改的变量**，分桶改的正是批组成。这不是"分桶没用"，是**这个工具问不了这个问题**。停止条件已改写：立项前置＝独立到达流＋批窗口可分离＋容量与批解绑。

**改了什么**：`bucketing_probe.py` 新增 `seed_sign_stability()`／`fifo_order_sensitivity()`／`effect_exceeds_band()`；`bucketing-probe.json` 新增 136 字段、**旧字段 0 变化 0 删除**（已 diff）；`test_group_split_tradeoff.py` 新增 6 条（两个类）；`bucketing-probe.md` 增第三版章节、第二版表加"不要单独引用"标记、一句话结论收窄；TM-011 增第四十一批条文并给第二版条文加订正指向。

**核对产物**：两文件 **54 passed**；`group_split_tradeoff.py --verify`／`legacy_group_arm_check.py --verify`／`real_lengths_policy.py --verify`（约 10 分钟重算）**三处均 exit 0**；`verify_anchors --verify` 34 一致／0 不一致；三次注入（`same_sign_fraction` 恒 1.0／敏感度不换序／`effect_exceeds_band` 恒 True）**全部被抓到**。只读仿真、未跑 GPU、未触上游。

**⚠ 推送状态：已恢复但直连不稳。** 上轮记的"连不上 GitHub"已解除——开工时 `curl` 200、`git ls-remote` 与 `git fetch` 均通；但 `git push`／`curl` 随即在 75s 后超时（同一次会话内时通时不通），而 **`gh api` 始终可用**。本轮按既有约定走 [git-push-api](tools/git-push-api/readme.md)：预演（`push_dryrun` 2752/2752）＋差异核对（8 个文件全为修改、0 删除）后推到远端 main **`d9180a03`**，8/8 个 blob 逐一核对一致，再用 gh-proxy 取回对象、本地 main 对齐（本地 `ec45f7ddb` 与远端同树不同 SHA）。**下次仍先试直连，不通再走 API；API 推送前必须先预演并核 `git diff --name-status <远端sha> HEAD` 没有删除项。**

**留给下一轮的两条**：① 探针仍不建模"分桶改变批组成"（本轮已把这一条变成 TM-011 的立项前置，不再是模糊缺口）；② 真实档每格 20 种子、未做 CI／置换检验，`effect_exceeds_band` 只是判读门槛不是显著性——引用时措辞要守住这条。

入口：[bucketing-probe.md](experiments/rollout-group-split-tradeoff/bucketing-probe.md) · [bucketing_probe.py](experiments/rollout-group-split-tradeoff/bucketing_probe.py) · [TM-011](queue_tech-miner.md) · [diary](diary/2026-11-03.md)。

### M1 上一轮（2026-11-03 13:1x–13:3x）：分桶探针的两处口径再修一次，并把 `grouping_tax` 组臂修复与档案一起落地

技术秘书侧无新用户任务；TM 系列对外动作仍全部待用户逐项授权。本轮接的是 `7b2f8db3b`（"组级排序平均降 R 5.95pp"）→ `e5d3386b9`（并行会话推翻为"真实负载上方向相反"）这条接力。**不再改结论，只把探针做对，顺带把上一条遗留的组臂修复提交掉。**（**注**：该轮的 cap256 读数已由本轮第四十一批订正为"符号不稳"，见上。）

**一、5.95pp 除基线泄漏外还有第二处配对错位。** 现行版让两臂共用同一份乱序——那个乱序被当成两臂共有的派发次序，而 `FIFO` 的定义是"按到达顺序"。改成 FIFO 臂吃原顺序后读数稳定下来：真实 1.5B cap1024 四档 **−3.34／−6.33／−2.93／−1.44pp**，1.5B cap256 四档 **−0.24／−1.97／+0.54／+1.23pp**。**与并行会话的订正逐格一致**（两边独立重写，落在同一组数上），说明这条订正确实是口径问题而不是随机性。

**二、探针现在把两臂各自的增益一并归档，机制因此闭合。** `R1 − R0 = (T0/G0) − (T1/G1)` 是代数恒等，符号由"哪一臂被排得更整齐"直接决定：1.5B cap256 4 批组臂 +0.3%／轨迹臂 +2.3%（`R` 升）、16 批组臂 +1.8%／轨迹臂 +0.4%（`R` 降）；cap1024 两档都是轨迹臂赢。**这与 §13.30 第四批"随积压加深受益方换人"一致，只是换了算法**（那边逐对拆臂，这边用恒等式）。

**三、`grouping_tax` 的组臂修复与档案落地。** 上一轮核出的"组臂归还粒度退化"（旧路径 `R` 低约 22%）现已在 `group_split_tradeoff.py` 中修好（新增 `_group_makespan`），负向对照两条测试在未修代码上 FAIL（`−3.0 != 0.0`）；`summary.json` 重算后只有 `layer2_grouping_tax` 与 `layer3_verdict.best_case_grouping_tax` 变，其余逐位不变。修正后 `grouping_tax` 与口径正确的 `_makespan_by_policy` **逐位相同**。`legacy_group_arm_check.py` 的守护用例原用 4 组 × 4 条（cap 恰等于工作量、旧路径也给 0，等于没守），已改成 4 组 × 8 条 × 2 批，旧路径给 **−100%**。

**核对产物**：`pytest` **48 passed**；`group_split_tradeoff.py --verify`／`real_lengths_policy.py --verify`／`legacy_group_arm_check.py --verify` 三处均 exit 0。只读仿真、未跑 GPU、未触上游。

**⚠ 推送未完成**：提交 `d7eeba0e6` 已在本地 main，但 `git push` 与 `git ls-remote` 均在 75s 后超时（`curl https://github.com` 20s 无响应，无代理配置）——本机当前连不上 GitHub。**恢复条件**：网络恢复后执行 `git push origin main`，先 `git log origin/main..main` 核对没有把并行会话的提交一起推上去。

**留给下一轮的两条**：① 探针仍不建模"分桶改变批组成"，TM-011 的停止条件已按否定读数改写（只排序有害、要救回来必须建模组成）；② 真实档每格 16 组 × 20 次，不主张 0.1pp 级精度。

入口：[bucketing-probe.md](experiments/rollout-group-split-tradeoff/bucketing-probe.md) · [legacy_group_arm_check.py](experiments/rollout-group-split-tradeoff/legacy_group_arm_check.py) · [TM-011](queue_tech-miner.md) · [diary](diary/2026-11-03.md)。

### M1 上一轮（2026-11-03 13:0x–13:2x）：§13.30 第一批的 `R` 六个读数作废并重算——组臂归还粒度曾经退化

技术秘书侧无新用户任务；TM 系列对外动作仍全部待用户逐项授权。本轮做的是**核并行会话的结论**：它的提交说明说 `grouping_tax` 旧组臂"量出负税 −33%"，我按规矩两边各算一遍，核出两件事。

**一、那句原因不成立。** 等大组上旧路径只让组臂**更快或一样**，所以税恒 ≤ 新路径、**不会变号**（20 万组等大随机输入里一次负税都没有）；−33% 其实是**参差输入**（`[[50,50,250,250],[100]*4+[300]*4]`），属第三十九批那条守卫管的事。**我自己也犯了同一错**：初稿照抄"负税"当原因，测试还用那个例子当负向对照，红了两轮才发现。

**二、但"旧路径坏了"成立，量级不小。** 旧组臂把整组拆成 `n_per` 个各自独立的单条任务，一完成就归还全部槽位 ⇒ 归还粒度退化成逐条。30 种子同口径对照（构造档 16 组 × 8 条）：cv 0.2 的 4 批 20.6%→**14.7%**、16 批 29.8%→**21.0%**；cv 0.6 的 4 批 43.7%→**34.0%**、16 批 54.7%→**42.4%**；cv 1.5 的 4 批 56.6%→**44.9%**、16 批 68.8%→**56.3%**。**§13.30 第一批那张表的六个数（43.7／54.7／20.6／56.6／27.7／38.6）逐项对上旧路径** ⇒ "前四批不受影响"这句话**对第二批成立、对第一批不成立**（第二／三／四批走 `_makespan_by_policy`，组号按整组取）。

**改了什么**：§13.30 第一批结论三表下加订正块（六值＋正确值＋结论四"上限 61%"的订正）；§11.4 语义面表行与第 3 条、§11.5 读数段与边界、§16 两处同步；新增 `legacy_group_arm_check.py`（旧/新并列，`--verify` exit 0）与三条单测 `TestGroupArmReturnsWholeGroup`（与策略层逐位相同／有积压时非负／旧路径低估的负向对照 `[[400,400],[40,40]]` 积压 4 批：旧 45.0%、正确 **0.0%**）。

**核对产物**：两文件 **48 passed**；`legacy_group_arm_check.py --verify` exit 0；`verify_anchors --verify` 34 一致／0 不一致；`group_split_tradeoff.py --verify` exit 0。只读重算、未跑 GPU、未改任何测量值。

**未接手并行会话在写的东西**：`bucketing_probe.py`／`bucketing-probe.*`／`group_split_tradeoff.py` 的 `_group_makespan`／`summary.json`／`queue_tech-miner.md` 的 TM-011 段当时都在工作区未提交，本轮**没动它们**（只改了 `readme.md` 的 L2 段与 `notebook`）。→ **已于 13:3x 由下一轮落地**（探针配对法＋组臂修复＋TM-011 改写），见上一条。

入口：[§13.30 第四十批](notebook/rollout-performance-research.md) · [legacy_group_arm_check.py](experiments/rollout-group-split-tradeoff/legacy_group_arm_check.py) · [归档](experiments/rollout-group-split-tradeoff/legacy-group-arm-check.json) · [diary](diary/2026-11-03.md) · [TS-001](queue_tech-secretary.md)。

### M1 上一轮（2026-11-03 11:3x–12:1x）：TS-001——给排程器补一条输入守卫（参差组能静默量出负税），并把"分桶／预测"缺口量了个上限

技术秘书侧无新用户任务；TM-001～TM-010 对外动作仍全部待用户逐项授权。TS-001 优先。

**先说清本轮没做成什么**：本意是给 §11.5 的「长度预测／分桶／迁移」缺口做一个上限读数，试下来**不足以成篇**——真实 1.5B cap256 上 FIFO `R` 4 批 2.4%、同负载"两臂各自按长度排"（`lof_flat`，不可实现）12.9%，差约 1.3pp；16 批 10.8%→13.8%。原因是真实组间（组最大）离散只有 **cv 0.07** 而组内极差 85 token：**组最大几乎齐平 → 任何组级排队顺序都无事可排**，三个组级策略在同一负载上给出完全相同的读数（0.25）。这解释了为什么第三／四批的策略维位移那么小。**上限档用的是不可实现的 `lof_flat`，要成篇需单独建模型（分桶粒度、预测误差），已登记为后续候选，未写进主文档。**

**本轮实际交付：一条输入守卫。** 构造探针时出现**负税（−33%）**，查出根因是输入违约——`_makespan_by_policy` 的 `n_per` 只取第 0 组，两组写成 4 条／8 条时组臂的槽位占用与归还量对不上。判据：**负税在"整组归还只会更慢"的定义下不可能**，而正文把 `R` 当上界引用，能为负的量不构成上界。对抗搜索（20000 组随机＋定向构造）**等大组下 `tax` 恒 ≥ 0**，负值只在参差输入出现。`_makespan_by_policy` 与 `grouping_tax` 两道入口现都 `raise ValueError`；三条测试（含"撤掉守卫后确实量出 < −0.1"的负向对照、等大组恒非负）。

**核对产物**：两个 `--verify` 逐字段相同；`summary.json` sha256 未变（`09466197…`）⇒ **归档里从未出现参差组，历史读数完全不受影响**；单测 30→33（本文件）、两文件合计 **43 passed**；负向对照实测撤掉守卫即 FAIL。run-info 新增 `ragged_group_guard` 记录。

**边界**：只拒绝非法输入，未建"真实负载会不会出现参差组"的模型——真实 rollout 每 prompt 固定 n 条，参差属调用侧违约，不是被测现象。

**并行会话（本轮全程）**：另一会话连续提交 `a06dc559b`／`067649cec`／`6e84e8523`（检查器空转分支、SKIP 降级），本轮守卫改动被打进 `6e84e8523`；已核对守卫文本与测试完整、43 测试通过、三份 sha256 与 run-info 一致。

### M1 上一轮（2026-11-03 09:0x–11:1x）：TS-001 复核——上一批归档字段里挑出两处「检查在报 OK 但没在看那个东西」

技术秘书侧无新用户任务；TM-001～TM-010 对外动作仍全部待用户逐项授权。§13.30 第四批已由前一轮落笔并提交，本轮按 §19.4 只做**复核**，不接新实验。

**正文全部复现**：表一（真实档 `R`）、表二（LOF−FIFO 位移）、机制拆臂（组臂 −1.60%／27-30、轨迹臂 −0.55%／23-30）、离散度扫描、ICC 扫描、并列诊断，逐项从 `real-lengths-summary.json` 重算，与正文一致。

**抓到两处真 bug（都不进任何结论，但都是「检查假阳性」）**：
1. **`describe()` 截断上限写死 `256.0`**——四档被读成同一对截断率，cap1024 报 45.2%／60.7%，而 §13.50 实测是 **0.03%／0.65%**（那两个数其实是 cap256 两档的）。已改为按本档 `max_response` 判定并新增该字段。**重跑后 `summary` 只有 6 个字段变化（两处 `truncated_fraction`＋四个 `max_response`），所有 `R` 与位移读数逐位不变。**
2. **`verify_anchors` 第 34 项有半条是死代码**——读 `arm['intra_group_cv']`，归档里**从来没有这个键**（真键 `structure['cv_within_pooled']`），`cv` 恒为 `None`，那段判据一次都没生效过却一直在报 OK。已改成三条真判据（cv 必须存在且 <0.4／真实档 `R` 必须低于构造档一半／cap1024 截断率必须按 1024 上限判定）＋第 6 组负向对照（喂"缺 cv＋cap1024 按 256 判"的假归档必须报出）。

**本轮最该记的教训：两处负向对照都先假通过。** 第一版测试读的是**归档 json**，撤掉修正后照样全绿（归档当时是对的，测试没穿过被改坏的代码）；第二版只测 `arm_cap()` 常量映射，仍是自证；第三版把结构描述抽成 `build_real_arm_structure()`、用 spy 截获 `describe()` 收到的 `cap`（断言 `{256: 2, 1024: 2}`）才变红。**判据已写进 §13.30 第四批：负向对照必须在"把被测代码改坏"的条件下运行，断言要落在被改坏的那根线。**

**一处操作事故（已恢复）**：为回退自己的一次失败重构用了 `git checkout -- .`，**连带回退了未提交的改动**（cap 修正与测试）。核后确认并行会话的提交都在、文档修正也已被其提交采纳，丢失内容已全部重做。**下次只按路径回退：`git checkout -- <file>`，不用 `.`。**

**核对产物**：`real_lengths_policy.py --verify` exit 0；两个测试文件 **43 passed**；`verify_anchors.py --verify` **34 一致／0 不一致 exit 0**，`--selftest` **6 组全过 exit 0**。

入口：[§13.30 第四批](notebook/rollout-performance-research.md) · [real_lengths_policy.py](experiments/rollout-group-split-tradeoff/real_lengths_policy.py) · [test_real_lengths_policy.py](experiments/rollout-group-split-tradeoff/test_real_lengths_policy.py) · [verify_anchors.py](tools/anchors/verify_anchors.py) · [diary](diary/2026-11-03.md) · [TS-001](queue_tech-secretary.md)。只读复算、未跑 GPU、未改任何测量值。对外动作全部未做。

### M1 上一轮（2026-11-03 08:0x–09:1x）：TS-001 补强——把 `R` 搬到真实 rollout 长度上，第三批的结论在真实负载上**反号**（§13.30 第四批／§11.4／§11.5）

技术秘书侧无新用户任务；TM-001～TM-010 对外动作仍全部待用户逐项授权。TS-001 优先，选的是上一轮留下的最直接的一处前提缺口：§13.30 前三批的负载全是 `sample_group_lens` **组内独立抖动**（组内相关 ≈0），而 M5 的 §13.50 已实测真实响应长度**组内高度相关**（ICC 0.36–0.52）、cap256 截断 45%／61%。组原子成本是 `max(g)`、逐条是全部 `g`，这个差直接由组内结构决定 ⇒ **`R` 从没在真实负载上量过**，第三批「更强的调度让 `R` 变大」也没做过前提检验。

**做法（纯 CPU 重算，真实长度数据早已随 §13.50 入库）**：不改排程器，只把 `experiments/rollout-length-structure-real/raw/*.jsonl.gz` 的 `response_tokens` 按 `prompt_idx` 聚组（480 组 × n=8，四档）替换负载来源。**先做口径校验**——构造档复现第三批 **0.3402／0.3677**（正文 34.0%／36.8%），排程器没漂。

**三条结论**：

1. **量级**：RL 实际配置（1.5B cap256、组内 cv 0.155）上 `R` 只有 **2.7%（4 批）／11.4%（16 批）**，是构造档的三分之一到十分之一；不截断档（cv 0.30–0.36）回到 21.8%／33.4%。**截断把组内离散压扁，是主要塑形因素。**
2. **符号**：第三批「更强调度抬税」在真实档**深积压上反号**——16 批 1.5B **11.4%→10.4%（27/30 为负）**、0.5B 5.6%→5.2%（18/30）；4 批仍为正（+1.99pp）。
3. **归因**：逐对拆两臂——1.5B 4 批时组臂只 −0.45%（5/30）、轨迹臂 −2.5%，比值升；16 批时组臂 −1.60%（27/30）、轨迹臂只 −0.55%，比值降。**同一负载随积压加深，受益方就换人**；第三批两档恰好都在前者。控制变量：ICC 只改量级（四档位移恒正）、截断并列不是原因（量化到 5／20 格同号）、**离散度扫描在 cv 0.25–0.40（4 批）／0.40–0.60（16 批）之间翻转**。

**本轮一处自查**：初稿写「16 批积压下组臂快 1.60%、轨迹臂只快 0.55%，所以比值下降」——**只对 16 批成立**。补上臂拆分后才发现 4 批是**轨迹臂**受益更多，原来那句是拿一个档的现象解释两档，已按逐档读数重写。

**核对产物**：`real_lengths_policy.py --verify` exit 0；单测 **36 个通过**（新增 6 个，含「高 cv 抬税／低 cv 降税」两条负向对照）；`group_split_tradeoff.py --verify` 仍 exit 0（既有 L1–L4 **逐位不变**）。三组识别性全过（构造档复现／零负载六档全零／`lof_flat` 在真实档 30/30 全正）。

**并行会话（08:3x）**：另一会话整理了我上一轮留下的 §11.5 三处重复（`dfe83d7a2`）与「未入库」性质澄清（`b17dbbfd8`），不影响本批结论；本轮编辑已基于最新文件状态，无冲突。

入口：[§13.30 第四批](notebook/rollout-performance-research.md) · [§11.4](notebook/rollout-performance-research.md) · [§11.5](notebook/rollout-performance-research.md) · [实验](experiments/rollout-group-split-tradeoff/readme.md) · [归档](experiments/rollout-group-split-tradeoff/real-lengths-summary.json) · [TS-001](queue_tech-secretary.md)。只读仿真、未跑 GPU、未改任何测量值。对外动作全部未做。

### M1 上一轮（2026-11-03 07:0x）：TS-001 补强——清掉语义面里一句从没被量过的推断（§13.30 第三批／§11.4／§11.5）

技术秘书侧无新用户任务；TM-001～TM-010 对外动作仍全部待用户逐项授权。TS-001 优先，选的是 §11.4／§11.5 里**最像结论、其实只是推断**的一句：「长度预测／分桶／迁移本来就部分补偿了组粒度，**`R` 在它们之上会变小**」。§13.30 前两批的排程器只有 FIFO 一种排队顺序——这句话没有任何读数。

**结论（与推断相反）**：在 §13.30 的换组粒度对照上只换**组的排队顺序**，跑了六档策略臂——`R` 不是策略不变量，而且方向是**变大**：4 批积压 FIFO **34.0%→LOF 36.8%（+2.75pp）**、16 批 **42.4%→42.9%（+0.51pp）**。机制：长度优先让两臂都更快，但**轨迹臂受益更多**（4 批组臂 −11.5%、轨迹臂 −15.2%），比值因此上升。**准确表述：调度器补偿的是绝对 makespan，不是 `R`；更强的调度让组粒度的相对代价更显眼。**

**三条必须一起写的边界**：① `arrive_max` 因本模型所有组同步到达而退化——**η 通道未建、未被检验，不得写其无效**；② `pending_max` 与 `lof` 在本模型下同义（整组一次放出时"剩余最长"≡"整组最长"）——**不是两条独立证据**；③ 单负载配对差方向不稳（4 批 3/30、16 批 5/30 为负），只有均值方向可信。

**本轮两处自查（都是先错后改）**：① **先写结论再跑数**——初稿在 `reading` 里已写"排队顺序不改变分组税"，跑出来 LOF 抬税 2.75pp，已按读数改写（同族第六处）；② **一次错误的 bug 诊断**——中途把"固定 pending 列表不重排"当成吞掉策略轴的 bug 并写进测试，连续两次断言失败后核清：释放的槽位总落入列表首部，两种写法**完全等价**，该 bug 不存在，诊断已撤回。

**核对产物**：既有 L1–L4 读数**逐位不变**（新旧 summary 对比仅新增 `layer2b_policy_invariance`）；`--verify` exit 0、30 个单测通过、`run-info.json` 三份 sha256 已更新并复核一致。识别性三对照全过（`lof_flat` +15.9pp 且 30/30 全正、`lof_at_arrival` +2.75pp、零负载六档全零）。

**推送（07:2x）**：`git push` 直连仍不通（75s 超时），用 git-push-api 推到远端 main **`1088bbe91`**；**推前核清远端 `da6b69c7` 与本地 `fcf093e5` 树相同（`32e30e03`）**，diff 只有本轮 11 个路径；逐文件 blob 已核对、本地已 reset 对齐。

入口：[§13.30 第三批](notebook/rollout-performance-research.md) · [§11.4](notebook/rollout-performance-research.md) · [§11.5](notebook/rollout-performance-research.md) · [实验](experiments/rollout-group-split-tradeoff/readme.md) · [TS-001](queue_tech-secretary.md)。只读仿真、未跑 GPU、未改任何测量值。对外动作全部未做。

### M1 上一轮（2026-11-03 06:0x）：TS-001 补强——按 §19.4 第 8 条把核验扩到放置效应，抓出一处分母串用

技术秘书侧无新用户任务；TM-001～TM-010 对外动作仍全部待用户逐项授权。上一轮把 `verify_anchors` 接进 checklist 后，**还没人真按它去核过东西**；本轮核的是第 2／7 章的承重结论——`§13.42`／`§13.45` 的放置效应，此前**一项都没被脚本覆盖**，而它是"加卡收益"叙述的前提。

**做了什么**：`verify_anchors` 由 16 项扩到 **33 项**（放置效应 A/B 八项、跨归档放置对照一项、SHM 带宽表八项，后者从 `raw/g8-comm-probe.txt` 解析控制台日志，因为归档只有文本）。`--verify` **33 一致／0 不一致／0 跳过 exit 0**，`--selftest` **5 组负向对照全过 exit 0**（新增两组：组间差混入 taskset 会变 16.89% 必须判不一致；SHM 时间比 0.514 必须能与错误模型值 1.000 区分）。

**抓到的真错误（已订正）**：§13.42 结果五原写"pC1（2+2）比 A 快 **16.3%**"——不绑定 A 两跑给 **16.8%**、四跑均值给 16.0%，**16.3% 两个都不是**，它恰是结果一里 **B 慢于 A** 的组间差。两对不同的跑次被写进同一句，与 §20.3.9 记的同族错误一致。已改为 16.8%（分母写进句内）。

**我自己的错误（已还原）**：登记"3+1 比 2+2 慢"时写成 `new/old−1` 得 −16.0%，与正文 19.1% 不符，我一度判定"正文错了"并改了主文档三处——**正文是对的**，"A 比 B 慢 X%"以 B 为分母（16.878／14.173−1＝19.1%），16.0% 只是拿 3+1 当分母的互补值。三处已全部还原。

**新口径（写进 §19.4 第 9 条）**：① 凡"比某组快/慢 X%"的句子，分母必须写在句内；② 核验报不一致时先假定检查器写错，改被核对象前手工把分母算一遍。

入口：[tools/anchors](tools/anchors/readme.md) · [§13.42](notebook/rollout-performance-research.md) · [§19.4](notebook/rollout-performance-research.md) · [§16](notebook/rollout-performance-research.md) · [TS-001](queue_tech-secretary.md)。只读复算、未跑 GPU、未改任何测量值。对外动作全部未做。

### M1 上一轮（2026-11-03 05:0x）：TS-001 补强——先修核对脚本的配对错档，再补同节点配对两档（§20.3.9／§11.6）

技术秘书侧无新用户任务；TM-001～TM-010 对外动作仍全部待用户逐项授权。TS-001 优先，选的是回溯上一批（§20.3.8）证据链时发现的问题——**出在核对脚本本身**。

**发现**：`check_reconstruct_form.py` 里同配置重复臂写成 `reanchor-summary.json` 的 **`g4p`** 配 `summary.json` 的 **`g4n`**（0.7030／−0.142%），而正文引的是 **`g4` vs `g4n`**（0.8372／−0.060%）。`g4p` 是独占窗口下的 4 卡加卡测量、`g4`／`g4n` 是 GPU 2 被占时非连续掩码的两跑——**同为"2+2 放置"但不是同一次测量**。没被报出来的原因：`--verify` 对对照臂只打印不判定，而**错配那一对 F3 照样闭合**。

**修了什么**：① 改取同一份归档＋`same_file/same_stack/same_window_rule` 自检，复原 0.8372；② 负向对照加第 ⑤ 组（正确 0.8372／`g4p` 配错 0.7030／`g1p` 配错 1.3652 须可区分）；③ 补同一归档里被漏掉的 2 卡两档：`2.0→0.2`＝**1.0045**（F3 +0.000%，全表最小效应）、`2.0→0.5`＝**1.0357**（+0.019%），与 `2.0→0.1`（1.0373）连成剂量线，`0.5` 与 `0.1` 差 0.15%，与 §13.49 独立吻合。

**判决**：主结论臂由 8 个扩到 **10 个**（加卡 2＋单轴 7＋async 1，F3 全在 0.1% 内）；**锚② 强度仍是「中强」，不回调；跨系统仍归 P-1**。新增项都不进主结论句。**一条新口径纪律**（已写进 §11.6）：形式的闭合不能代替配对正确——引用任何 A/B 臂前先确认两跑出自同一份归档、同一窗口规则。这是本专项第三次同族错误（前两次：只取两条跑、两套放置混用）。

**核对产物**：`check_reconstruct_form.py --verify` exit 0、`negative_control_form.py` 5/5 抓到 exit 0；`check_identity.py --verify`／`negative_control.py` 重跑仍 exit 0。

入口：[§20.3.9](notebook/rollout-performance-research.md) · [§11.6](notebook/rollout-performance-research.md) · [归档](experiments/rollout-anchor2-identity/README.md) · [TS-001](queue_tech-secretary.md)。对外动作全部未做。

### M1 上一轮（2026-11-03 04:0x）：TS-001 补强——P-6 的适用面加 async 轨迹与同配置重复臂（§20.3.8／§11.6）

技术秘书侧无新用户任务；TM-001～TM-010 对外动作仍全部待用户逐项授权。TS-001 优先，选的是上一批留下的**最明显的一处缺口**：§20.3.7 的 8 个臂**全在 sync 路径上**，"形式不挑执行模式"此前没有数据。

**补了两行已入库归档，不改形式本身**：① **async 2 卡 poll 2.0→0.1（两跑池化）**实测整步比 1.0194，完整式误差 **+0.049%**（第一项单独 +0.029%，两侧缺口差 −0.027%）；② **g4 vs g4n（同配置重复，2+2 与 0,1,3,4）**实测 0.8372，误差 **−0.060%**，是本栈两侧缺口差最大的臂（0.029%，约其他臂 3 倍）、加速只有 1.19×。

**判决**：P-6 的适用面由"8 个 sync 臂"扩为"8 sync＋1 async＋1 同配置重复，误差都在 0.1% 内"——形式的成立**不依赖 sync 的步结构**。**两行都不进主结论**：同配置重复臂的"加速"是运行间噪声（同一配置两次跑、只有放置微差），跨放置臂（`pbase→p4base`，0.163%）按既有约定只作对照。**锚② 强度仍是「中强」，本批不回调**；跨系统仍归 P-1。

**核对产物**：`check_reconstruct_form.py --verify` exit 0（主结论臂 0.1% 阈值，两行新臂只打印不判定）、`negative_control_form.py` 4/4 抓到 exit 0；`check_identity.py --verify`／`negative_control.py` 重跑仍 exit 0。

入口：[§20.3.8](notebook/rollout-performance-research.md) · [§11.6](notebook/rollout-performance-research.md) · [归档](experiments/rollout-anchor2-identity/README.md) · [TS-001](queue_tech-secretary.md)。对外动作全部未做。

### M1 上一轮（2026-11-03 03:0x）：TS-001 补强——P-6 完成，整步比的可重建形式（§20.3.7／§11.6）

技术秘书侧无新用户任务；TM-001～TM-010 对外动作仍全部待用户逐项授权。TS-001 优先，承接上一批（§20.3.6）留下的投稿前必补项 P-6。

**结论**：形式为 `整步比 = [1/(Σ_k s_k^基线/G_k)] × (δ^基线/δ^目标)`，`s_k^基线 = t_k^基线/Σt^基线`、`δ = S/T`（计时缺口）。**是代数恒等不是拟合**（由 `Σ_k s_k^基线·(T^基线/t_k^基线) = 1/δ^基线` 化简）。同栈 **8 个臂**（2 加卡＋6 单轴）误差 **≤0.04%**，加卡臂由旧式的 −5.1%／−15.4% 变为 −0.007%／+0.010%；跨放置对照臂 0.16%，只作对照。

**旧式失效的机理（本批新增诊断）**：同一算子把份额取在**目标侧**时，`Σ_k s_k^目标·(T^目标/t_k^目标) = 1/δ^目标`，式子退化为 `δ^目标·(S^目标/S^基线)`，连加速倍数都不是。问题**不是"用了份额"，而是份额取了加速之后的那一侧**——效应被提前算进权重，越大算得越少。

**两个限定已写进正文**：① 本栈两侧缺口几乎相同（|差| ≤0.08%），第一项单独就接近整步比，但这是**数据事实不是恒等**（负向对照把缺口拉开 20%，第一项偏 20%、完整式偏 0）；② 单栈单点，显著程度不外推。**锚② 强度保持「中强」不回调**——补的是形式可用，跨系统仍是 P-1。

**核对产物**：`check_reconstruct_form.py --verify` exit 0、`negative_control_form.py` 4/4 注入抓到 exit 0；上一批的 `check_identity.py --verify`／`negative_control.py` 重跑仍 exit 0，未破坏既有结论。

入口：[§20.3.7](notebook/rollout-performance-research.md) · [§11.6](notebook/rollout-performance-research.md) · [归档](experiments/rollout-anchor2-identity/README.md) · [TS-001](queue_tech-secretary.md)。对外动作全部未做。

### 上一轮（2026-11-03 02:0x）：TS-001 补强——锚② 的可换算性复核（§20.3.6／§11.6）


技术秘书侧无新用户任务；TM-001～TM-010 对外动作仍全部待用户逐项授权。TS-001 优先，按其"在已有范围内筛选仍有明确新增信息的补强项"继续，选的是 §20.3.5 里一句一直被当已核实的话。

**结论**：把 `s_k` 与 `G_k` 代回 §13.25 写下的乘积式、与**实测整步比**对照（这一步此前从未做过）——2 卡重建 **1.4485 vs 1.5264（−5.1%）**、4 卡 **1.6429 vs 1.9419（−15.4%）**；且该式在 `G_k≡1` 时给 **1** 而非 `T1/TN`，说明它度量的是**单配置内的相对敏感性**，不是两配置的整步秒数比。差额**是形式问题不是口径问题**：份额随配置移动（`update_actor` 占分项和 0.484→0.450、`update_weights` 0.066→0.166），而"取哪一侧份额当权重"只值 0.2–0.5%（§13.46）。真正闭合的账是"**逐阶段秒数相加 + 计时缺口**"，缺口在 8 个臂上 **0.23–0.31%**。

**改了什么**：锚② 收窄为"分阶段中介"、强度由「强」降「中强」；新增投稿前必补项 **P-6**（给出能重建整步比的形式）；会计面判据的可证伪形式改写进 §11.6。**没改什么**：§13.21–§13.27 任何读数、§13.45／§13.46 结论、§20.3.4 对 P-1 的判断均不动——本轮否定的是一个**读法**，不是测量值。

**核对产物**：两套多卡锚点（2+2 与 2+0／3+1）的分阶段 `G_k` 已并列入 `experiments/rollout-anchor2-identity/summary.json`，写作时从该文件取、不转述（§20.3.5 引用的五对本身无错配，风险在转述环节）。

入口：[§20.3.6](notebook/rollout-performance-research.md) · [§11.6](notebook/rollout-performance-research.md) · [归档](experiments/rollout-anchor2-identity/README.md) · [TS-001](queue_tech-secretary.md)。对外动作全部未做。

### 上一轮（2026-11-03 01:0x–01:3x）：新登记 TM-010

技术秘书侧无新用户任务；TM-001～TM-009 的对外动作均待用户逐项授权、无并行会话占用，按规则从掘金 B 类选题。选了 rLLM 的**训练路径**（TM-009 是同一仓库的工具层）。

**结论**：Fireworks 后端自 PR #934（10-09）默认 `router_replay: R3`，但路由矩阵装不进训练帧——`transform.py:85-87` 用 `model_copy(update=...)` 塞的属性不在 `tinker.ModelInput` 声明字段里（只有 `chunks`、`from_ints()` 无该参数、pydantic frozen+forbid），赋值抛 `ValidationError`。三个 tinker 版本一致。上游 cookbook 用的 `from_ints(routing_matrices=...)` 在所有可用 tinker 版本里也不存在 ⇒ **不是 rLLM 单方面能补的**。

**排除了两条先怀疑的方向**（不写进结论）：多步矩阵布局错位（两形状实测都对齐）、verl 蒸馏 `compute_logprobs` 缺失（实为已定义）。

**未验证**：Fireworks 服务端对未声明属性是忽略还是报错；是否已影响线上训练。**不主张训练出错或性能损失。**

入口：[取证](open-source/rllm/fireworks-r3-routing-matrices.md) · [复现](experiments/rllm-fireworks-r3-replay/readme.md) · [TM-010](queue_tech-miner.md)。对外动作全部未做。

### 上一轮（2026-11-03 00:0x）：TM-009 材料齐备

M1 上一轮把 TM-009 从"已取证待决定"推到"材料齐备待决定"：生产可达性结构化核实（`prod_reachability.py`，`use_async=` 在 `__main__` 外 0 处）、查清 rLLM 两套工具栈、候选补丁就绪并暴露"上游 `test_python_tool` 靠缺陷吞异常才通过"。细节见[取证](open-source/rllm/tool-use-async-autodetect.md)。

### M5 本轮（2026-11-03 03:0x–04:3x CST）：第五十六批 §13.52 + 第五十七批（stale 逐出补测）

**§13.52 `max_inflight_gen_batches=3` 已测**：GPU 4,5（同 socket＝2+0）、同一 patched 树、DAPO filter on、2.0／0.1 交替各两跑、15 步/跑、4/4 exit 0（18:59–19:39 UTC）。池化 gen **8.589→7.357s（−14.3%）**、整步 **26.964→25.738s（−4.5%）**、生成组 **113.7→133.0（+17.0%）**；其余相位 |Δ|≤0.5%。臂内 FE 斜率 38.8／44.2 ms/组，固定等待 −1.97／−2.04s、多生成 +0.75／+0.83s，残差 ≈−0.01s。跨深度 +5.3%→+13.1%→+17.0%、净收益 −0.96→−1.29→−1.23s ⇒ **depth 2 之后饱和**；depth 2→3 增量 <1 SE，不外推。**方法订正**：跨臂共斜率被处理效应压低（21.2 vs 38.8），同族分析按臂内固定效应报。归档 [experiments/verl-v1-poll-quantum-dapo](experiments/verl-v1-poll-quantum-dapo/README.md#补测max_inflight_gen_batches3-2026-11-03m5-第五十六批)，四份日志 md5 与服务器一致。

**昇腾状态（本轮复核，未可用）**：VPN 网关 `10.205.0.1` 本轮 2 包全丢，回到不可达；`192.168.13.154/155/156` 仍不可达。无新增 NPU 任务需要用户处理。

**§13.53 stale 逐出通道已测**：threshold=2 冒烟 4 步**零逐出**（span 恰为 2，warmup=1 的自然跨度就是 1–2）；threshold=1 才触发——每步丢弃上一步整批 32 组、staleness=2、生成组/步 64。四跑 A/B（2.0／0.1 各两跑，15 步，4/4 exit 0，19:53–20:29 UTC）：2.0 臂 gen 严格两态 2.43/2.47 与 4.43/4.39（差 2.00/1.92s），撞量子 5/12、8/12 步；0.1 臂 0/12。池化 **gen 3.506→2.786s（−20.5%）、整步 22.473→21.997s（−2.1%）**；2.0 臂两跑整步差 0.835s、0.1 臂差 0.049s ⇒ −0.475s ≈1 SE，只作方向性证据。**逐出量与 poll 无关（两臂都 32 组/步）**；threshold=1 本身每步多花约 1.1–1.5s，轮询收不回被丢弃的批次。归档 [experiments/verl-v1-poll-quantum-stale](experiments/verl-v1-poll-quantum-stale/README.md)，md5 与服务器一致。

### M5 本轮补记（2026-11-03 04:1x–05:3x CST，第二会话）：第五十六批独立复核 + main 树运行时冒烟 + `wait` 臂

**独立复核第五十六批**：另写 `verify-inflight3.py`（不同解析路径、按跑分层 bootstrap），逐跑/池化/生成组/有效率全部与 `summary-inflight3.json` 在 1e-12 内一致；md5 12/12、服务器 raw 与本地 4/4 字节一致；四跑均 15/15 步。臂内 FE 斜率 38.79／43.69 ms/组、固定等待 −1.98／−2.07s（原值 −1.97／−2.04）。**口径提示**：`observed = fixed_wait + generation_cost` 是定义式，残差≈0 不算独立证据；独立的只有斜率 CI（36.2–40.8 ms/组）与"跨臂共斜率被压低"。入口 [verification-inflight3.md](experiments/verl-v1-poll-quantum-dapo/verification-inflight3.md)、`verify-inflight3.py` exit 0。

**TM-008 补丁在 main `3fdf2b54` 的运行时冒烟**：浅克隆 main `3fdf2b548882` ＋ `patch -p1 --fuzz=0`（3/3），GPU 4,5、DAPO filter、poll 0.1、inflight 2、3 步，**3/3 exit 0**（20:38:50–20:43:26Z）；解析配置 `'poll_interval': 0.1`，三步均带逐出。`_build_replay_buffer` 全树仅 `trainer_base.py` 一处定义、子类不覆写。**两处环境适配与补丁无关**：① TransferQueue 需显式 `total_storage_size`（main yaml 的 null 与 venv 版本错配，`math.ceil(None/n)` 抛 TypeError）；② `single_controller/base/worker.py` 本地 device-map 适配（非连续掩码 4,5 否则 `invalid device ordinal`）。**非计时测量**；`3fdf2b54` 上的效应量仍需完整 A/B。入口 [README](experiments/verl-v1-poll-quantum/README.md#补丁在-main-3fdf2b54-上的运行时冒烟2026-11-03m5-第二会话)。

**§13.53 补记：`max_off_policy_strategy=wait`（dropless）**：同配置（colocate_async、threshold=1、GPU 4,5、15 步、2.0／0.1 各两跑、4/4 exit 0，20:53–21:28Z）。**零逐出**（`evicted_samples` 缺失、spans／staleness 恒 1），生成组/步 **32**（drop 为 64）。2.0 臂 gen 严格两态 **4.24／2.22（差 2.02s）**，0.1 臂 1.85–2.51。池化 **gen 3.151→2.177（−30.9%）、整步 22.323→21.079（−5.6%）**；2.0 臂跨跑差 0.408s、0.1 臂 0.029s ⇒ 强于 drop 的 ≈1 SE。**跨策略（相邻窗口，方向性）**：wait 在 2.0／0.1 两档都比 drop 快（22.32 vs 22.47；21.08 vs 22.00）且生成量减半 ⇒ wait 在本窗口支配 drop。归档 [stale 实验](experiments/verl-v1-poll-quantum-stale/README.md)，md5 与服务器一致。

**wait 臂独立复核（第二会话，只读）**：另写 `verify-wait.py`（逐 token 切分、按跑分层 4000 次 bootstrap），逐跑/池化 gen、step 与 `summary-wait.json` 在 **1e-9** 内一致；md5 12/12、服务器 raw 与本地 4/4 字节一致；四跑 15/15 步。bootstrap（步级、按跑分层）Δstep **[−1.786, −0.724]s**、Δgen **[−1.392, −0.566]s**，两臂只有两跑 ⇒ **没有 run 级 CI**，−1.24s 只能说窗口内稳定。**代码层两条独立证据**：① `_stale_terminal_keys` 在 `strategy != "drop"` 时直接返回空集 ⇒ wait 永不逐出，`evicted_samples` 键是**缺失而非 0**；② `_has_enough_samples` 用 `>= threshold`，threshold=1 时任何在飞批次都阻塞采样 ⇒ **完整屏障**，`spans==1` 是结构必然不是经验发现；threshold>1 的 wait 行为不同、未测。**口径**：跨策略表同时变了策略与每步生成量（32 vs 64）、且相邻窗口，只作方向性。入口 [verification-wait.md](experiments/verl-v1-poll-quantum-stale/verification-wait.md)、`verify-wait.py` exit 0。

### M5 本轮（2026-11-03 06:5x–07:4x CST）：第五十八批 §13.54——separate_async 跑通，轮询量子不进关键路径

**GPU 现场（启动前核对）**：GPU 0 为 liyaowei 的 Embedding EngineCore（20.8 G）、2/3/6/7 为其 4 卡 Qwen3-VL-32B vLLM（各 23.3 G，已跑 5.4 h）⇒ **只有 1/4/5 空闲，4 卡 `inflight=1` 的窗口仍不具备**。改用 2 train（4,5 同 socket）＋1 rollout（1）。**昇腾**：网关 `10.205.0.1` 与 `192.168.13.154` 本轮各 2 包／1 包全丢，仍不可用；只记一次、未重试，无 NPU 任务待办。

**做了什么**：TM-008 清单上最后一个未测的 trainer mode。上游自带 `tests/special_e2e/run_v1_separate_async.sh`（此前只查了 `examples/`），其校验允许 2 train + 1 rollout（训练卡偶数、rollout 卡被 TP 整除，TP=1）。2 步冒烟先撞两处本机环境限制，均非上游缺陷：① `nccl_checkpoint_engine` 加载期 `import cupy`，venv 没有 → 装到 `/tmp/cupy-pkgs`（`/home` 只剩 8 G）；② **`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` 在本模式不可用**——`ray.util.collective` 的 `barrier()` 在 `_used_gpu_indices` 为空时按本进程可见卡数建通信器，NOSET 下每 actor 见 3 张卡 ⇒ 每 rank 都占 0/1/2，报 `Multiple Ranks are using the same GPU`；去掉 NOSET（Ray 单卡掩码）后训练侧 FSDP 广播撞 `Cuda failure 217 'peer access is not supported'`，`NCCL_SHM_DISABLE=1` 绕开。代价：训练集合通信退化为 socket，**绝对步时不可与其他批次比**。

**结论（4 跑、8 步、2.0／0.1 各两跑、4/4 exit 0，23:15–23:31Z）**：稳态 `timing_s/gen` **0.031–0.038 s，两档无差别**；整步 **8.211→8.201 s＝+0.010 s（t=0.12）** ⇒ **poll 不在关键路径上**。机制：训练步 8.2 s（old_log_prob 3.1＋update_actor 3.0＋update_weights 1.85），rollout 池单卡生成 16 条/步远快于此，队列始终非空，`ReplayBufferAsync.sample()` 不进入睡眠循环；step 1 的 4–6 s 等待是 warmup 且与 poll 无关（0.1 臂两跑 5.73 vs 0.03 s）。`spans`／`staleness` 恒 1.00／1.00。**TM-008 的模式覆盖至此完整**：收益是"训练步是否阻塞在 `sample()`"的函数——固定批 −3.2%、DAPO −3.4／−4.9%、4 卡 −4.0%、colocate −1.9%、stale drop −2.1%、wait t=1 −5.6%、wait t=2 ≈0、**separate_async ≈0**。归档 [separate 实验](experiments/verl-v1-poll-quantum-separate/README.md)，`verify.py` 34 项全 PASS、md5 5/5、`summary.json` 入库。

**边界**：单栈单点（0.5B／gsm8k／单 seed／8 步／每臂两跑）；本配置是 rollout-ahead 区间（1 rollout 卡 vs 2 训练卡），rollout 成为瓶颈时会空队列、量子可能回到关键路径，**该区间未测**；rewards 池化为 0（0.5B 在 256 token 内做不出 gsm8k），训练侧偏轻。

### M5 上一轮（2026-11-03 02:0x–02:4x CST）：第五十五批 §13.51

**TM-008 的 async 分支已测**：`trainer.v1.trainer_mode=colocate_async`（warmup=1）、2 卡（GPU 4,5 同 socket＝2+0）、同一 patched 树（`replay_buffer.py` 与 pristine 逐字节相同，只加配置接线），poll 2.0／0.1 交替各两跑、15 步/跑、4/4 exit 0。现场：liyaowei 的 4 卡 vLLM（GPU 2/3/6/7）＋1 个 EngineCore（GPU 0）占其余，仍只有 3 卡空闲；`/home` 仍满（8.0 G 余），全部写 `/tmp`。

**三个结论**：① 量子在 async 同样存在——2.0 臂 gen 两态 **0.22s vs 2.23–2.33s**，8–9/12 步命中（sync 2 卡 6/12）；② **2.245s ≠ 浪费 2.0s**——0.1 臂 0/12 命中、gen 落到 **0.75–1.89s**，量子只是把真实等待向上取整，超额 0.1–1.3s/步；③ 池化 gen −16.7%（−3.4 SE）、整步 21.212→20.809s＝**−1.9%**（run 级 ≈2 SE；单跑 −3.0%／−0.7%），整步差只有 gen 可归因。**补丁适用面不是「仅 sync」**；async 也没有消掉等待（仍 1.4–1.7s/步）。归档 [experiments/verl-v1-poll-quantum-async](experiments/verl-v1-poll-quantum-async/README.md)，`verify.py` exit 0，四跑日志 md5 与服务器一致。

**昇腾状态变化（未可用）**：910C VPN 网关 `10.205.0.1` 第 28 天来首次 ping 通（0% 丢包、6ms），但 `192.168.13.154/155/156` 仍 100% 丢包、SSH 22 端口超时——链路只恢复到网关，NPU 主机仍不可达，不据此认为可以开工；无新增 NPU 任务需要用户处理。

### M5 上一轮（2026-11-03 00:5x–01:4x CST）：第五十四批 §13.50

真实 rollout 的响应长度结构实测（gsm8k `train_sub480` × n=8，1.5B／0.5B 五档，19,200 条响应）：构造档 `realistic=(0.60,0.08)` 隐含 ICC ≈0.98，实测 **0.450 [0.409,0.489]（1.5B cap256）／0.364 [0.321,0.407]（0.5B）**，且真实组内离散大于组间（0.166 vs 0.161）；`max_response=256` 截断 45.3%／60.8%，干净组重算的 0.174／0.128 是选择偏差不可引；回代 §13.38 两主读数在 1 SE 内不动（`ldm/grouped` 1.0462→1.044–1.048、`interleaved/ldm` 1.0045→1.0027–1.0065）。归档 [experiments/rollout-length-structure-real](experiments/rollout-length-structure-real/README.md)，`verify.py` exit 0。

| 执行者 | 当前任务 | 接续入口与核对事项 |
|---|---|---|
| M1 | TS-001 第十批（核实未提交的分母拆分；四档 `sign_regime` 进归档；订正 §13.30 第五批「cap256 各格符号混杂」的判词——**结论不变、判词要改**）完成；此前第九批（分桶探针与真实负载纳入逐值对照；两个探针脚本的硬写容差接入共享工具）；此前第六批（同一族配对差判据抽成 `tools/paired-diff`；`wall_ratio` 的阈值拆解，§13.47.2）；此前第五批补遗（阈值摆幅拆成同水平步数差，§13.47.1 补遗）；此前第五批（C 类账的收敛阈值带宽）；此前第四十二批（真实负载上的策略位移分档）；TM-010／TM-009／TM-011 待用户决定 | 取证、上游查重、可达性复核、候选补丁均完成：[取证](open-source/rllm/tool-use-async-autodetect.md)、[可达性探针](experiments/rllm-tool-use-async-probe/prod_reachability.py)、[候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)。**对外动作全部未做**。**第六批**：五个脚本改接共享判据后五份归档 `--verify` 逐字段不变、测试 121 passed；`wall_ratio` 摆幅 2.2–9.5pp，总量比穿过 1.0 的 `replay` 逐种子方向仍稳（18–20/30），真正翻的只有 `C vs B`，`cap` 是 14–23/30 种子无分辨率。**第四十批把 §13.30 第一批的六个 `R` 读数作废并按正确组臂重算**（43.7／54.7／20.6／56.6／27.7／38.6 → 34.0／42.4／14.7→44.9／17.3→34.0），第二／三／四批不受影响。**第四十一批把分桶探针的 16 格读数按符号稳定性分档**：只有 cap1024 ≥4 批六格稳定（19–20/20 同号），cap256 各档符号在种子间翻转 ⇒ **不再引用 cap256 的均值当效应**；并核出 TM-011 立项前置是**改造排程器**（容量与批绑定、同步到达、单批内乱序零效应），不是加分桶逻辑。**第四十二批把同一分档判据补到真实负载**：FIFO 基线自身换序就动 0.7–11.7pp，§13.30 第四批 cap256 的"反号"（−0.95／−0.39pp）落在带宽内⇒**不可引用、方向未定**；可引用的只有 cap1024（cv 0.30–0.36）六格 28–30/30 同号。`R` 的量级读数（2.7%／11.4%）不受影响。TS-001 的可做补强已到边际：纯归档重算、语义面排队顺序维、真实长度负载维都已纳入。语义面剩余缺口仍是**长度预测误差／分桶／迁移／抢占／KV 搬运**，P-1 跨系统需用户立项。 |
| M5 | TS-001 算力支持 | **第五十八批 §13.54：separate_async 跑通并测完（4/4 exit 0，整步差 +0.010s ⇒ poll 不在关键路径）**；此前第五十六批 §13.52、第五十七批 §13.53（drop）与 §13.53 补记（wait t=1／t=2）、第五十六批与 wait 臂独立复核、main `3fdf2b54` 3 步冒烟均完成（见上）。TM-008 剩余未测只剩 4 卡 `inflight=1`（需 ≥4 卡空闲窗口）与 `wait` 的 t≥3 档（结构上等同默认 async）。**separate_async 的两条本机前置必须记住**：不能设 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`（checkpoint 组会撞「多 rank 同卡」），且需 `NCCL_SHM_DISABLE=1` 规避训练侧 CUDA 217；cupy 装在 `/tmp/cupy-pkgs`。P-1 跨系统（AReaL／OpenRLHF／SkyRL，需用户立项）；P-2 FP8 训练侧（待实测）。**下一轮先核对 GPU 归属与占用再启动，不据旧记录假定空闲。** |


## 未完成操作与并行工作

- **带宽为 `0.0` 不一定是「基线很稳」（2026-11-04 第十一批新增）**：`exceeds_band(|效应|, 0.0)` 对所有非零效应返回 True，所以 `fifo_band = 0` 的格子最容易被读成「效应远超基线抖动」。0 有两种成因，判词相反：`no_reading`（**每个种子**的 `R` 都恒为 0，打乱顺序也不变 ⇒ 这一格问不出来）、`degenerate`（过半种子带宽为 0 ⇒ 带宽由少数种子撑起）。档名在 `real-lengths-summary.json` 的 `real_arms.*.sign_stability.*.band_floor`，可 `python3 band_floor.py` 复算并逐格交叉核对（不一致 exit 1）。**引用任何 `|效应| > 带宽` 之前先读这一档**；带宽是调用方算出的一个数，`paired_diff` 收到时已分不出是哪种 0，所以判据不在工具层。**已知未做**：`no_reading` 只说明「这一格问不出来」，没有回答「换一种抽样（放宽 `n_groups`／换负载档）能否问得出来」。
- **截断会同时压扁两个维（同批）**：cap256 的组最大 86.0%／94.2% 恰等于 `max_response`（cap1024 只 0.2%／4.8%），于是 ① 组间齐平 ⇒ 组级排队顺序无事可排（第三十九批）、② 16 组组最大全等 ⇒ 基线带宽为 0（本批）。在截断负载上做「组级排序」实验，连噪声都测不到——这是 TM-011 若立项必须选不截断负载档的理由。
- **推送通道时通时不通（2026-11-03 22:2x 实测）**：22:2x 时 `git push` 一次成功（积压 16 个提交 → `878600d56`）；同一次会话内随后**两次 push 均 75s 超时**，改用 [git-push-api](tools/git-push-api/readme.md) 推到 `a7b862748`，本地 `git reset --hard` 对齐（树 `e6acaa5f`，与本地原提交同树不同 SHA）。⇒ **下次先试直连，超时就换 API；API 用前必须先 `push_dryrun.py` 预演，并把父提交写成完整 40 位 SHA**（短 SHA 会被 API 返回 422；当前远端有 `878600d56` 与 `be37df1a6` 两条历史线，写错父提交会报 "not a fast forward"）。下面这条"推送受阻"记录保留作历史。**2026-11-04 04:4x 复核**：直连一次成功（`9becde235..f685d58f9`），`origin/main..main` 归零。
- **[历史] 推送受阻（2026-11-03 18:0x–22:2x）**：两条通道都不通——HTTPS 报 `self signed certificate`（`curl` 同样，是中间设备替换证书而非 git 配置），SSH 22 端口 `Connection timed out during banner exchange`；`gh-proxy` 亦不可用。**本地积压 11 个提交**（凭据安全链 6、文档导航 2、并行会话 3）。
  **恢复后依次执行**：① `git fetch origin` 确认远端无冲突；② `git push`；③ 逐项核对这 11 个提交在远端存在。
  **不要**用 `--insecure`／`http.sslVerify=false` 绕过——那等于接受未验证证书。
  **不反复重试**：每次重试耗时 75s 且无收益，等网络自行恢复。
  **M1 20:2x 复核**：仍不通（`git ls-remote` 报同一句 `SSL certificate problem: self signed certificate`）。**本地积压 13 个提交**（含并行会话的 `60fbf6939`）。恢复后按上面三步执行，并把"核对的 11 个提交"改成实际数量。
- **"零分歧"要问这份输入在检验什么（2026-11-03 第八批新增）**：96 列"真实输入零分歧"里 **40 列是整数步号**，旧版 `d == 0` 与新版 `|d| <= TOL` 在整数输入下精确等价，**这些列根本没检验容差**；真正检验容差的只有 56 列浮点比率，相对 1 翻号的只有 2 列。⇒ 报告判据通过时必须同报**覆盖面**，否则"零分歧"会被读大。同族的第二件事：`TOL` 是**绝对**容差，同一列比率在原量纲（`v-1`）与 pp 刻度（`100*(v-1)`）上容忍的偏差**差 100 倍**（`across_scale()`），一个刻度的 `n_identical` 不能解释另一个刻度。
- **舍入口径要写明（2026-11-03 第八批新增）**：归档第 3 位小数用内建 `round`，对**精确平局取偶**（`0.3125 → 0.312`，人类读法是 0.313）。实测真实输入落在平局上的个数为 **0**（差值来自浮点除法与求和，末位不会恰好落 5），所以不影响已发布读数；但新数据若出现平局，第 3 位取决于用哪一种，引用前须确认。
- **判据里的常数要有自己的边界测试（2026-11-03 第六批新增）**：把共享判据的容差 `TOL` 从 1e-12 注入成 1e-3，**113 个测试全绿**——既有测试的配对差一律取 0 或明显非零，从没造出"小于 1e-3、大于 1e-12"的差值。补 `TestSharedTolerance` 三条边界测试后才被测红。**另两条同批教训**：① 比率读数的"翻号"参考点是 **1 不是 0**（`steps_ratio` 0.978→1.005 明明翻了，对零却在同侧），抽公共判据时机械替换会静默丢语义，用 `sign_flip_around(values, 1.0)`；② `def f(tol=TOL)` 把值绑进默认参数，改 `TOL` 不生效，共享库的函数签名一律 `tol=None` 并在体内解析。**注入时先确认改的是被执行的那一行**——本族已发生**三次**：第六批 `str.replace(...,1)` 改的是文档字符串里相同的句子，常量没动，「注入没抓到」是假结论；第七批又重犯一次，改成按 `line.startswith("TOL = ")` 定位真实赋值行才抓到。⇒ 注入脚本一律先打印被改的行号与原文。
- **同号占比的分母要拆开（2026-11-04 第十批新增）**：`same_sign_fraction` 把 identical 种子留在分母里，读数恒 ≤ 只看表态种子的口径。**「低 ssf ⇒ 方向混杂」是过度概括**——真实档 10 个带重复的格子里 7 格表态种子其实同号（方向一致、分辨率薄），只有 3 格真混杂。四档 `sign_regime()`（`stable`／`mixed`／`consistent_but_thin`／`no_resolution`）已进两份归档；引用配对差时报档名，不要只报一个 ssf。**同族第二条**：带宽为 0（该格 `R` 恒为 0）时 `exceeds_band` 恒真，**噪声为零不等于基线很稳**——先看带宽是不是 0 再谈是否超带宽。门槛 0.5／0.8 是判读门槛，**未用已知真相的模拟校准过假阳率**。
- **均值不能代替符号分布（2026-11-03 第四十一批新增）**：分桶探针报的是 20 个种子的**均值**，而同一个均值既可能来自"全同向"，也可能来自"一半正一半负正好抵消"。引用任何配对差之前必须同时看**同号比例**与**种子间 sd**——本轮据此把 16 格读数劈成两半（cap256 各档 3/13、13/7、10/8、5/15、10/10，全部不能当方向性结论）。第二个判据是**基线自身的抖动**：FIFO 臂的读数随传入顺序就能动 1.4–5.5pp（同组集合只换序），效应小于这个带宽时不该下结论。`effect_exceeds_band` 只是**判读门槛不是显著性检验**，且每格仅 20 种子、未做 CI／置换。
- **设置项也要先验预期值（2026-11-03 第五批补遗新增）**：负向对照"把曲线压平到点之后、摆幅应为 0"**在数学上不成立**——压平按到点那一步的值取值，抹掉此后所有新高，更高阈值再也跨不过去，所以它必然报大摆幅。这条曾被当成代码 bug 查了一整轮。判据：**先独立算出"正确实现下这个输入该给什么"，再拿它当断言**；否则检查永远红、而红的原因会被归到实现上。要问"快慢几步"读**同水平步数差**（逐种子"到阈值步数(臂)−(基线)"），`steps_ratio` 只用于同一阈值内的相对比较；p4 的 0.3–3.4 步落在 ±1.5 步分辨力内，不可再细读。
- **凡配对差必须同时报符号分布与基线抖动（第四十二批扩展，覆盖构造档与真实档两处）**：真实 rollout 负载上，FIFO 臂**组集合固定、只换传入顺序**，`R` 就动 **0.7–11.7pp**（4 批档中位 2.8–4.0pp）；§13.30 第四批的 +1.99／−0.95／+0.44／−0.39pp **全部落在带宽内或带边**，且 cap256 低积压有 11–19/30 个种子两臂读数完全相等（`n_identical`，无分辨率）。另外核过"到达序"这个假设**不是**原因：换成随机序配对差几乎不动（逐种子同号 23–30/30）。⇒ 判读顺序是：**先看 `n_identical` 有没有吃掉大半种子 → 再看同号比例 → 再看 |效应| 是否超过带宽**；三步任一不过就只能写"方向未定"。带宽是判读门槛，不是显著性检验（未做 CI／置换）。
- **问"分桶有没有用"之前先问工具能不能问（第四十一批新增）**：本排程器三条结构约束使"哪些组进同一批"**不可变**——`cap = n_groups * n_per`（容量=单批轨迹数）、`arrival = b * batch_span`（同步到达）、单批内乱序对 `group_makespan` 零影响。故任何"排序"实验测到的都只是排序，**测不到分桶**。TM-011 若立项，前置是改造排程器（独立到达流＋批窗口可分离＋容量与批解绑），未改造前不接分桶分支。
- **回退只能按路径（2026-11-03 第三十八批新增）**：`git checkout -- .` 会把**未提交的改动**一起抹掉（本轮实际发生一次，cap 修正与测试全丢）。回退自己的试探性改动必须写全路径 `git checkout -- <file>`；动手前先 `git status --short` 确认哪些是你的、哪些是并行会话的。
- **负向对照必须在「把被测代码改坏」的条件下运行（2026-11-03 第三十八批新增）**：只读产物（归档 json）或只测常量映射都不算负向对照——当天两版测试在撤掉修正后照样全绿。断言要落在被改坏的那根线上；检查器读的键先去归档里核一遍存在（第 34 项就读了个从不存在的 `intra_group_cv`）。
- **别人说某条路径坏了，也要两边各算一遍（第四十批新增）**：并行会话的提交说明说旧组臂"量出负税 −33%"，我照抄进正文并当负向对照，**测试连红两轮**才发现那是参差输入的例子、与组臂分组方式无关。判据：**任何"某路径给出怪值"的说法，先在同口径下把旧路径与新路径各跑一遍**，再看怪值出现在哪一侧、是否换号；负税在等大组上结构性不可能（旧路径只会让组臂更快）。
- **推送前必须预演（2026-11-03，已发生两次误删）**：`tools/git-push-api/push_via_api.py` 在 base 不在本地时会把「远端有、HEAD 无」判为删除。**工作区有文件 ≠ HEAD 有文件**。推送前先跑 `python3 tools/anchors/push_dryrun.py <远端 main sha>`，退出码 1 就不要推。两次事故：各误删 7 个实验文件与 M5 的多条记录（均已恢复）。正确顺序：下载/修改 → `git add`+`commit` → 预演 → 推送。
- **TS-001 第三十九批（M1 2026-11-03 11:3x–12:1x）**：排程器加了等大组守卫（`_makespan_by_policy`／`grouping_tax` 参差即 `raise`），`summary.json` sha256 未变、两个 `--verify` 逐字段相同 ⇒ 历史读数不受影响。**下次碰语义面"分桶／预测"前必须知道的**：① 真实负载（1.5B cap256）上组最大 cv 仅 **0.07**、组内极差 85 token，**组最大几乎齐平**，所以任何**组级**排队顺序（FIFO/LOF/pending_max/arrive_max）在同一负载上读数可以**完全相同**（试读中为 0.25）——长度预测／分桶改的是组的顺序，在组成本本来齐平时救不了组粒度；② 逐条补位（`lof_flat`，不可实现）的上限差只有 1.3pp（4 批）／3pp（16 批），比构造档小得多；③ **要成篇必须另建模型**（分桶粒度、预测误差如何映射到组序），不能拿 `lof_flat` 当分桶的代理——它是"两臂各自最优"，不是任何可实现的策略；④ 参差组会量出**负税**，而 `R` 被当上界引用：负值只可能来自调用侧违约，不要当读数。
- **TS-001 第三十七批（M1 2026-11-03 08:0x–09:1x）**：§13.30 第四批已落笔，§11.4 表行与第 3 条、§11.5 缺口表与注记均已同步。**下次触碰语义面 `R` 前必须知道的（更新，覆盖第三十六批的口径）**：① 「更强调度抬税」**不是无条件结论**——它只在组内离散度较高（cv≳0.4–0.6，翻转点随积压深度移动）时成立；真实 RL 短答案负载（cap256、cv 0.13–0.16）深积压上**方向相反**（1.5B 11.4%→10.4%，27/30 为负）。② `R` 的**量级**也远低于构造档：真实 cap256 上只有 **2.7%（4 批）／11.4%（16 批）**，构造档是 34.0%／42.4%。③ **ICC 只改量级、不改符号**（组内 cv 固定 0.6 时四档位移恒正）；截断造成的并列也**不是**原因（量化到 5／20 格同号）。④ 机制是两臂空等的相对厚薄：同一负载加深积压，受益方就从轨迹臂换成组臂，**不能拿一档的现象解释两档**（本轮初稿即犯此错）。⑤ 真实档的 16 批积压是**合成负载**，真实训练积压深度无语料报值——反号只在"深积压＋低离散"象限成立，**不能写成"第三批错了"**。⑥ `R` 小 ≠ 拆组无价值（`R` 仍是上界）。语义面剩余缺口 = 长度预测误差／分桶／迁移／抢占／KV 搬运（需新建模型）。
- **TS-001 第三十六批（M1 2026-11-03 07:0x）**：§13.30 第三批已落笔，§11.4 表行与第 3 条、§11.5 缺口表与注记均已同步。**下次触碰语义面 `R` 前必须知道的**：① **不要再写"`R` 会在更强调度下变小"**——在构造档上实测为**变大**（4 批 34.0%→36.8%；但见第三十七批：真实负载可反号）；② 构造档上的正确表述是"调度器补偿绝对 makespan，不补偿 `R`"；③ `arrive_max` 在本模型退化成 FIFO，**η 通道仍是完全未测的**，不得据 `arrive_max` 的读数写 η 无效；④ `pending_max` 与 `lof` 同义，不许当两条证据引用；⑤ 第三批的 34.0%／42.4% 与第一批旧表的 43.7%／54.7% **参数不同，不可相减**；**且那对旧表数已于第四十批作废**——按正确组臂第一批是 34.0%／42.4%，与第三批同值（第三批本来就走 `_makespan_by_policy`）。
- **TS-001 第三十五批（M1 2026-11-03 06:0x）**：`verify_anchors` 覆盖 33 项、全一致。**下次触碰放置效应相关数字前必须知道的**：① 两套分母都合法但含义不同——**组间差（16.3%）只用不绑定的 pA1／pA2 vs pB1／pB2**，**结果五比 2+2 时用四跑均值**（16.8% 对不绑定 A、28.1% 对 B 四跑）；② "3+1 比 2+2 慢 19.1%"以 2+2 为分母，跨归档比较要各按自己那次的 1 卡基线换算整步，**不能拿两个 speedup 直接相除**。**新纪律**：凡"比某组快/慢 X%"必须写明分母；核验报不一致先查检查器。
- **TS-001 第三十四批（M1 2026-11-03 05:0x）**：§20.3.9 已落笔；核对脚本的配对错档已修、负向对照 5/5。**下一次触碰锚② 前必须知道的（更新）**：可发表形式仍是 §20.3.7 那个式子；同栈 **10 个主结论臂**（含 async）F3 都在 0.1% 内。**新纪律**：引用任何 A/B 臂前确认两跑出自同一份归档、同一窗口规则——**形式的闭合不检验这一点**（错配的一对 F3 照样闭合）。锚② 强度「中强」，不回调；跨系统仍归 P-1。纯归档重算这条线已到边际，剩下只有 P-1。
- **TS-001 第三十三批（M1 2026-11-03 04:0x）**：§20.3.8 已落笔，P-6 适用面扩到 async。**下一次触碰锚② 前必须知道的（更新）**：可发表形式仍是 §20.3.7 那个式子（份额取**基线侧**、乘两侧缺口之比），现在可以说它「在同栈 sync＋async 两条轨迹上闭合」，但**仍不能说跨系统**（P-1 全归跨系统）。锚② 强度「中强」，不回调。
- **TS-001 第三十二批（M1 2026-11-03 03:0x）**：§20.3.7／§11.6 已落笔，P-6 完成。**下一次触碰锚② 前必须知道的（更新）**：可发表形式是 §20.3.7 那个式子（份额取**基线侧**、乘两侧缺口之比），不要再写"恒等式已获同栈实测校验"，也**不要**把份额取在目标侧——那是被否掉的旧式。强度「中强」、跨系统仍缺 P-1。
- **TS-001 第三十一批（M1 2026-11-03 02:0x）**：§20.3.6／§11.6 已落笔；`experiments/rollout-anchor2-identity/` 已入库（`check_identity.py --verify` exit 0、`negative_control.py` exit 0）。**推送（2026-11-03 02:1x）**：`git push` 直连 github.com:443 仍不通（75s 超时），已用 [tools/git-push-api](tools/git-push-api/readme.md) 推到远端 main `0eee4eb7`；本地 commit 为 `b1c8e4786`，**两者树相同、SHA 不同**（API 以远端父提交重建），本地尚未对齐——下次运行先 `gh api` 取远端 sha 再决定是否 reset。**推送（2026-11-03 02:1x）**：`git push` 直连 github.com:443 仍不通（75s 超时），已用 [tools/git-push-api](tools/git-push-api/readme.md) 推到远端 main `0eee4eb7`；本地 commit 为 `b1c8e4786`，**两者树相同、SHA 不同**（API 以远端父提交重建），本地尚未对齐——下次运行先 `gh api` 取远端 sha 再决定是否 reset。**推送（2026-11-03 02:1x–02:2x）**：`git push` 直连 github.com:443 仍不通（75s 超时），用 [tools/git-push-api](tools/git-push-api/readme.md) 分两次推到远端 main `9bc7cd05`；本地为 `b1c8e4786`＋`5d0e48b86`（树相同、SHA 不同，API 以远端父提交重建）。**两次推送合起来才完整**——第一次推的是工作树，第二次补上了收尾写进 CONTEXT／diary 的两处；逐文件 blob 已核对一致。**下次运行先 `gh api` 取远端 sha，本地未对齐属已知偏差**。
  **下一次触碰锚② 前必须知道的**：不能写"恒等式已获同栈实测校验"——乘积式重建不出整步比，`G_k≡1` 时给 1。可发表写法是"逐阶段秒数相加 + 计时缺口"。
  **P-6 已于第三十二批补完**（§20.3.7），可发表形式见该节。
- **推送通道（2026-11-03 03:0x 更新）**：`git fetch`／`git push` 直连**已恢复**。M1 本轮先 commit 再 push 被拒（API 分叉），`git fetch` 后与 `origin/main`（M5 第五十五批）合并、冲突按「保留双方、按时间重排」解决，已推至 **`f006fbf94`**，本地 origin/main 一致。约定不变：先试 `git push`，不通再用 [git-push-api](tools/git-push-api/readme.md)；API 推送会重建 commit 对象，与本地同树不同 SHA 属已知偏差。
- **TM-010（掘金 B 类，M1 2026-11-03 01:3x）**：rLLM Fireworks 后端默认 `router_replay: R3`，但路由矩阵装不进 `tinker.ModelInput`（无声明字段 / `from_ints` 无该参数 / 赋值被 pydantic 拒）。**未验证项：服务端行为与线上影响**，不得主张训练出错。对外动作全部未做，待用户决定是否提 issue（以及是否连"默认改回 disabled"一起）。见 [TM-010](queue_tech-miner.md)。
- **TM-009（掘金 B 类，M1 2026-11-03 完成四件套）**：rLLM `Tool.__call__` 的 `use_async=None` 自动探测失效 + 守卫不可达。**对外动作（issue／PR）全部未做，待用户逐项授权**；需用户决定：① 是否提独立小 issue；② 若提 PR，是否连补丁一起（补丁会让上游 `test_python_tool` 变红）。上游 `rllm-org/rllm` main `4e828c81`（2026-11-02 核对，未变）。见 [TM-009](queue_tech-miner.md) 与[候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)。
- **TM-008（掘金 C 类，M5）**：verl V1 `ReplayBuffer.poll_interval` 死值 2.0s。已测：2 卡 sync 固定批 −3.2%、2 卡 DAPO inflight=2 −4.9%／inflight=1 −3.4%、4 卡 DAPO −4.0%、**2 卡 colocate_async −1.9%（≈2 SE，§13.51）**。补丁 3 文件 +8/−1，默认 2.0 不变，对 `3fdf2b54`／`4b26c788` dry-run 均 3/3 通过，并在 `3fdf2b54` 上实跑 3 步冒烟（exit 0，非计时）。**对外动作全部未做，待用户逐项授权**；需用户决定「只加可达性」还是「同时改默认 0.1–0.2」。
- **TM-006（掘金 B 类）**：verl `tiled_mlp` 十引擎无消费点。[补丁就绪](open-source/verl/patches/tiled-mlp-wiring/readme.md)；真机冒烟完成（FSDP2 生效、FSDP1 硬报错、默认梯度检查点下无显存收益）。**对外动作全部未做，待授权**。
- **TM-005（掘金 B 类）**：verl `use_prefix_grouper` 在 FSDP 上不生效但副作用照常。取证与影响量化（DP 负载不均衡 0–6.6%）已入库。**对外动作全部未做，待用户逐项授权**。
- `experiments/verl-v1-poll-quantum-async/`（上轮新增）：`run-async-ab.sh`／`run-chain.sh`／`make-summary-async.py`／`verify.py`＋`raw/`（4 份控制台日志＋md5）＋`summary.json`／`README.md`。服务器侧原始日志在 `/tmp/rollout-poll-async-*.log`、patched 树在 `/tmp/m5-poll-ab/verl`（均非持久）。
- `experiments/verl-v1-poll-quantum-stale/`（本轮新增）：`run-stale-ab.sh`／`make-summary-stale.py`＋`raw/`（4 份日志＋md5）＋`summary.json`／`README.md`；服务器侧日志 `/tmp/rollout-poll-stale-*.log`（非持久）。
- `experiments/verl-v1-poll-quantum-separate/`（本轮新增）：`run-sep-async.sh`／`run-sep-async-ab.sh`／`make-summary-separate.py`／`verify.py`＋`raw/`（4 份 A/B 日志＋2 步冒烟＋md5）＋`summary.json`／`README.md`；服务器侧树 `/tmp/m5-main-smoke/verl`、日志 `/tmp/sa*.log`、cupy 在 `/tmp/cupy-pkgs`（均非持久）。
- `experiments/rllm-tool-use-async-probe/`（TM-009）已入库：`repro.py`／`prod_reachability.py`／`readme.md`／`run-info.json`；`open-source/rllm/patches/use-async-autodetect/` 含补丁与 readme。复现依赖 uv 建的 Python 3.12 venv（系统 3.9.6 因 PEP 604 无法导入 rllm）。
- `experiments/rollout-length-structure-real/`（第五十四批）：`sample_lengths.py`／`analyze_lengths.py`／`reanalysis_length_regime.py`／`verify.py`＋`raw/`＋`summary-*.json`／`README.md`。服务器侧原始文件在 `/tmp/m5-length/`（非持久）。
- `tools/pyflag/` 新增 `scan-rllm-4e828c8.json`，readme 加第五类易错点"三态形参"。
- **`experiments/verl-v1-poll-quantum/`、`verl-v1-poll-quantum-dapo/`、`verl-v1-poll-quantum-dapo-g4/`、`verl-tiled-mlp-fsdp-smoke/`、`rollout-g8-gen-bimodality/`、`rollout-c-class-ledger/` 等均已入库**，不重复登记。
- **上游状态（本轮只读核对，均为历史快照）**：verl main `3fdf2b54`（11-02T11:38Z，未变，04:1x 由 clone 核实）；verl PR #8313 open、`mergeable=true`、head `71fd3562` 未变；`mergeStateStatus` 在 **04:19 与 04:45 CST 两次核对均为 BLOCKED**，M1 04:3x 记为 `unstable`——该字段在两者间跳变，只作快照，不据此推断 CI 结论（非本 Agent 动作）；rLLM main `4e828c81` 本轮未复核。下一轮重新核对，不据此推断实时状态。
- 服务器网络、磁盘、GPU 与上游 PR 状态均是历史记录，不是实时可用条件。M1 不探测远程资源。
- **共享 checkout 被并行会话并发使用**：动手前先看 `git log --graph` 与 `git worktree list`，冲突集中在 CONTEXT.md 与当日日志时按「保留双方条目、按时间重排」解决。
- `diary/2026-08-14.md` 仍为用户未跟踪工作，不代为提交或删除。

## 恢复入口

- 专项：[rollout 主文档](notebook/rollout-performance-research.md)（§13.54 M5 本轮、§13.30 第三批／§11.4／§11.5 M1 本轮）。
- 真机分工：[实验交接](experiments/handoff-m5.md)、[AReaL 单一负责人交接](open-source/areal/handoff-m5.md)。
- 历史追溯：[迁移前 WORK.md 固定快照](https://github.com/Jackie2049/perpetual-learning/blob/76edee31b650d27a110fa6f82df5789abb237cfc/WORK.md)；批次结论仍以专项文档和 [diary](diary/readme.md) 为准。

本文件只覆盖式更新执行现场；任务的目标、状态、优先级和等待条件在队列维护，不在此复制研究历史。
