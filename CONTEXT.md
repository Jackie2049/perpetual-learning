# perpetual-learning — 执行接续

更新时间：2026-11-03（M1 13:2x 第四十批补遗：分桶探针配对法再修一处＋`grouping_tax` 组臂修复落地；M1 13:1x 第四十批 §13.30 第一批 `R` 作废重算；M1 12:0x 第三十九批 排程器输入守卫＋分桶缺口备料；M5 07:4x 第五十八批 separate_async）。规则入口：[AGENTS.md](AGENTS.md)；任务入口：[技术秘书](queue_tech-secretary.md) / [技术掘金](queue_tech-miner.md)。

## 当前现场

### M1 本轮（2026-11-03 13:1x–13:3x）：分桶探针的两处口径再修一次，并把 `grouping_tax` 组臂修复与档案一起落地

技术秘书侧无新用户任务；TM 系列对外动作仍全部待用户逐项授权。本轮接的是 `7b2f8db3b`（"组级排序平均降 R 5.95pp"）→ `e5d3386b9`（并行会话推翻为"真实负载上方向相反"）这条接力。**不再改结论，只把探针做对，顺带把上一条遗留的组臂修复提交掉。**

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
| M1 | TS-001 第四十批（§13.30 第一批 `R` 作废重算）完成；TM-010／TM-009 待用户决定是否对外 | 取证、上游查重、可达性复核、候选补丁均完成：[取证](open-source/rllm/tool-use-async-autodetect.md)、[可达性探针](experiments/rllm-tool-use-async-probe/prod_reachability.py)、[候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)。**对外动作全部未做**。**第四十批把 §13.30 第一批的六个 `R` 读数作废并按正确组臂重算**（43.7／54.7／20.6／56.6／27.7／38.6 → 34.0／42.4／14.7→44.9／17.3→34.0），第二／三／四批不受影响。TS-001 的可做补强已到边际：纯归档重算、语义面排队顺序维、真实长度负载维都已纳入。**第三十九批试过**语义面的"逐条补位"上限档（真实负载 1.5B cap256：4 批 2.4%→12.9%、16 批 10.8%→13.8%），**不足以成篇**——真实组最大 cv 只有 0.07，组级排队无事可排；要成篇需另建分桶／预测误差模型，已登记为后续候选。语义面剩余缺口仍是**长度预测误差／分桶／迁移／抢占／KV 搬运**，P-1 跨系统需用户立项。 |
| M5 | TS-001 算力支持 | **第五十八批 §13.54：separate_async 跑通并测完（4/4 exit 0，整步差 +0.010s ⇒ poll 不在关键路径）**；此前第五十六批 §13.52、第五十七批 §13.53（drop）与 §13.53 补记（wait t=1／t=2）、第五十六批与 wait 臂独立复核、main `3fdf2b54` 3 步冒烟均完成（见上）。TM-008 剩余未测只剩 4 卡 `inflight=1`（需 ≥4 卡空闲窗口）与 `wait` 的 t≥3 档（结构上等同默认 async）。**separate_async 的两条本机前置必须记住**：不能设 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`（checkpoint 组会撞「多 rank 同卡」），且需 `NCCL_SHM_DISABLE=1` 规避训练侧 CUDA 217；cupy 装在 `/tmp/cupy-pkgs`。P-1 跨系统（AReaL／OpenRLHF／SkyRL，需用户立项）；P-2 FP8 训练侧（待实测）。**下一轮先核对 GPU 归属与占用再启动，不据旧记录假定空闲。** |


## 未完成操作与并行工作

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
- 本轮 M5 成果：[separate_async 轮询实验](experiments/verl-v1-poll-quantum-separate/README.md)（4 跑＋2 步冒烟、`verify.py` 34 项 PASS、md5 5/5）。
- 上一轮 M5 成果：[第五十六批独立复核](experiments/verl-v1-poll-quantum-dapo/verification-inflight3.md)、[main `3fdf2b54` 运行时冒烟](experiments/verl-v1-poll-quantum/README.md#补丁在-main-3fdf2b54-上的运行时冒烟2026-11-03m5-第二会话)、[wait（dropless）补测](experiments/verl-v1-poll-quantum-stale/README.md#补测max_off_policy_strategywaitdropless2026-11-03m5-第二会话)、[async 轮询实验](experiments/verl-v1-poll-quantum-async/README.md)。
- **本轮 M1 成果**：[§13.30 第四批（真实长度上的 `R` 与策略反号）](notebook/rollout-performance-research.md) · [§11.4／§11.5 同步收窄](notebook/rollout-performance-research.md) · [实验 readme](experiments/rollout-group-split-tradeoff/readme.md) · [读数脚本](experiments/rollout-group-split-tradeoff/real_lengths_policy.py) · [归档](experiments/rollout-group-split-tradeoff/real-lengths-summary.json) · [run-info](experiments/rollout-group-split-tradeoff/run-info.json)。
- 上一批 M1 成果：[§13.30 第三批（调度策略维）](notebook/rollout-performance-research.md) · [§11.4／§11.5 订正](notebook/rollout-performance-research.md)。
- 上一批 M1 成果：[tools/anchors](tools/anchors/readme.md)（33 项）· [§13.42 结果五订正](notebook/rollout-performance-research.md) · [§19.4 第 9 条](notebook/rollout-performance-research.md)。
- **上一批 M1 成果**：[§20.3.9](notebook/rollout-performance-research.md)、[重建形式脚本（含配对自检）](experiments/rollout-anchor2-identity/check_reconstruct_form.py)、[负向对照（5 组）](experiments/rollout-anchor2-identity/negative_control_form.py)、[归档](experiments/rollout-anchor2-identity/README.md)、[§11.6](notebook/rollout-performance-research.md)。
- 上一轮 M5 成果：[长度结构实验](experiments/rollout-length-structure-real/README.md)。
- 上一轮 M1 成果：[§20.3.8 适用面](notebook/rollout-performance-research.md)、[§20.3.7](notebook/rollout-performance-research.md)、[锚② 复核归档](experiments/rollout-anchor2-identity/README.md)、[TM-010 取证](open-source/rllm/fireworks-r3-routing-matrices.md)、[TM-009 候选补丁](open-source/rllm/patches/use-async-autodetect/readme.md)。
- 历史追溯：[迁移前 WORK.md 固定快照](https://github.com/Jackie2049/perpetual-learning/blob/76edee31b650d27a110fa6f82df5789abb237cfc/WORK.md)；批次结论仍以专项文档和 [diary](diary/readme.md) 为准。

本文件只覆盖式更新执行现场；任务的目标、状态、优先级和等待条件在队列维护，不在此复制研究历史。
