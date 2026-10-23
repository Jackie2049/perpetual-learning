# UniRL #523 unified 栈锚生命周期修复 —— L0 真实模块冒烟（M5，2026-10-23）

任务来源：[handoff-m5 #523 专设小节](../../experiments/handoff-m5.md)（用户 2026-10-23 审阅认可 fork PR #1 后指派 M5 测试闭环）。本轮完成 **L0 真实模块生命周期冒烟（核心交付）** 与 **L1 e2e 可行性判定（待资源/待权重）**。

## 物料与字节校验（第一步，已完成）

- 补丁最终来源：项目仓库 `open-source/unirl/523-lifecycle-fix.patch`（**本轮已修正为与 fork 一致**，见下）。
- 校验方法：对 pristine 上游 main `6fd82cba` 的 `unirl/train/stack/unified.py` 应用补丁，结果必须满足：
  - sha256 = `9a55deabdf1821acbfb57d36bb1377418a4880a652c8a28dd7e89163e541a4e3` ✓
  - git blob = `2cced46552f98e8d44cf57df9f91d46945902f33` ✓（与 fork head `04101ad5` blob 一致，`gh api` 拉 fork 文件双向核对）

### ⚠️ 发现的记录偏差（本轮定位并修正）

项目仓库原补丁文件含 **3 行注释**（`# Freeze both parts' anchors ...`），应用到 pristine 后 sha256=`cd465b31…`、blob=`1814dd43…`，**不等于** handoff 记录的验收指纹；而 fork 分支 `04101ad5`（用户已认可）的提交文件**无注释**，blob=`2cced465…` 与 handoff 验收指纹一致。结论：**权威版本是 fork 分支；原补丁文件与 fork 存在同步偏差**（fork 提交早于补丁落库，补丁后来被加入注释但未推回 fork）。已将项目补丁文件修正为 fork 一致版（去注释），修正后 sha256 见 [handoff](../../experiments/handoff-m5.md) 专设小节。修正补丁在 pristine 上复现 fork 指纹已验证（见下）。

## L0 执行记录（4090-1，verl-e2e 环境）

- 环境：Python 3.12.15、torch 2.11.0+cu128、ray 2.59.0（verl-e2e conda env）。
- 源码：`/home/zxw/Tencent-Hunyuan-UniRL-6fd82cb`（GitHub tarball `6fd82cba` 经 M5 `gh api` 下载 scp 至服务器解压 + 应用修正补丁；tarball 同时解出 pristine 对照树）。
- 方法：真实导入 `unirl.train.stack.unified`（import 链含 ray/transformers 等，verl-e2e 齐备）→ `UnifiedModelTrainStack.train_track.__wrapped__`（`@distributed` 用 `functools.wraps`）取真实方法体 → 绑定替身 self 执行。
- **替身清单**（逐项记录）：`fsdp_backend.model` = 真实 torch nn.Module（单层 Linear，重入保护记录外层 eval()/train()）；ar/image 算法 = AlgoStub 三种变体（a no-op prepare_part + recomputes_anchor=False 复刻 shipped GRPO/FlowGRPO/BagelFlowUniGRPO；(b) 记录型 prepare_part 复刻 PPO/DiffusionOPD override；(c) prepare_part 抛 RuntimeError）；`prepare_segment_anchors` = **真实函数**（`unirl/train/stack/anchor.py`，脚本断言 co_filename）；sample/Part/update_planner/_train_one_step/_align_track_inputs/on_rollout_end = 最小替身（Part/Plan 构造依赖过重，按交接允许）；`_train_step_profiler`=None（UNIRL_PROFILE 默认 off 走 nullcontext）。
- 断言结果（patched 模式）：

| # | 断言 | 结果 |
|---|---|---|
| ① | 调用序 = eval → freeze(ar) → prepare_part(ar) → freeze(image) → prepare_part(image) → train | PASS |
| ② | (c) prepare_part 抛异常 → RuntimeError 上抛且 model.train() 被 finally 恢复 | PASS |
| ③ | (a) no-op：两个 part 原样返回（同一对象流入 _train_one_step） | PASS |
| ④ | (b) 两个算法 prepare_part 各被调恰一次 | PASS |

事件序列（patched (a)）：`gen_part(AR)→gen_part(Diffusion)→align×2→model.eval()→prepare_segment(ar)→prepare_part(ar)→prepare_segment(image)→prepare_part(image)→model.train()→train_one_step→on_rollout_end`。

- **pristine 对照**（同脚本 `--mode pristine`，指向未打补丁树）：旧代码事件序列无 eval()/train()/prepare_part，四项 patched 断言全部失败——证明断言对修复有区分度；(c) 失败替身在旧代码下永不触发（prepare_part 不存在）。

脚本 `run_l0_lifecycle.py`（本目录）支持 `UNIRL_REPO=... python run_l0_lifecycle.py --mode patched|pristine`；完整日志 `l0_patched.log` / `l0_pristine.log`。运行中发现的真实行为细节：torch `Module.eval()` 内部即 `train(False)`，朴素包装会重复记录，脚本用重入保护只记外层调用（真实 `self.training` 语义保留）。

## L1 e2e 统一训练冒烟 —— 可行性判定（待资源/待权重）

- 前提核查：**GPU 0/1/7 空闲（3 卡）、2–6 被他人 VLLM 占用** → 无法整段占用 ≥8 卡（hi3_it2i 等需 num_devices=8，bagel 需 32 多机）；**Bagel/HI3 权重未授权/未下载**（HI3 为 MoE，远超单卡 24 GiB；磁盘剩 149G）。
- **判定：待资源/待权重**，恢复条件 = ≥8 卡同时空闲 + 获准权重与存储。
- 已按停止条件停（不下载未授权权重、不为 L1 新租算力、不打断其他会话）；正向回归与 replay 数值验收按交接四步设计在条件满足后执行。

## 边界

- L0 用替身协作者（真实 anchor.py + 真实 train_track 方法体 + 真实 torch 模块），非完整训练；recomputes_anchor=True 的多 micro 收集路径（anchor.py 的另一分支）未在 L0 覆盖——该路径由真实配方/L1 覆盖。
- L1 全部为可行性判定，未跑训练。
- 未对上游 Tencent-Hunyuan/UniRL 做任何操作（铁律不变）；fork PR #1 与上游 PR 仍待用户决定。
