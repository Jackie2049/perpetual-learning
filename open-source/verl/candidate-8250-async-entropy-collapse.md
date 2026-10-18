# Candidate: async 熵爆炸→坍塌模式诊断 (#8250)

> Issue: [verl-project/verl#8250](https://github.com/verl-project/verl/issues/8250)
> 创建 2026-10-15；[Investigation]（用户明确"未证实是 verl bug"，请求诊断建议）；0 评论；查重 2026-10-19（修正方法学）：**0 PR，真实无人认领**
> 类别：investigation / 评论型贡献（非 bugfix patch）
> 状态：**analysis-done**（诊断草稿已拟，发布需用户授权）

## 0. 为什么值得做

| 维度 | 评估 |
|---|---|
| I | 4 — 多轮工具 agent RL 的熵爆炸+坍塌是社区高频痛点（与 #7899 用户同症状：退化→截断），4 组 run 数据完整可诊断 |
| C | 4 — 评论型贡献：无需 patch/GPU，我们的 GRPO 数学 + 退化分类学直接适用 |
| A | 3 — 非 bugfix，是诊断建议；若建议有效可转化后续 bug 候选 |
| F | 5 — **熵爆炸/退化家族 = 我们的核心深耕区**（Session 10 GRPO/GSPO 推导 + 3 框架退化模式 + σ=0 知识） |
| 分 | 15（评论型；若转化 bugfix 再重评） |

## 1. 症状（4 组 run 数据）

| Run | 配置 | 轨迹 |
|---|---|---|
| A | 27B GRPO token-mean + PPO clip ±0.1 + kl_in_reward 5e-4 | step 64 后 KL 0.019→0.18→1.35→15.6，grad_norm 0.26→7.3→1010→12442，熵 0.60→6.17→0.23，奖励坍塌不恢复 |
| B | 27B GRPO seq-mean-token-mean 无 KL | 熵 0.15→0.33（22 步），记录截止，早期漂移 |
| C | 27B GSPO seq-mean clip 0.0003/0.0004 仅结果奖励 | **稳定反例**：熵 0.41→0.32（80 步），val 奖励 0.77→0.80 |
| D | 9B GSPO seq-mean 同 clip，无 KL，双 reward 组件（max 1.4） | 123 步稳定后 step 125 KL 0.02→0.22，响应长度 2833→233，熵 0.39→4.03→0.58，grad_norm 归 0，四类 result_mean/completed_ratio 全 0 |

## 2. 诊断分析（基于 Session 10 数学 + 跨框架退化分类学）

### 核心机制：KL runaway → 优势离群 → 灾难性更新 → 同质化坍塌

Run A 的时间线是标准前兆序列：
1. **rollout_corr/kl 先涨**（0.0189→0.0385@64→0.1797@70）：异步下 actor 策略与采样策略漂移增大（staleness 增长）
2. **KL 爆炸**（1.345@71→15.63@72）：陈旧轨迹的优势变成离群值（off-policy）
3. **grad_norm 爆炸**（7.3→1010→12442）：离群优势 × 指数比率 → 灾难性参数更新
4. **坍塌**：策略退化 → 输出变短/畸形 → 同组奖励同质化
5. **grad_norm=0**：组内奖励同质 → σ_G=0 → 优势全 0 → 零梯度（退化终点，见下）

### Q3 答案（grad_norm=0）：数学上**符合预期**

GRPO 组归一优势 A=(R-μ_G)/σ_G（σ_G=0 时 verl fallback A=0）。坍塌后四类 result_mean=0 → 同组全奖励≈同质 → **σ_G≈0 → A≡0 → 梯度≡0**。这不是 mask/数据管线问题，是退化终态的自洽信号。GSPO 同理：-advantage×ratio，A=0 则梯度 0。区分方法：打印 `advantages.mean()/std()/max(|A|)` per step——坍塌后应全 0；若 std>0 而 grad=0，才是 mask/管线问题。

### Q2 答案（区分类别）：三层诊断

1. **数值/更新问题**：按步记录 advantage 分布（mean/std/max|A|、|A|>3σ 离群比例）。Run A 的 grad 爆炸必然伴随 |A| 离群激增。
2. **actor-rollout 对齐**：用户建议的 teacher-forced log-probs 对比**足够且正确**（同一 tokenizer/padding 下，actor 与 rollout 引擎同 prompt 的 log-probs 应一致；不一致 = weight-sync/cache 问题，对应我们的 weight-sync 模式家族）。补充：对比 `rollout_corr/kl` 的**变化率**而非绝对值——稳定期 <0.02，爆炸前 1-2 步跳 5-10× 是前兆。
3. **长上下文/融合 kernel**：65k context + "long-context fused fixes" 是新鲜代码路径——检查 fused 路径在长序列的 token 数（batch_num_tokens all-reduce）与 padding 对齐（关联 #8089 形状家族）。

### Q1 答案：已知模式 + 预防

熵爆炸→坍塌家族在我们跨框架分类学中已记录（verl/rLLM/TRL 三框架 REINFORCE 退化；gs≥4 规则；#7899 用户同症退化→截断）。本案例特有组合：**异步 staleness × 无任何 KL 约束 × 奖励后期全 0**。预防建议（不改变实验意图）：
- `use_kl_loss=true` + 小 coef（如 0.001-0.01）作为 dampener——Run A/D 的坍塌前 KL 都已 10× 于稳定期，有 KL 项可显著压低；
- 或 kl_in_reward 从 5e-4 上调（Run A 的 5e-4 在爆炸前 KL 15.6 面前形同虚设）；
- rollout_corr/kl 阈值守卫（>0.05 时缩小 lr 或暂停）——verl 无内置，可脚本外挂。

### Q4 答案：FSDP2/fused 路径关注点

- `loss_agg_mode=seq-mean-token-mean` 依赖 `global_batch_size` 元数据——确认 48×8=384 轨迹的全局计数正确（#7913 家族曾暴露 seq-mean 聚合不变量问题）；
- **GSPO clip 0.0003/0.0004 是极紧窗且来自 verl 官方示例默认（2026-10-19 论文对照修正）**：verl GSPO 在 ratio 空间 `clamp(ratio, 1-ε_low, 1+ε_high)`（core_algos.py），且 `examples/gspo_trainer/run_qwen3_30b_a3b_megatron.sh` 默认即 3e-4/4e-4 → ratio 窗口 1±0.0004（≈ token KL 窗口 ±0.0004 nats）。与 PPO/GSPO 常见 ε=0.2 相差 ~500 倍（GSPO 论文实验 ε 待官方代码确认）。后果：绝大多数 token 的 ratio 在窗外 → 梯度饱和 → **漂移后无法恢复**（解释坍塌持续而非触发）；Run C 同参数稳定说明紧窗本身不必然触发爆炸（奖励/起点不同）。注：geo_mean 的 clip 是 log 空间 clamp（GMPO 论文对照见 notebook/papers/gmpo.md、gspo.md 三向对照表）。
- fused/长上下文路径先跑 teacher-forced log-probs 对齐（同 Q2.3）。

## 3. 草稿状态

评论草稿完成（上述 4 问答案结构）。发布需用户授权 + 用户按需精简（草稿偏长）。若用户后续提供 TensorBoard/CSV，可进一步定位 Run D step 125 的 weight-sync 事件（响应长度 2833→2519 且 KL 10×，疑似同步到异常模型版本——对应我们 weight-sync/cache 家族）。

## 4. 待办

- [ ] 用户授权后发布评论（精简版）
- [ ] 若用户回传数据 → 深入分析 → 可能转化 bug 候选（如 weight-sync 时序）
- [ ] 沉淀：本案例写入 notebook/grpo-training-debugging-guide.md 的 async 熵爆炸章节
