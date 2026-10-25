# verl GRPO End-to-End Data Flow 源码级阅读

> 2026-06-15 | 源码: verl/verl/trainer/ppo/core_algos.py + ray_trainer.py + workers/utils/losses.py + utils.py
> 核心: GRPO=PPO损失+group-relative advantage → need_critic自动False → 3种advantage变体 → rollout_n interleave → ref_in_actor=LoRA → Dr.GRPO=不除std → total_loss=pg+0.001*KL-entropy_coef*H

## 1. GRPO vs PPO: 只改advantage,不改loss

```
PPO: loss = clipped_surrogate(advantage=GAE(λ-return))
GRPO: loss = clipped_surrogate(advantage=group_relative(r_i - μ_g / σ_g))

→ 同一个"vanilla" loss function! (core_algos.py:1278-1371)
→ GRPO不是新loss,是新advantage计算方式!
→ 关键差异: need_critic() → adv_estimator!=GAE → critic自动disabled (utils.py:96-107)
→ GRPO跳过: critic forward + critic backward + value loss + GAE
→ PPO需要: actor + critic + ref → 3个模型 → 3倍GPU内存!
→ GRPO需要: actor + ref → 2个模型 → 但ref_in_actor(LoRA)=共享 → 1个GPU!

→ Total GRPO loss = pg_loss + kl_loss_coef * KL(pi || pi_ref) - entropy_coef * H
→ kl_loss_coef=0.001 (default) → KL在loss端加 → 不在reward端减!
→ GRPO example: use_kl_in_reward=False + use_kl_loss=True → KL只在loss端
```

## 2. GRPO Advantage: 3种变体

### GRPO (loop-based) — core_algos.py:268-331

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, ...):
    scores = token_level_rewards.sum(dim=-1)  # outcome reward: 只取最后一个token的reward!

    # 按prompt index分组 → 同一prompt的n个response形成1个group
    id2score = defaultdict(list)  # {prompt_id: [r_1, r_2, ..., r_n]}
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    # 计算group mean和std
    for idx in id2score:
        scores_tensor = torch.stack(id2score[idx])
        id2mean[idx] = torch.mean(scores_tensor)
        id2std[idx] = torch.std(scores_tensor)

    # ★ GRPO advantage核心公式:
    # norm_adv_by_std_in_grpo=True (原始GRPO):
    #   a_i = (r_i - μ_g) / (σ_g + ε)
    # norm_adv_by_std_in_grpo=False (Dr.GRPO, arxiv 2503.20783):
    #   a_i = r_i - μ_g  → 不除std → 防止梯度消失!

    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]

    scores = scores.unsqueeze(-1) * response_mask  # broadcast到每个token
    return scores, scores  # advantage = returns (outcome-only)
```

**关键细节**:
- 单样本group(len=1): μ=0、σ=1；默认advantage=score/(1+epsilon)，关闭std归一化为score，见[两实现CPU核对](verl-grpo-core-algos-reading.md#单样本与同分组的不同回退)
- outcome reward = token_level_rewards.sum(dim=-1) → 只关心最终结果,不是per-token
- advantage broadcast: 每个token获得相同advantage → 乘response_mask

### GRPO_VECTORIZED — core_algos.py:335-358

```python
@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(...):
    # 同样公式,但vectorized → 不需要loop → 更快!
    scores = token_level_rewards.sum(dim=-1)
    g = as_torch_index(index, device=scores.device)
    mean_g, std_g, _ = group_mean_std(scores, g, eps=0.0, device=scores.device)

    if norm_adv_by_std_in_grpo:
        scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
    else:
        scalars = scores - mean_g[g]

    advantages = scalars.unsqueeze(-1) * response_mask
    return advantages, advantages
```

**性能差异**: GRPO_VECTORIZED用torch操作代替Python loop → 大batch快10-100x!

### GRPO_PASSK — core_algos.py:472-530

```python
@register_adv_est(AdvantageEstimator.GRPO_PASSK)
def compute_grpo_passk_outcome_advantage(...):
    # ★ 只有最佳response获得非零advantage!
    # a_best = r_max - r_second_max / (σ + ε)
    # 其他response → advantage = 0 → 不参与训练!

    for idx in id2scores:
        rewards = torch.stack(id2scores[idx])  # (k,)
        topk, topk_idx = torch.topk(rewards, 2)
        r_max, r_second_max = topk[0], topk[1]
        i_max = id2indices[idx][topk_idx[0].item()]
        advantage = r_max - r_second_max
        if norm_adv_by_std_in_grpo:
            std = torch.std(rewards)
            advantage = advantage / (std + epsilon)
        advantages[i_max] = advantage  # 只有best获得!
```

**GRPO_PASSK特点**:
- 只有best response参与训练 → 极大减少compute!
- advantage = r_max - r_second_max → 鼓励保持领先
- 至少需要2个样本/group → 不适用于n=1

## 3. GDPO: Decoupled Normalization (新增!)

```python
@register_adv_est(AdvantageEstimator.GDPO)  # core_algos.py:361-468
def compute_gdpo_outcome_advantage(...):
    # ★ 不是先sum再normalize → 先per-dimension normalize再sum!
    # Step 1: 每个reward dimension独立normalize → 防止dominant dimension淹没弱信号
    # A_k = (r_k - μ_group(r_k)) / (σ_group(r_k) + ε)
    # Step 2: 加权聚合 → A_sum = Σ_k w_k · A_k
    # Step 3: batch-level whiten → A_final = whiten(A_sum, response_mask)
```

**GDPO vs GRPO**:
- GRPO: sum all dimensions → normalize → dominant dimension控制advantage
- GDPO: normalize each dimension → sum → balanced across dimensions
- 适用: 多维度reward(格式+准确性+推理) → 防止格式reward淹没准确性

## 4. PPO/GRPO Loss: Clipped Surrogate (不变!)

```python
@register_policy_loss("vanilla")  # core_algos.py:1278-1371
def compute_policy_loss_vanilla(old_log_prob, log_prob, advantages, response_mask, ...):
    clip_ratio = config.clip_ratio  # ε for PPO clipping
    clip_ratio_low = config.clip_ratio_low  # asymmetric lower bound
    clip_ratio_high = config.clip_ratio_high  # asymmetric upper bound
    clip_ratio_c = config.get("clip_ratio_c", 3.0)  # dual-clip lower bound

    # ratio = π_new(a|s) / π_old(a|s) = exp(log_prob - old_log_prob)
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)

    # ★ Standard PPO clipped loss:
    # L = max(-A * ratio, -A * clip(ratio, 1-ε_low, 1+ε_high))

    pg_losses1 = -advantages * ratio  # unclipped
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    # ★ Dual-clip PPO (for negative advantages):
    # When A < 0: L = min(-A * clip_ratio_c, max(-A*ratio, -A*clip(ratio)))
    # → 防止negative advantage时ratio无限大 → 稳定训练!

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

**loss聚合**: agg_loss(loss, response_mask, mode="token-mean")
- token-mean: 所有token平均 → 标准PPO
- token-sum: token sum → 大batch更稳定

## 5. End-to-End Data Flow: 从prompt到update

```
ray_trainer.py 全流程:

Step 1: gen_batch = prompts.repeat(rollout_n, interleave=True)
  → rollout_n = config.actor_rollout_ref.rollout.n
  → interleave=True → [p1_r1, p1_r2, ..., p1_rn, p2_r1, p2_r2, ..., p2_rn, ...]
  → 同一prompt的n个response连续排列 → group index = prompt_id

Step 2: rollout → actor_rollout_wg.generate_sequences(gen_batch)
  → vLLM/SGLang inference → 生成response
  → output: {responses, log_probs, input_ids, ...}

Step 3: reward → reward_manager.compute_reward(rollout_output)
  → rm_scores = outcome reward (每个response一个score)
  → reward写入batch.batch["rm_scores"]

Step 4: extract_reward → reward.py:160-167
  → reward_tensor = batch.batch["rm_scores"]
  → 可能包含per-token rewards或outcome-only

Step 5: compute_advantage → core_algos.py
  → GRPO: group_relative normalization
  → index = prompt grouping (uid) → 同一prompt的response归入同一group
  → a_i = (r_i - μ_g) / (σ_g + ε) → broadcast到每个token

Step 6: ref_log_prob → ref_policy_wg.compute_ref_log_prob()
  → ref_in_actor=True(LoRA): 同一个worker → no_lora_adapter=True → 用base model
  → ref_in_actor=False: 独立ref worker → 需要额外GPU!

Step 7: actor update → ppo_loss(losses.py:57-)
  → log_prob = current policy → old_log_prob = rollout policy
  → ratio = exp(log_prob - old_log_prob)
  → loss = clipped_surrogate(ratio, advantage)
  → + KL penalty: kld = kl_penalty(log_prob, ref_log_prob) → kl_loss * 0.001
  → backward → optimizer.step()

★ GRPO跳过: critic forward + critic backward + value loss + GAE
→ 只有actor update → 简单2/3 compute+memory!

★ need_critic() (utils.py:96-107):
  → adv_estimator != GAE → critic disabled → no CriticWorker spawned
  → unless critic.enable=True explicit override → 通常不需要!

★ Data shape through pipeline:

| Stage | Shape | Key Fields |
|-------|-------|------------|
| Dataloader | [train_batch_size] | raw_prompt, reward_model, uid |
| After repeat | [train_batch_size * rollout_n] | Same, interleaved by uid |
| After rollout | [train_batch_size * rollout_n, prompt_len + response_len] | prompts, responses, attn_mask |
| After reward | Same | + rm_scores, token_level_scores |
| After old_log_prob | Same | + old_log_probs, entropys |
| After ref_log_prob | Same | + ref_log_prob |
| After advantage | Same | + advantages, returns |
| Actor input | [ppo_mini_batch * rollout_n] per micro-batch | All fields, no-padding |

★ ppo_mini_batch_size multiplied by rollout_n (ray_trainer.py:1310-1311)
  → effective mini-batch = ppo_mini_batch * rollout_n
```

### 拼批中的形状与引用

**RL collator 保留每条样本的 Python 对象边界，不承担完整字段校验或深拷贝。** 2026-10-22 固定官方提交 `1876b06d0a3e4e71e06230be10af14492ca8a75b` 的 [`collate_fn`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/utils/dataset/rl_dataset.py)：tensor 列用 `torch.stack` 增加 batch 维，非 tensor 列用 `np.fromiter(..., dtype=object)` 建立一维数组，不把内层列表长度当作新维度。

两条 `raw_prompt_ids` 都长2时，直接 `np.array([[1,2],[3,4]], dtype=object)` 仍得到 `(2,2)`；指定 object 不会阻止 NumPy 展开矩形结构。原 collator 则得到 `(2,)`，每个元素仍是原来的 list。变长列表、messages 的 list[dict]、reward_model 的 dict 各占一个 batch 槽，不需要为拼批而统一内层长度。不同形状的 tensor 则在 stack 抛 RuntimeError。

| 层次 | 本机行为 | 含义 |
|---|---|---|
| 非 tensor 外层数组 | 等长/变长列表均为一维 object 数组 | 保持样本边界，不是模型可直接消费的二维 tensor |
| 单进程内层对象 | 修改源 messages 字典，拼批结果同步变化 | 数组容器是新的，内层对象没有深拷贝 |
| tensor 堆叠 | 修改源 tensor，已堆叠结果不变 | stack 输出存储独立 |

字段一致性是另一项契约。两条样本只有第二条含 `optional` 时，输出该列长度1，不会在第一条位置补 None。同一字段一条是 tensor、另一条是 Python list 时，两套容器各收一条；返回的 `{**tensors, **non_tensors}` 让非 tensor 列覆盖同名 tensor 列，结果也只有一行。这些是违反一致字段约定的局部观察，不是正常 RLHFDataset 的训练故障复现。自定义 dataset/collator 应明确每条样本的键集合、同字段类别与 tensor 形状；缺值若要占据 batch 槽，须明确保存为值而非省略字段。

[`RayPPOTrainer._create_dataloader`](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/ray_trainer.py)默认使用该 collator，训练循环随后用 DataProto.from_single_dict 包装结果。[DataProto.check_consistency](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/protocol.py)在同时有 tensor batch 与非 tensor 列时检查各列首维等于 batch size。因此保留2行 ids、只有1行 optional 的输入，不能由“collate 成功”推出“整个数据管道接受”。后置检查仅作源码核对，未构造完整 DataProto；它也不是每个字段语义的完整模式校验。

[复查脚本](../../experiments/scripts/verify_verl_collate.py)AST 选择完整原 collate_fn 和[原测试](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/tests/utils/dataset/test_rl_collate_fn_on_cpu.py)，直接调用1个原测试体通过。另用真实 torch.utils.data.DataLoader、合成 list 数据集，`num_workers=0 / pin_memory=False / shuffle=False` 执行7组检查：等长列表、变长列表、普通非 tensor 值、引用/存储差别、缺字段、类型冲突、不同形状 tensor，均符合上述结果。对象身份结论仅适用于这个同进程条件，不外推 worker 序列化或进程间共享。

```bash
python experiments/scripts/verify_verl_collate.py /path/to/verl
```

Python 3.12.13、torch 2.9.1、NumPy 2.0.2；非完整 pytest，未导入 RLHFDataset/verl 根包或执行下载、tokenizer、StatefulDataLoader、DataProto、训练与设备路径。官方API和本地模块 Git blob 均为 `aad0655b9d97037cbd7a5ee4d4f45a4bfec0040e`；模块 SHA256 `8571ff8527a0a8c5aa0951f248052e3ea7a8386cc236c1324c81e764d5b1ff65`，原测试 `f2816d4c441937e74f303124dfc99b11bd35b96af674f9e97891aa01eb2c66cf`。

### Rollout 完整性与会话计数

**成功输出行数不等于独立 rollout 会话数。** 一个 agent-loop 会话可以产出多行；同一 prompt 需要两个会话时，只有 `prompt_a_0_0`、`prompt_a_0_1` 两行，仍然缺少第二个会话。2026-10-22 读取的 [verl #6641](https://github.com/verl-project/verl/pull/6641) 为 `main_ppo_sync` 增加终态检查，目标是在组不完整时拒绝返回部分数据，而不是改变 GRPO 算法。

固定 head `9e4e85da050cc1cbf156cd8c9d89217c68c4ea66`，API 目标 base `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`，实际共同祖先 `8911f243022aac3558ec7fa8e444cc960ea8d93c`；真正 PR 两文件273增9删。以下对照以共同祖先为旧版，不把目标分支的其他更新混入修复差异。

[`AgentLoopManagerTQ.generate_sequences`](https://github.com/verl-project/verl/blob/9e4e85da050cc1cbf156cd8c9d89217c68c4ea66/verl/trainer/main_ppo_sync.py#L555)分派前给每个根 prompt 保存 `expected_rollouts`：逐 prompt 的 `__rollout_n__` 优先，否则区分训练 n 与验证 n。worker 的 `_run_prompt` 等待本 prompt 的会话 gather 后写 `finished`，捕获异常则写 `failure`；输出为空的 postprocess 不写成功子行。ReplayBuffer 的 add 按字段合并 marker，因此后来的状态更新不会自行移除先前的期望数。这条 manager→worker→轮询链只作源码核对。

[`ReplayBuffer.sample`](https://github.com/verl-project/verl/blob/9e4e85da050cc1cbf156cd8c9d89217c68c4ea66/verl/trainer/main_ppo_sync.py#L300)仅检查目标 partition/step：有 `running` 就继续轮询；全部不再 running 后，拒绝任何 `failure`，并对带期望数的 finished 根比较独立成功 session 数。key 用 `rsplit("_", 2)` 分开 UID/session/index，保留带下划线的 UID，以 set 去重 session，而不是数输出行。

| 原测试中的根与子行 | 旧版 | 新版 |
|---|---|---|
| 期望2会话，0号两行、1号一行 | 源码推导：返回3行 | 原测试：返回3行，并携带根 marker |
| 根 failure，已有一行成功输出 | 返回部分结果 | RuntimeError，是否有期望数都拒绝 |
| 根 finished，期望2、只有0号一行 | 返回部分结果 | RuntimeError，expected 2 / got 1 |
| 根 finished，期望2、只有0号两行 | 返回2行 | 同样拒绝，独立会话仍只有1 |

根 marker 生命周期也属于完整性条件：新版将根 key 放在 `KVBatchMeta.extra_info["rollout_root_keys"]`，cleanup helper 合并子行与根并去重。训练/验证最终清理调用该 helper，防止同一步的多批验证被旧 marker 污染。原测试实际覆盖 helper 的 key 列表、ReplayBuffer.remove 及同 step 两批验证；没有运行真实 TQ 的删除操作。

这里的 fail closed 有明确范围。源码先等待 running 结束才报告失败，没有新增超时、取消或回滚；一个挂住的会话仍可能阻止返回。另5个本机控制输入确认：expected2但零输出被拒绝；finished 缺期望数仍走旧行为且不收集该根；成功 session 超过期望值仍接受（比较是 `<`，非 `!=`）；无根的成功行仍接受；expected0 可返回空批，另一步的 failure 不影响本步。这些控制说明它是带 marker 的数量下界护栏，不是任意元数据的完整验证器；不据构造输入定性正常入口存在新的训练缺陷。

**CPU 复查范围**：[脚本](../../experiments/scripts/verify_verl_rollout_completeness.py)从固定 Git 对象原样隔离完整 ReplayBuffer 和三个新 helper，选择[原测试文件](https://github.com/verl-project/verl/blob/9e4e85da050cc1cbf156cd8c9d89217c68c4ea66/tests/trainer/test_main_ppo_sync_replay_buffer_on_cpu.py)中的7个 replay-buffer 测试、原 fixture 和 FakeKVBatchMeta，保留原断言。pytest 执行候选7过；旧版只选4个拒绝行为的原测试，全部以 DID NOT RAISE 预期失败，脚本核对失败集合/阶段/原因及退出1后成功退出。另5个控制全过。并未把新 cleanup helper 混入旧版。

```bash
python experiments/scripts/verify_verl_rollout_completeness.py /path/to/verl
```

Python 3.12.13、NumPy 2.0.2、pytest 8.4.2；此环境缺 TensorDict/Ray/TQ/OmegaConf，未安装。另2个 manager/TensorDict 原测试未选取，不用字典替身冒充原9项全过。AST 排除框架 imports、延后注解求值，禁用 conftest/外部插件/cache/字节码及 Git lazy fetch。原 fixture 用 `__new__` 初始化锁与本地字典，不启动 ReplayBuffer 轮询线程；因此7项证明的是原方法对测试元数据的行为，不是 manager、真实服务、训练统计或设备实测。

完整文件 SHA256：

```text
共同祖先 main_ppo_sync.py  0cc6a1427253c962e7aa4ac88d634ee349325387dda32c3270d6012300730370
候选 main_ppo_sync.py      975b4b93337324f080ffdd6e97bf5b9489046a0974518d43edf0817c8afe482a
候选原测试                ac80178e95d410dd2f1c78246d6f48834487bdf2215d88b2e10c7b14ced35478
```

### PF PPO 重采样与原 key 回写

整行内部一致，不代表它仍属于输入位置上的那个样本。[verl #7956](https://github.com/verl-project/verl/issues/7956)报告的V1 GAE路径正是这一差别：PF-PPO有放回重采样整个DataProto，而V1只把部分计算结果写回原key。已有报告者的[#7957](https://github.com/verl-project/verl/pull/7957)选择拒绝该组合，并未实现V1重采样支持；不另做竞争修复。

2026-10-23读取PR open、非draft，作者YeonwooSung；固定head `1eb80e2212f24a5b71ca113c3e42797734961e7d`、目标base `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`、实际祖先 `23a341e361e18063b4ec53e1f7cc4eae68f102cf`，相对目标ahead2/behind6。真正diff五文件133增4删。另核官方main为 `a4b9354968705a441fb7da7367924b15b902e771`，不能将未合并的拒绝行为写成主干现状。

#### 重采样必须携带的身份

原[compute_pf_ppo_reweight_data](https://github.com/verl-project/verl/blob/a4b9354968705a441fb7da7367924b15b902e771/verl/trainer/ppo/core_algos.py)从token_level_scores的逐行总分生成权重，`torch.multinomial(..., replacement=True)`取得新行索引；所有batch张量、non_tensor_batch数组以及长度等于batch_size的meta_info列表都按该索引取值。它是选择/重复整行，不是给每个token损失乘一个IS系数。

[legacy compute_advantage](https://github.com/verl-project/verl/blob/a4b9354968705a441fb7da7367924b15b902e771/verl/trainer/ppo/ray_trainer.py)先计算GAE，再在GAE分支中调用该重采样函数，返回整个新DataProto。原[V1 helper](https://github.com/verl-project/verl/blob/a4b9354968705a441fb7da7367924b15b902e771/verl/trainer/ppo/v1/utils.py)将非GRPO estimator委托给它，因此不是“V1忽略use_pf_ppo”。问题出在[V1调用者](https://github.com/verl-project/verl/blob/a4b9354968705a441fb7da7367924b15b902e771/verl/trainer/ppo/v1/trainer_base.py)仍持有原batch.keys和重采样前的nested response_mask；之后只把advantages/returns等字段转换并回写，而响应和logprob没有随之重采样。

本机补充控制用原helper、真实DataProto/TensorDict及原[response_to_nested](https://github.com/verl-project/verl/blob/a4b9354968705a441fb7da7367924b15b902e771/verl/workers/utils/padding.py)执行，B=8、T=4、seed=0：

| 新行位置 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 实际来源行 | 7 | 7 | 6 | 7 | 6 | 7 | 4 | 5 |
| 原位置响应长度 | 1 | 2 | 3 | 4 | 1 | 2 | 3 | 4 |
| 来源行响应长度 | 4 | 4 | 3 | 4 | 3 | 4 | 1 | 2 |

返回的每个张量、UID和行级metadata都与来源索引一致，原输入未被重采样函数改写。但将新行送入原切片函数并传旧mask，得到的是表中原位置响应长度，而非来源长度；函数不核对样本身份。有放回采样还会重复来源7，不能把它当普通一对一置换：若未来真支持V1，重复样本引用、key消费、响应/logprob与长度的共同身份都需定义。本控制没有调用kv_batch_put，不声称观测到真实队列覆盖或完整训练错配。

#### 拒绝的时机与 CPU 验证

#7957只在 `adv_estimator=gae` 且use_pf_ppo为真时，从V1优势helper抛NotImplementedError；legacy整批路径不变，非GAE原本不进入PF-PPO分支，也不因开关为真就获得这项算法。保护发生在第一次优势计算，而非启动校验；此前rollout等成本仍可能发生。一个trainer.use_v1配置值不能替代实际调用路径判断。

[复查脚本](../../experiments/scripts/verify_verl_pf_ppo_identity.py)对祖先、PR head、上述main分别执行[同一份原四测试](https://github.com/verl-project/verl/blob/1eb80e2212f24a5b71ca113c3e42797734961e7d/tests/trainer/test_pf_ppo_v1_on_cpu.py)，保留测试字节和断言。重复运行结果一致：祖先和main各3过、拒绝用例以DID NOT RAISE预期失败；head四项全过。补充的一组身份/长度控制也重复通过，不算进原四项。原用例验证重采样、拒绝、关闭开关时保持行序、legacy仍重采样，并非作者所报其他文件合计九项的复跑。

环境复用 `/private/tmp/verl-8281-venv/bin/python`：Python3.12.13、torch2.14.1、NumPy2.5.4、TensorDict0.15.2、OmegaConf2.3.1、pytest9.1.1，无新增依赖。完整原DataProto类、四个序列化函数、原枚举/注册器、GAE/PF函数、四个masked统计函数及两层优势函数经AST隔离；原方法体和算法不改，延后注解求值，跳过包初始化及框架导入。DataProto显式使用原numpy序列化路径，不走设备发现；张量均在CPU。没有原生verl/测试conftest导入、Ray/TQ、trainer/critic/actor或训练运行，也没有TQ替身。

```bash
python3.12 -B experiments/scripts/verify_verl_pf_ppo_identity.py /path/to/verl-git
```

脚本按版本打印完整来源SHA。原测试SHA256 `26cd9277a15613ffc181b092e6bc46418b5cb22d8b8e4e96c3fcb6e2800a67ae`；三版protocol均为 `8d69430131e9e894a6c7aea673d489e19e44117b1bc573085a6db521b38c6792`，head V1 utils为 `9b82466e08c9d59fd280882c4e440e462d5ea42607181a59727950ca3d48cceb`；完整PR patch为 `14e71a8528d85d73cf7cb043d8ae412ba4fda4799c0f8c915f2f752ab8f4623a`。变更Python另做祖先2份/head3份完整AST/内存compile。CPU数据契约可在本机核实，不因此新增M5算力任务或推断训练收益；现有GPU/AReaL/#8313/#523专项不变。

### 响应长度触顶与真实停止原因

批次中的最长响应不等于配置上限，达到某个长度也不等于因长度限制停止。[#7817](https://github.com/verl-project/verl/issues/7817)已有[#7821](https://github.com/verl-project/verl/pull/7821)，[#7823](https://github.com/verl-project/verl/issues/7823)已有[#7906](https://github.com/verl-project/verl/pull/7906)；2026-10-25读取均open、非draft、未合并，不另实现或自动拼接两候选。#7821 head `7255b49cf2c5c26745865e1881332fc0e41554a2`、实际共同祖先 `a9f2985159536a607211dcac730d3f5d55028950`，两提交三文件231增4删；#7906 head `2b5536028b6a17aa9d2ce5adf893093629f58b51`、父提交／祖先 `5426b9265e73b37a58d238f59b282e7cbf713408`，一提交16文件412增17删。两API目标base均为3f9d4c0d；#7821直接父提交已含主体修复，不能把它当完整修复基线。

[#7821指标函数](https://github.com/verl-project/verl/blob/7255b49cf2c5c26745865e1881332fc0e41554a2/verl/trainer/ppo/metric_utils.py)接收显式prompt／response cap，V1调用方传已解析rollout配置，缺参仍取张量宽度。真实CPU DataProto控制中，prompt长度[3,5,5,2]、response长度[2,3,4,4]，配置cap为8／6；基线按补齐宽度5／4计算，prompt、response及non-aborted触顶比例均0.5，候选显式传cap后均0。这修正了V1批次形状引入的假阳性，但并没有观察停止原因；其相等比较仍不计超出cap的长度。

后端还有逐请求的上下文预算。若配置response cap为8，实际上下文剩余预算让三条响应在6／5／4停止，单一配置阈值不会命中。[#7906指标函数](https://github.com/verl-project/verl/blob/2b5536028b6a17aa9d2ce5adf893093629f58b51/verl/trainer/ppo/metric_utils.py)改为逐行优先使用`response_truncated`，None才回退到宽度比较。因而它与#7821解决不同层：前者补停止证据，后者修正长度估计的阈值；单用#7906时，未知行与prompt指标仍保留原宽度语义。

| 原指标函数的CPU输入 | #7906候选结果 | 解释 |
|---|---|---|
| 长度[6,5,4]，三个True | response比例1 | 不必等到宽度8才计入 |
| 单行长度8、宽度8，False／None／True | 分别0／1／1 | 已知正常停止覆盖相等估计，未知不是False |
| 长度[8,2,4]，flags=[False,True,None]，宽度8 | 约1/3 | 按全部三行计分母，不是只看两条已知行的1/2 |
| 长度[6,5,0]，flags=[True,False,False] | 全部1/3，non-aborted为1/2 | non-aborted分母排除零长度行 |

这里的`response/aborted_ratio`原实现按响应长度为0识别，不能把它当作所有后端抢占／中断次数。全零响应的non-aborted比例为NaN，也不应在看板解释成0。显式正常停止恰好落在cap时，从旧估计的1改成0，是指标语义变更，不是生成质量突然改善。

标志还须走完整传输路径：#7906给[TokenOutput](https://github.com/verl-project/verl/blob/2b5536028b6a17aa9d2ce5adf893093629f58b51/verl/workers/rollout/replica.py)新增可选bool，各后端按finish_reason赋值；agent经extra_fields传入V0／V1指标，V1还按非padding行选择。源码中resume采用最后一次返回标志，并处理累计预算耗尽；工具loop保留任一已知截断，无已知截断但存在未知turn时保留None，工具观察也可能耗尽agent预算。因此该字段不是只记录最后一个模型请求的停止原因，更不是丢弃样本或训练loss mask。这里只核源码与作者测试的替身边界，没有执行这些loop或真实后端。

Python3.12.13／torch2.14.1／NumPy2.5.4／TensorDict0.15.2，复用现有环境。AST原样加载固定main `3c6d9b43b67c74ec343d77874cd8fa6e841523da`的DataProto类与各版两个完整指标函数，真实CPU TensorDict运算，原测试模块、参数和断言不改，直接调用测试体。#7821原5项候选全过，基线5项因不支持新kwargs出现TypeError，上述0.5反例另以旧签名执行；#7906原13参数项候选全过，基线7过6次AssertionError，两次一致。两基线指标函数AST与固定main相同。原DataProto也为AST隔离，不是完整verl导入；没有pytest原生收集、Ray／TQ、tokenizer、模型、后端推理或端到端训练，作者41／101项套件不计本机结果。七份完整源码compile及[命令／hash](../../diary/2026-10-25.md)可追溯，真机缺口只补[原M5条件入口](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不应用候选、改专项或新增基准队列。任何社区动作及已有PR-head推送仍逐项待用户明确同意。

## 6. LoRA Integration in GRPO

```
ray_trainer.py:358-360:
  self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path")

→ LoRA: ref和actor共享同一GPU!
→ ref_in_actor=True → self.ref_policy_wg = self.actor_rollout_wg (同一WorkerGroup)
→ compute_ref_log_prob时: no_lora_adapter=True → disable LoRA → 用base model weights
→ compute_actor_log_prob时: LoRA enabled → 用LoRA-adapted weights
→ 无需额外GPU → RTX 4090可行!

→ 无LoRA: ref和actor分开 → 需2个GPU → RTX 4090不可行!
→ COLOCATED: actor+ref在同一PG → 可2GPU但内存紧张
→ STANDALONE: actor+ref分离 → 需要≥3个GPU!
```

## 6b. Reward Integration in GRPO

```
RewardLoopManager (reward_loop.py:273-):
  → 管理 reward computation: RewardLoopWorker(Ray actors) + RewardModelManager

两种reward来源:
1. custom_reward_function (rule-based) → 大多数GRPO用这个!
   → 数学题: correctness check → 0/1分数
   → 代码题: execution result → pass/fail
   → 不需要GPU → 只需要CPU → 轻量!

2. reward_model (disrm) → vLLM/SGLang HTTP API → 需要额外GPU!
   → vLLM: POST classify → probs[-1] → RM score
   → SGLang: POST v1/embeddings → embedding[-1] → RM score
   → 需要单独GPU → RTX 4090空间不够!

GRPO reward路径 (ray_trainer.py:1518-1603):
  → self.use_rm = need_reward_model(config) → config.reward.reward_model.enable
  → GRPO typically: use_rm=False → rule-based → 不需要RM GPU!

  → if use_rm=True:
    → sleep rollout GPU → wake RM GPU → compute → wake rollout back
    → RewardLoopWorker → compute_rm_score → assemble_rm_scores → rm_scores tensor

  → GRPO: token_level_rewards = token_level_scores (无KL-in-reward!)
  → PPO: token_level_rewards = token_level_scores - kl_penalty_in_reward
```

**Reward数据形状**:
```
Outcome reward (GRPO典型):
  → rm_scores: [bsz * rollout_n, response_length]
  → 只有最后一个response token非零 → outcome-only!
  → token_level_rewards.sum(dim=-1) = scalar per response → group-relative!

Token-level reward (PPO):
  → 每个token都有reward → 需要RM forward per token → 更昂贵!

Rule-based reward:
  → compute_score函数 → Python函数 → 不需要GPU → 极快!
  → 但: 只有outcome → 无法区分好的中间步骤 → 可能不够精细
```

**RTX 4090 GRPO reward选择**:
```
RTX 4090: 只能用rule-based reward → 不需要额外GPU!
  → math: correctness check → 简单但有效
  → code: execution sandbox → 需要sandbox server → 但不在GPU上

→ 不能用reward model → RTX 4090空间已经占满actor+ref
→ LoRA mode: actor=14GB(base)+1GB(LoRA) → 24GB只剩9GB → 不够7B RM!
```

## 7. AdvantageEstimator完整注册表

```
core_algos.py AdvantageEstimator enum:

GRPO            → loop-based group-relative
GRPO_VECTORIZED → vectorized group-relative (推荐!)
GRPO_PASSK      → only best response (arxiv 2503.19595)
GDPO            → decoupled per-dimension normalization (arxiv 2601.05242)
REINFORCE_PLUS_PLUS         → REINFORCE++ baseline
REINFORCE_PLUS_PLUS_BASELINE → outcome-only REINFORCE++
REMAX                        → REMAX with greedy baseline
RLOO                         → Leave-One-Out baseline
PWIL                         → Wasserstein distance
IS                           → Importance Sampling correction
```

## 8. RTX 4090 GRPO可行配置

```
RTX 4090 (24GB) GRPO配置:

1. HYBRID mode → actor+ref+rollout同一进程 → 1 GPU
2. LoRA rank=16-64 → ref_in_actor=True → 无需额外ref GPU
3. naive weight sync → Python generator zero-copy → 不需要NCCL/IPC
4. GRPO_VECTORIZED → 最快advantage计算
5. 7B BF16 LoRA: ~14GB(base) + ~1GB(LoRA) = 15GB → 24GB可行!
6. rollout_n=4-8 → group足够大 → std有意义 → advantage有效

→ RTX 4090唯一可行RL方案: GRPO + LoRA + HYBRID + naive
→ PPO: 需要3个模型 → 3×14GB=42GB → RTX 4090不可行!
→ rLLM TinkerBackend: 也可行 → 但verl生态更成熟
```

## 9. 关键设计洞察

```
1. GRPO不是新loss → 是新advantage → loss完全不变!
   → PPO和GRPO用同一个clipped_surrogate → 只是advantage来源不同
   → 这意味着: PPO→GRPO迁移 → 只需改adv_estimator → 其他代码不变!

2. Dr.GRPO(不除std) → 防止梯度消失 → 当σ_g接近0时
   → 小group(n=2-4): σ可能很小 → 除σ → advantage极端 → 梯度不稳定
   → Dr.GRPO: a_i = r_i - μ_g → 更稳定 → 但可能不够normalized
   → 推荐: n≥8时用GRPO(除std), n≤4时用Dr.GRPO(不除std)

3. GRPO_PASSK → 极致compute节省 → 只有best response训练
   → 但需要n≥2 → 至少2个response/group
   → 适合: 大batch+小GPU → 只训最有价值的样本

4. GDPO → 多维度reward的关键改进 → 防止dominant dimension
   → 数学推理reward: 正确率>格式 → GRPO可能只优化格式
   → GDPO: 先per-dimension normalize → 平衡优化所有维度
   → 生产级reward需要多维度 → GDPO比GRPO更好!

5. ref_in_actor = LoRA的杀手级优化 → 省1个GPU!
   → 无LoRA: ref需要独立GPU → 2GPU起步 → RTX 4090完全不行
   → LoRA: ref和actor共享 → disable_adapter()切换 → 只1GPU
   → 这不是hack → 是verl正式特性 → ray_trainer.py明确支持!

6. interleave=True → group内response连续 → advantage计算正确
   → gen_batch.repeat(n, interleave=True)
   → 排列: [p1_r1, p1_r2, ..., p1_rn, p2_r1, ...]
   → index = prompt_id → 自然分组 → 不需要额外sort

7. 10种policy loss → 不只clip!
   → vanilla(clip) / dppo_tv / dppo_kl / gspo / sapo / gpg / clip_cov / kl_cov / geo_mean / cispo / bypass_mode
   → GRPO推荐: vanilla(标准clip) → 最简单最稳定
   → 高级: cispo/geo_mean → 更复杂的clip策略 → 可能更好但需调参
```

---

Sources:
- verl/verl/trainer/ppo/core_algos.py — GRPO advantage (3 variants) + GDPO + 10 policy losses
- verl/verl/trainer/ppo/ray_trainer.py — rollout_n interleave + ref_in_actor + advantage routing
- verl/verl/workers/utils/losses.py — ppo_loss + sft_loss implementation
- verl/verl/trainer/ppo/reward.py — extract_reward logic
- notebook/frameworks/verl-worker-lifecycle-ray-weight-sync-reading.md — WorkerGroup/RolloutMode
- notebook/frameworks/verl-multi-turn-agent-loop-reading.md — ToolAgentLoop
