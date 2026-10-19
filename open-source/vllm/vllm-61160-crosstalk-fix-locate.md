# Candidate: vLLM #61160 跨请求污染的修复定位（回答 reporter 提问）

> Issue: [vllm-project/vllm#61160](https://github.com/vllm-project/vllm/issues/61160)
> [Bug]: Cross-request context contamination via shared-prefix KV（NIXL P/D 分离 + DeepSeek-V4.1-Flash + dspark 投机解码）
> 创建 2026-10-12；6 评论；查重 2026-10-19：**0 个引用 PR**（reporter 未认领；维护者 ZJY0516 确认 v0.31.0 复现、latest main 不再复现，reporter 请求指出修复 PR）
> 类别：调查/评论型（定位修复 = 回答 issue 中的明确问题）
> 状态：**analysis-done**（修复定位完成，评论草稿待授权发布）

## 0. 为什么值得做

| 维度 | 评估 |
|---|---|
| 影响 | 4 — 静默跨请求污染（48 请求 46 串号）是最高危 bug 类；reporter 明确请求指出修复 PR（部署升级验证需要） |
| 成本 | 4 — 纯调查（GitHub API 搜索 + PR 对照），无 GPU 需求 |
| 接受度 | 3 — 评论型贡献；若定位正确直接解决 issue 未决问题 |
| 契合度 | 5 — **前缀缓存 + 域碰撞家族是我们的深耕区**（#44701 分析、prefix-cache 阅读笔记、P/D 分离笔记） |
| 分 | 16 |

## 1. 症状与现状

- NIXL prefill/decode 分离部署 + 共享前缀（多轮 agent 会话同 system prompt）→ 响应间歇性含其他请求内容（canary 串号 46/48）；`--no-enable-prefix-caching` 可消除。
- 维护者 ZJY0516：v0.31.0 确认复现，**latest main（10-12/10-14 测试）不再复现** → 修复已在 main，但无人指出具体 PR。
- 关联 #61229（lease 过期变体，**仍开放**——家族未完全关闭）。

## 2. 修复定位（2026-10-19 调查，三层修复全在 10 月窗口）

reporter build = main@db9527a46（10-02），早于以下全部修复：

| PR | 合并 | 机制 | 与 #61160 的对应 |
|---|---|---|---|
| **#51899** Tag prefix-cache extra keys by source | 09-30 | LoRA 名/cache_salt/多模态 id 混入无域标签 extra_keys → 碰撞 | 域碰撞家族（我们 6 月 #44701 分析的修复最终形态）。**2026-10-19 验证：db9527a46 已含此修复（`("lora",name,path)` 与 `("cache_salt",...)` 均在场）→ 排除**：构建已含仍复现 46/48 串号 |
| **#59103** Drop stale block hashes when a session truncates tokens（fixes #49377） | 10-06 | 流式会话截断时 `_all_token_ids` 截到 num_computed_tokens，丢弃的最后一个采样 token 若凑成一个 hash block，该 hash 继续标识替换后的序列 → 后续 commit 到错误 key | **主修复（排除法 + 机制匹配）**：db9527a46 不含修复行（已验证）而 ZJY0516 10-12 测试时已含——时间窗完全吻合；多轮 agent 会话（截断+续接）+ 共享前缀 = repro 场景 |
| **#61174** Clear stale shared block-table rows（Spec Decode/MTP） | 10-17 | 目标 batch 缩小后较大 MTP 解码图读 stale 共享 block-table 行 → "**padded work 访问另一请求的 cache 或 state**" | 与污染描述逐字对应（"another request's cache"）；repro 用 dspark（MTP 族）投机解码；但 10-17 才合，晚于 ZJY0516 10-12 复测 → **排除为主修复**，保留为投机解码路径补充 |
| #60533 Fix hybrid Mamba prefix cache（从 #59826 提取） | 10-08 | Mamba prompt checkpoint 边界（P-1 cap / EAGLE block drop 双重应用） | **次要候选**：若 DSV4.1-Flash 含 Mamba 层可能相关；机制（checkpoint 边界）与"会话截断+续接"场景关联弱于 #59103；评论中可提及或省略 |

**结论（2026-10-19 强化版）**：主修复 = **#59103**。推理链 = 排除法（#51899 已证在报障者构建中仍复现 → 排除；#61174 晚于维护者复测窗口 → 排除）+ 时间窗吻合（10-02 构建不含 / 10-06 合入 / 10-12 复测已修复）+ 机制匹配（会话截断 stale hash 对应多轮续接场景）。次要候选 #60533（hybrid Mamba 边界）若模型含 Mamba 层亦可能相关。建议 reporter 用当前 main（≥10-17）重跑 canary 验证；#61229（lease 变体）为独立路径。

## 3. 评论草稿（发布需用户授权）

> 定位修复的评论（回复 reporter 的 "could you point us to the PR/commit that fixed it"）：
> - 指出 #59103（10-06 合并，fixes #49377）：流式会话截断时 stale block hash 未丢弃 → 替换后的序列在旧 hash 下被复用 → 跨请求共享前缀场景的污染；与你们的多轮 agent 会话（截断+续接）场景精确匹配。
> - 补充 #61174（10-17 合并）：MTP/spec-decode 路径 target batch 缩小后 stale 共享 block-table 行导致 padded work 访问另一请求 cache——与你们 dspark 投机解码配置对应。
> - 补充 #51899（9-30）：extra_keys 域打标（salt/LoRA 碰撞家族）。
> - 建议：用 ≥10-17 的 main 重跑 canary；#61229（lease 变体）仍在开放，如复现建议核对是否同根因。

## 4. 待办

- [ ] 用户授权后发布评论（或先核实 #59103 的 diff 细节强化证据）
- [ ] 可选：检查 #61229 是否可归并到同一修复家族（另一潜在评论价值）

## 5. 家族分离结论（2026-10-19 补充，#61229 核查）

**#61229（lease 侧）与 #61160（hash/block-table 侧）是两条独立路径**：
- #61229：NixlConnector producer 在 `kv_lease_duration` 过期后无条件释放 KV 块，consumer 读已复用块 → 静默错误。讨论已收敛（维护者 robertgshaw2-redhat 10-16 认可 "positive fix"），issue 仍 open。
- 其修复系列与 #61160 的 3 个 PR **不重叠**：#61549（sawsa307，"Fixes #61229"，**closed unmerged**——经 KV load failure path 拒绝过期读）、#62065（open，核对 producer lease slot）、#61488/#61578/#61580（KV 校验和系列，open——系统化验证 P/D 传输）。
- 对 #61160 评论的意义：可顺带澄清两个 crosstalk 形状根因不同，#61229 已有独立跟踪与进行中修复，避免维护者误认为重复。
