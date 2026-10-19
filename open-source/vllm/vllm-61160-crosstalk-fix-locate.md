# Candidate: vLLM #61160 跨请求污染的修复定位（回答 reporter 提问）

> Issue: [vllm-project/vllm#61160](https://github.com/vllm-project/vllm/issues/61160)
> [Bug]: Cross-request context contamination via shared-prefix KV（NIXL P/D 分离 + DeepSeek-V4.1-Flash + dspark 投机解码）
> 创建 2026-10-12；6 评论；查重 2026-10-19：**0 个引用 PR**（reporter 未认领；维护者 ZJY0516 确认 v0.31.0 复现、latest main 不再复现，reporter 请求指出修复 PR）
> 类别：调查/评论型（定位修复 = 回答 issue 中的明确问题）
> 状态：**analysis-done**（修复定位完成，评论草稿待授权发布）

## 0. 为什么值得做

| 维度 | 评估 |
|---|---|
| I | 4 — 静默跨请求污染（48 请求 46 串号）是最高危 bug 类；reporter 明确请求指出修复 PR（部署升级验证需要） |
| C | 4 — 纯调查（GitHub API 搜索 + PR 对照），无 GPU 需求 |
| A | 3 — 评论型贡献；若定位正确直接解决 issue 未决问题 |
| F | 5 — **前缀缓存 + 域碰撞家族是我们的深耕区**（#44701 分析、prefix-cache 阅读笔记、P/D 分离笔记） |
| 分 | 16 |

## 1. 症状与现状

- NIXL prefill/decode 分离部署 + 共享前缀（多轮 agent 会话同 system prompt）→ 响应间歇性含其他请求内容（canary 串号 46/48）；`--no-enable-prefix-caching` 可消除。
- 维护者 ZJY0516：v0.31.0 确认复现，**latest main（10-12/10-14 测试）不再复现** → 修复已在 main，但无人指出具体 PR。
- 关联 #61229（lease 过期变体，**仍开放**——家族未完全关闭）。

## 2. 修复定位（2026-10-19 调查，三层修复全在 10 月窗口）

reporter build = main@db9527a46（10-02），早于以下全部修复：

| PR | 合并 | 机制 | 与 #61160 的对应 |
|---|---|---|---|
| **#51899** Tag prefix-cache extra keys by source | 09-30 | LoRA 名/cache_salt/多模态 id 混入无域标签 extra_keys → 碰撞 | 域碰撞家族（我们 6 月 #44701 分析的修复最终形态）；边界在 build 前后，需确认 db9527a46 是否含 |
| **#59103** Drop stale block hashes when a session truncates tokens（fixes #49377） | 10-06 | 流式会话截断时 `_all_token_ids` 截到 num_computed_tokens，丢弃的最后一个采样 token 若凑成一个 hash block，该 hash 继续标识替换后的序列 → 后续 commit 到错误 key | **多轮 agent 会话（截断+续接）+ 共享前缀 = repro 场景**；ZJY0516 10-12 复测大概率命中此修复 |
| **#61174** Clear stale shared block-table rows（Spec Decode/MTP） | 10-17 | 目标 batch 缩小后较大 MTP 解码图读 stale 共享 block-table 行 → "**padded work 访问另一请求的 cache 或 state**" | 与污染描述逐字对应（"another request's cache"）；repro 用 dspark（MTP 族）投机解码；10-17 才合，ZJY0516 测试时未含 |

**结论**：主修复很可能是 **#59103**（10-06，与多轮会话截断场景精确匹配）；#61174（10-17）覆盖投机解码/MTP 路径的同类污染；#51899（9-30）是早先的域碰撞修复。三者均应在当前 main 中。建议 reporter 用当前 main（≥10-17）重跑 canary 验证，并核对 #61229（lease 变体）是否同根因。

## 3. 评论草稿（发布需用户授权）

> 定位修复的评论（回复 reporter 的 "could you point us to the PR/commit that fixed it"）：
> - 指出 #59103（10-06 合并，fixes #49377）：流式会话截断时 stale block hash 未丢弃 → 替换后的序列在旧 hash 下被复用 → 跨请求共享前缀场景的污染；与你们的多轮 agent 会话（截断+续接）场景精确匹配。
> - 补充 #61174（10-17 合并）：MTP/spec-decode 路径 target batch 缩小后 stale 共享 block-table 行导致 padded work 访问另一请求 cache——与你们 dspark 投机解码配置对应。
> - 补充 #51899（9-30）：extra_keys 域打标（salt/LoRA 碰撞家族）。
> - 建议：用 ≥10-17 的 main 重跑 canary；#61229（lease 变体）仍在开放，如复现建议核对是否同根因。

## 4. 待办

- [ ] 用户授权后发布评论（或先核实 #59103 的 diff 细节强化证据）
- [ ] 可选：检查 #61229 是否可归并到同一修复家族（另一潜在评论价值）
