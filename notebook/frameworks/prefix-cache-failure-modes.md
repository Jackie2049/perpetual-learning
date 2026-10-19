# 前缀缓存失效模式分类（Prefix-Cache Failure-Mode Taxonomy）

> 2026-10-19 沉淀：基于 vLLM #61160 调查 + 6 月 #44701 分析 + FP8 KV 家族，把前缀缓存的正确性失效系统化。
> 关联：[vLLM V1 前缀缓存 KV 管理](vllm-v1-prefix-caching-kv-management.md) · [前缀缓存基础](fundamentals/prefix-caching.md) · [vLLM #61160 修复定位](open-source/vllm/vllm-61160-crosstalk-fix-locate.md)

## 为什么需要这张表

前缀缓存把"内容指纹"（block hash）映射到共享的物理 KV 块。**任何让指纹与内容脱节、或让块归属错乱的因素，都会造成静默错误**（最严重：跨请求内容污染/串号）。失效模式可按"指纹错误"与"块归属错误"两大类组织。

## 分类表

| # | 模式 | 机制 | 失效点 | 表现 | 修复 | 状态 |
|---|---|---|---|---|---|---|
| 1 | **域碰撞**（domain collision） | LoRA 名/cache_salt/多模态 id 混入**无域标签** extra_keys 元组，不同请求 hash 相等 | 指纹计算 | 用别的权重算出的 KV 服务本请求（首块碰撞，链式传播） | [#51899](https://github.com/vllm-project/vllm/pull/51899) 域打标（9-30 合） | ✅ 已修（我们 6 月 #44701 分析预判） |
| 2 | **截断 stale hash**（session truncation） | `_update_request_as_session` 截断 token 列表但**不截 block_hashes** → 被丢弃尾巴的 hash 残留 → 续接序列在旧 hash 下提交 | 指纹链（会话续接） | 多轮会话续接后缓存复用错误；共享前缀 fleet 中指纹链相同 → 块共享 → **跨请求串号** | [#59103](https://github.com/vllm-project/vllm/pull/59103) 同步截断（10-06 合） | ✅ 已修（#61160 主修复候选） |
| 3 | **MTP/spec-decode stale 行** | target batch 缩小后大 MTP 解码图读 stale 共享 block-table 行 → padded work 访问**另一请求的 cache** | 块表归属（投机解码） | 投机解码配置下的跨请求污染 | [#61174](https://github.com/vllm-project/vllm/pull/61174) 清除 stale 行（10-17 合） | ✅ 已修（#61160 补充候选） |
| 4 | **lease 过期释放**（lease expiry） | NixlConnector producer 在 `kv_lease_duration` 过期后**无条件释放** KV 块；consumer 无内容/身份绑定地读已复用块 | 块归属（P/D 分离） | 解码读另一请求内容，静默错 | [#61549](https://github.com/vllm-project/vllm/pull/61549)（未合）/ [#62065](https://github.com/vllm-project/vllm/pull/62065)（open）/ KV 校验和系列 #61488/#61578/#61580（open） | ⚠️ 进行中（#61229） |
| 5 | **FP8/量化 KV + 前缀**（CUDA） | compressed-tensors FP8 覆盖 KV dtype → FlashInfer FP8 kernel 仅 SM90+ → SM89 崩溃 | 后端选择 | SM89 上 FP8 KV 前缀缓存崩溃 | [#45038](https://github.com/vllm-project/vllm/pull/45038)（未合自关）/ #44879 仍 open | ⚠️ 未决 |
| 6 | **FP8 KV + 前缀**（ROCm） | GLM-5.2-FP8 + FP8 KV + 重复相同 prompt → 空内容/乱码 | 后端/平台 | ROCm/AITER（MI300X） | 无 | ❌ 排除（平台外 + 无复现） |

## 两条主机制链（#61160 家族，详细）

### 指纹错误链（模式 1、2）

```
指纹 = 缓存查找的唯一依据
  ↓ 脱节
同一指纹代表不同内容（域碰撞）或同一内容算出不同指纹（stale hash）
  ↓ 后果
请求 A 的块被请求 B 命中 → B 读 A 的 KV → 输出串号
```

共享前缀 fleet 放大：相同 system prompt + 相同截断点 → 相同（错误）指纹链 → 缓存视为同一前缀 → **块共享 → 内容漂移**。报障者 46/48 串号即此链（禁用前缀缓存后消失 = 最强证据）。

### 块归属错误链（模式 3、4）

```
块释放/重指派与引用不同步
  ↓
引用方读已复用块（lease 过期 / spec-decode stale 行）
  ↓
解码读另一请求内容，HTTP 200 无警告
```

## 诊断清单（遇到"响应串号/内容错乱"时）

1. **隔离**：`--no-enable-prefix-caching` 复测——污染消失 ⇒ 走前缀缓存路径（#61160 报障者实测有效）
2. **指纹侧**：检查 extra_keys 是否域打标（模式 1）；检查会话截断路径 block_hashes 是否同步（模式 2）
3. **块归属侧**：P/D 分离 + KV connector → 查 lease 逻辑与块表状态（模式 3、4）
4. **量化侧**：FP8 KV + SM<90 → 查后端选择（模式 5）
5. **跨请求 canary**：注入每请求唯一标记实测污染率（量化严重度）

## 开放项

- #61229（lease 变体）未决：讨论已收敛（维护者认可 positive fix），修复系列进行中——跟进价值中等
- #44879（SM89 FP8 KV）未决：0 活跃修复
- 模式 2/3 是否还有未覆盖路径（如更复杂的会话操作）——vLLM 持续演进中

## 经验

- **指纹与块归属是两条独立失效轴**：修一边不代表另一边安全（#61160 3 个修复各自覆盖一条路径）
- **现象级 issue 往往对应多机制**：定位需按"时间窗 + 机制匹配"排除，评论要保留验证步骤
- **禁用缓存是最快的归属判定**，先做它再往下查
