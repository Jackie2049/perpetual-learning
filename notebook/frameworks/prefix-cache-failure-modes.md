# 前缀缓存失效模式分类（Prefix-Cache Failure-Mode Taxonomy）

> 2026-10-19 沉淀：基于 vLLM #61160 调查 + 6 月 #44701 分析 + FP8 KV 家族，把前缀缓存的正确性失效系统化。
> 关联：[vLLM V1 前缀缓存 KV 管理](../fundamentals/vllm-v1-prefix-caching-kv-management.md) · [前缀缓存基础](../fundamentals/prefix-caching.md) · [vLLM #61160 修复定位](../../open-source/vllm/vllm-61160-crosstalk-fix-locate.md)
> 下表旧项状态为2026-10-19记录；2026-10-23补[权重版本与外部缓存](#权重版本与外部缓存)，不将旧合并/未决标记当作本轮状态核验。

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
| 7 | **权重代次与外部缓存** | 权重已变而查找键未变，且local reset未触达外部层；或全局清空共享store影响其他replica | 查找身份与失效范围 | 相同token可能命中旧策略KV；不要求hash函数碰撞 | [verl #8323](https://github.com/verl-project/verl/pull/8323)候选按run namespace/step加盐 | 2026-10-23读取open；本机仅源码/语法核对，非设备验收 |

## 权重版本与外部缓存

### 命名隔离不等于删除或权重认证

固定阅读[verl #8323](https://github.com/verl-project/verl/pull/8323)，head `4e93e4761ddde2ea5bf9d3032b86b6c3844126fd`、目标base `16ec093cc4a235c603c29aed1f2c4e9f551fab48`，共同祖先 `f2923cbf154731918b1e36db8c567433f349fede`，ahead2/behind3；实际PR为20文件869增20删。候选默认仍`external_kv_cache_invalidation=reset`；可选salt改变外部缓存查找身份，不新增跨版本KV复用或自动沿用在途任务旧版本的策略。

[prefix_cache_salt](https://github.com/verl-project/verl/blob/4e93e4761ddde2ea5bf9d3032b86b6c3844126fd/verl/workers/rollout/replica.py)生成`namespace:global_steps`。namespace缺省取模块导入时的进程级UUID，同进程所造replicas共享；并非每次构造新UUID或自动识别同进程第二个逻辑run，已有非None配置也会保留。[vLLM0.29文档](https://docs.vllm.ai/en/v0.29.0/design/prefix_caching/)说明salt直接进入首块hash，后续块通过parent hash承接；不能说每块都直接插入salt字段。

只要版本身份真实且外部store沿用带salt的block key，新版本就不会查到旧版本条目。**旧条目仍存在，回收依store自己的eviction/容量策略；不匹配不等于已删除或内存已释放。** 这也不是权重内容校验、所有rank安装确认、访问控制或多租户安全认证。若不同权重复用同一namespace/step，helper不会检测内容差异。

[checkpoint manager](https://github.com/verl-project/verl/blob/4e93e4761ddde2ea5bf9d3032b86b6c3844126fd/verl/checkpoint_engine/base.py)在非naive路径、abort之前拒绝salt模式缺失global_steps；没有比较上次step是否重复/递增。naive提前返回走原worker流程，注释以不传step的调用只恢复未变权重为前提；helper本身也没有拒绝None，可以构成`namespace:None`。这些是调用方契约边界，不能将一个ValueError guard称为全路径“每次权重变化都有唯一版本”的证明。

### 请求时序与三后端接线

盐必须对应请求实际执行时的权重；请求若在暂停期间排队，提前读取版本会把新权重算出的KV写入旧namespace。候选将计算位置放到各自服务端准入门之后：

| 后端 | 放盐位置 | 失效动作与限制 |
|---|---|---|
| vLLM | `_park_until_admitted`之后加入prompt | salt下仍清本地prefix/MM/encoder cache，但reset_connector=False保留外层；入场barrier超时改为抛错，不带着迟到请求继续更新 |
| SGLang | 新增verl侧Event门，wait后填写GenerateReqInput | abort前关门，pause失败重开；continue成功后开门。原local flush路径不因salt消失；依赖native请求进入pause检查的时序 |
| TRT-LLM | `_park_until_allowed`之后，非None才传cache_salt | 保持reset模式兼容不接受该参数的旧版本；verl本候选没有为它配置外部store，不能外推外部层清理/命中 |

vLLM的abort_only旁路仍不关闭门或清缓存，它不是“安全开始权重变化”的同义词。失败门、barrier或暂停API也没有提供参数回滚/集群自动恢复。本轮核对[SGLang v0.5.20源码](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/managers/tokenizer_manager.py#L848)：generate_request早期在is_pause_cond中检查暂停条件；未执行其并发路径，候选注释中的调度假设不当作版本无关的原子性保证。

[配置guard](https://github.com/verl-project/verl/blob/4e93e4761ddde2ea5bf9d3032b86b6c3844126fd/verl/workers/config/rollout.py)拒绝salt+PD分离；vLLM按连接器名称允许MooncakeStoreConnector、OffloadingConnector、LMCacheMPConnector、HF3FSKVConnector，递归检查MultiConnector，并处理两种kv_transfer_config拼写/JSON字符串。它检查配置名称，不执行连接器实现或认证任意版本都保留block hash。LMCacheConnectorV1等不在此名单，不能因名字相近或同为LMCache照搬支持结论。

### 已有测试究竟覆盖什么

本轮完整读候选diff、两份CPU测试和三份GPU测试，Python3.12.13对11个变更Python模块仅AST/compile，通过不等于原生导入或测试运行。复查使用固定对象的`git diff base...head`和`git show head:path`，语法检查不exec模块。没有执行原CPU套件：本机缺Ray/vLLM/SGLang/TRT-LLM/OmegaConf等原生依赖，服务端用例又包含FakeEngine/FakeTokenizerManager及绕构造；未新造引擎替身来代跑。也未运行模型、connector、GPU/NPU或训练。完整patch SHA256为`1820aa50290774090256cd1b9b169113cd2918f7a2d76d64ad03e040e0e18197`；replica.py为`ec938575ca5202bbc1b443752bd8568289450a07558e10434b0f835b9ca146c0`。

作者报告1×RTX PRO4000/Blackwell、Qwen2.5-0.5B-Instruct，vLLM0.29.0、SGLang0.5.20、TRT-LLM1.3.0rc15。固定三份GPU测试均从standalone模型出发，改变global_steps并观察缓存命中，**没有写入另一组模型权重、调用实际checkpoint传输或执行optimizer更新**：

- [vLLM用例](https://github.com/verl-project/verl/blob/4e93e4761ddde2ea5bf9d3032b86b6c3844126fd/tests/workers/rollout/rollout_vllm/test_vllm_prefix_cache_versioning.py)使用native CPU-offload connector，abort→改标签→resume；同版本再清local后salt仍命中，而reset清外层后不命中。这验证缓存域/外层保留行为，不验证异构权重同步后输出正确。
- SGLang/TRT用例在不flush时改标签，salt miss而reset hit，再核salt的abort/resume；用真实cache计数，但不据标签变化称已更新权重。TRT用例没有外部store。
- SGLang HiCache file层“400-token prompt在reset下复用384、salt下0”是作者另行one-off脚本报告，非上述仓内默认SGLang GPU用例，也非本项目复现。vLLM无store对照同样来自作者正文，不能添为仓内测试断言。

因此当前证据支持“候选隔离键的接线与特定backend缓存实验报告”，不支持全部连接器/模型/并发窗口下的端到端参数一致性或性能收益。设备缺口只补[原M5一致性参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，待独立选题；不自动应用、重开#8250因果诊断或在生产共享store上注入故障。

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
