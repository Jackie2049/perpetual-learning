# vLLM pyflag 第二批：12 条「静默失效」里只有 1 条真缺陷，外加一条新形态（2026-11-05）

## 一句话结论

vLLM main `da0b6ac2`（2026-11-05）上 pyflag 报 **263 个布尔开关、12 条 `silently_inert`**。逐条绑事实后：
**真缺陷 1 条**（`enable_kv_events`——文档写明的逐层事件开关，全库没有任何调用点传它，包括 vLLM 自己的工厂），
**另一条是"模板硬编码"**（`no_thinking`——Jina re-ranker 的输出模板里 `</think` 恒被补上，没有入口能关），
其余 **10 条**分四类结构原因，每条都有可机读的判据。
对照基准 `use_page_cache` 也在这 12 条里：它被**误报**，真生产者是 `self.use_page_cache` 属性转发。

## 为什么做

TM-013／TM-014 在 SGLang 上得出"386 个开关 / 34 条失效 / 真缺陷 2 条"，并写进 pyflag readme：
**再扫应换新框架而非重扫**。vLLM 是常规扫描池里最大、最活跃的对象，一直没扫过。
本轮按扩池要求只纳入这一个对象，先做短评估，再决定要不要深入。

## 读数

扫描命令与耗时：

```bash
git clone --depth 1 https://github.com/vllm-project/vllm /tmp/vllm-scan-20261105   # 5976 个 .py
python3 tools/pyflag/scan_unwired_flags.py /tmp/vllm-scan-20261105 > /tmp/vllm-pyflag-out.txt
# 2 分 42 秒（SGLang 8000+ 文件约 10.5 分钟）
```

| 项 | 数 |
|---|---|
| 发现的布尔开关 | **263** |
| `silently_inert` | **12**（`never` 7 / `always` 5） |
| 逐条取证后真缺陷 | **1**（`enable_kv_events`） |
| 模板硬编码（另一类） | **1**（`no_thinking`） |
| 结构原因排除 | 10（见下表） |

### 12 条逐条判定

| # | 开关 | pyflag `mode` | 判定 | 依据（`verify_findings.py` 的 check 名） |
|---|---|---|---|---|
| F1 | `enable_kv_events` | never | **真缺陷** | 见下节 |
| F2 | `no_thinking` | always | **模板硬编码** | `io_processor.py:853` 声明、`:884` 分支、`:885` 拼 `</think`；4 个调用点（`:954/960/1002/1008`）**都不传** |
| F3 | `use_page_cache` | never | **误报：属性转发** | `simple_cpu_offload_connector.py:153` 读配置 → `:232` `use_page_cache=use_page_cache` → `worker.py:49` 存 self → `disk_backend.py:104` 分支 |
| F4 | `enable_trace_replay` | never | 误报：config 字段属性 | `config/model.py:313` 字段 → `model_runner.py:507` 读 `self.model_config.enable_trace_replay` → `sample/sampler.py:83` 用 |
| F5 | `use_mix`／`use_abs_pos`／`use_sliding_window`／`use_grads` | always | 模型实现内部默认 | 四个都是构造期一次性读（`hyperconnection.py:167/185`、`deepencoder.py:246/282`、`flint/configs.py:40/73`、`cohere_asr.py:54/162`） |
| F6 | `use_fused_tilelang`／`use_pre_gemm_splits` | never | 同文件小表驱动 | `tilelang_kernels.py` 里生产者是 4 行具名实参 + 局部变量，不是前端配置 |
| F7 | `use_heuristic` | always | 测试专用旋钮 | `triton_scaled_mm.py:187/216`；库内无调用点，只有 `tests/kernels/quantization/test_triton_scaled_mm.py:127` 参数化两档 |
| F8 | `enable_telemetry` | never | 生产者是字典键 | `kvcr/manager.py:289` `extra_config.get("enable_telemetry", False)` → `:426` 再转发 |

12 条 = F1(1) + F2(1) + F3(1) + F4(1) + F5(4) + F6(2) + F7(1) + F8(1)。

## 真缺陷：`enable_kv_events` 没有生产者，**vLLM 自己也不传**

**它是什么。** `secondary_tiers` 是多级 KV 卸载（DRAM → 文件系统 / S3）的配置段。PR #47923（2026-07-10 合入，
收 #46544）给 `fs` 和 `obj` 两个 tier 各加了一个 `enable_kv_events` 形参，语义是"这个 tier 存成功的块，
要发 `BlockStored` KV 事件"。PR 描述的原话是：

> Per-tier opt-in: a new `enable_kv_events: bool = False` key in the tier's `secondary_tiers` config entry
> (passed through `SecondaryTierFactory` kwargs).

**证据链。**

1. 两个 tier 都声明了形参并真的按它分支：`fs/manager.py:117/143`、`obj/manager.py:114/142`。
2. 唯一构造路径是 `spec.py:433` → `factory.py:96`，把 **tier 配置字典原样 `**config` 展开**。所以"文档说的
   透传"在**字面上确实成立**——只要配置里真写了这个键，它会一路走进去。
3. 但**没有任何代码往里写这个键**：全库（`vllm/`，不含 `tests/`）出现 10 处，逐处分类是
   形参声明×2、docstring×2、分支×2、告警文案×2、`enable_kv_events=` 内部转发×2（kvcr 的 `enable_telemetry`
   那种是另一条链）——**赋值 0 处**。
4. 文档在教用户配它：`docs/features/kv_offloading_usage.md:221/248/291` 三处，其中两处是正式的字段表
   （`| enable_kv_events | no | false | ... |`）。
5. **vLLM 自己的工厂不传**：`spec.py` 的 `create_secondary_tier` 只把配置字典展开，而配置字典来自用户的
   YAML/dict —— 也就是说**唯一能写这个键的人就是用户**，代码里没有任何默认注入。

**行为分叉（CPU，真实源码）。** 把 `fs/manager.py` 的 `__init__` 那 8 行事件缓冲逻辑按**真实源码文本**取出执行：

| `enable_kv_events` | 全局 `--kv-events-config` | `tier.events` | 日志 |
|---|---|---|---|
| `False`（唯一可达值） | 开 | `None` | 无 |
| `True` | 开 | `[]`（真的会缓冲） | 无 |
| `True` | 关 | `None` | 一条 `... but KV cache events are disabled globally` 告警 |

⇒ **这条分支不是死代码，是"永远走不到"的活代码**；一旦有人传 `True`，`get_finished_jobs()` 会在 store
成功时往 `self.events` append，`take_events()` 再把它们交给上游。

**那到底算不算缺陷？** 分两种读法，本文两种都写出来：

- **如果不算**：键是**用户可写在 `secondary_tiers` 条目里**的配置项，`**config` 透传是真的，用户写
  `enable_kv_events: true` 确实生效。那这就是"默认值 + 文档已写明"，不是缺陷。
- **如果算**：**`enable_kv_events` 与同段的其它 tier 字段不同类**——`root_dir`、`n_read_threads`、
  `store_config`、`locality` 都是 tier 专属配置，而"要不要把 tier 的 store 发成 KV 事件"是**引擎级决策**，
  与全局 `kv_events_config` 天然耦合；同一矩阵里 `kvcr/manager.py:289` 就从 `extra_config` 直接读
  `enable_telemetry`，**没有让用户手写两层**。而且 pyflag 抓到的正是这个不对称。

⇒ **本文的结论只到"这条分支在库内无生产者"这一步**，价值定 **4/10**（不像 SGLang 那两条是纯内部逻辑；
这一条用户可用写法绕过去）。对外**待用户逐项授权**。

### 上游查重（2026-11-05，只读 GitHub API）

| 检索 | 结果 |
|---|---|
| `"enable_kv_events"` in title | 1 条：**#61893（open）**"Per-connector enable_kv_events switch for KV cache events"——但它是 **KV connector 级**、不是 tier 级，方向不同 |
| `"secondary_tiers" "enable_kv_events"` | 1 条：**#47923**（closed，就是引入这个键的 PR） |
| `tier.enable_kv_events` | 0 条 |
| `secondary tier` + `kv events`（60 条里翻前 30） | 无与本条重复者；相关的是 #63037／#64613／#65548（都是事件**元数据丢失 / 身份**方向，不是开关可达性） |

⇒ **无重复 issue**。但"无重复 issue"不等于"值得提"：这条更像**设计不一致**（B 类里的"讨论"而非"bug"），
且用户已有可用写法。

## 第二条：`no_thinking` 是模板里的一行硬编码

`JinaRankingIOProcessorMixin.format_docs_prompts_func(..., no_thinking: bool = True)` 在 `:884-885`：

```python
if no_thinking:
    suffix += "<think>\n\n</think>\n\n"
```

4 个 online/offline 调用点全都不传它（`scoring/io_processor.py:954/960/1002/1008`），所以 Jina re-ranker 的
**每一条**文档 prompt 都带一段空的 thinking 块。**没有入口能关**（`model_config`、`PoolingParams`、
`chat_template_kwargs` 都没有对应键）。判定标签不写成"缺陷"：这更像作者有意固定输出形状，
价值 **2/10**，不建议对外。

## 方法：三条自我纠错 + 一条对照组

**① 子串计数会把 docstring、告警文案、`if flag:` 一起算进去。** 第一版 F1 判据用
`text.count("enable_kv_events") == 3` 判"两个 tier 各 3 处"，实际各 4 处（多一处**告警文案**），当场报 MISMATCH。
现改为逐行正则分类（形参声明 / 分支 / 告警 / 赋值），并用"赋值 0 处"这个有语义的条件下判断。
这正是 pyflag readme 第四类易错点的同族。

**② 自检不能靠"注入了就翻转"的空壳。** 第二版把注入写成 `ok = False if inj else ok`——8 条注入全部"被抓住"，
但**一条判据都没参与**（红的是同一个 verdict 变量）。现在每条注入走 `inj_fail(name, cond)`，强制让
**对应那条判据**变红；自检输出会打印红的是哪几条，能看出是否只有目标条。

**③ 分叉判定写反了。** `kv_events_fork.py` 第一版用 `<=` 的两档位置切片比较，排序后把
"global=ON/enable=False" 与 "global=OFF/enable=True" 配成一对，判出 `behavioral_fork=False`（看似没分叉）。
实际同一全局配置下 enable 两档不同。现在显式 `pick(tier, enable, global_events)` 取对。

**对照组（不只报真缺陷）。** `use_page_cache` 同样被 pyflag 报成 `silently_inert`，但它是接通的：
`page_cache_e2e.py` 直接 `exec` **真实的配置处理段**（`simple_cpu_offload_connector.py` 的
`extra_config` 解析，去掉 GPU 构造），跑了配置缺省 / `true` / `false` 三种写法，读数：

```
case=unset  config=<absent> parsed=False
case=true   config=True     parsed=True
case=false  config=False    parsed=False
scheduler_passes_to_worker: True   worker_stores_attr: True   backend_branches: True
verdict: WIRED
```

⇒ **同一批里既有真缺陷也有误报**，这是"报总量没意义、必须逐条判"的正面证据。

## 核对产物与复现

```bash
UPSTREAM=/tmp/vllm-scan-20261105 python3 verify_findings.py            # 37 checks, 0 failed, exit 0
UPSTREAM=/tmp/vllm-scan-20261105 python3 verify_findings.py --json     # -> findings.json
UPSTREAM=/tmp/vllm-scan-20261105 python3 verify_findings.py --selftest # 8 条注入全部命中目标条, exit 0
UPSTREAM=/tmp/vllm-scan-20261105 python3 page_cache_e2e.py --json      # verdict: WIRED, exit 0
UPSTREAM=/tmp/vllm-scan-20261105 python3 page_cache_e2e.py --selftest  # 去掉转发 -> BROKEN, 1/1
UPSTREAM=/tmp/vllm-scan-20261105 python3 kv_events_fork.py --json      # behavioral_fork=True, exit 0
UPSTREAM=/tmp/vllm-scan-20261105 python3 kv_events_fork.py --selftest  # 两个 tier 都分叉 + 全局门控告警, 2/2
```

| 文件 | 作用 |
|---|---|
| `verify_findings.py` | 12 条逐条绑行号，37 条 check，8 条注入 |
| `page_cache_e2e.py` | 对照组：真实配置段跑三档，证明"误报确实是误报" |
| `kv_events_fork.py` | 真缺陷的 CPU 行为分叉（2 tier × 2 开关 × 2 全局） |
| `findings.json` / `page-cache-e2e.json` / `kv-events-fork.json` | 读数归档 |

**边界**：全部是**静态 + CPU 源码执行**。本机没有 torch / nixl，**没有真正启动 vLLM**，没有跑 transfer、
没有发过一条真实 KV 事件。`enable_kv_events` 是否被 out-of-tree tier 使用（第三方可以在自己的
`tier_config` 里写）**未调查**——这恰好是"算不算缺陷"的另一种可能，本文不作判断。

## 留给下一轮

1. 对外动作（issue／PR／文档修订）**全部待用户逐项授权**；本条更接近"设计不一致"，若要动，最小形态是
   在文档里写清"这个键目前需要用户手写"，而不是提 bug。
2. 若判"不算缺陷"，本批价值降为**方法读数**：vLLM 上 pyflag 的 12/263 里真缺陷 1 条、误报 4 类，
   与 SGLang 的 34/386 同形——**大仓库上总量永远不能直接用**。
3. `enable_trace_replay` 与 TM-008（verl 的 `poll_interval`）是同一形状（配置字段 → 属性 → 形参），
   若将来统计"配置可达性"跨框架口径，这三条是同一张表里的行。
4. 下一批换池：TensorRT-LLM／LMDeploy／llama.cpp 都还没扫过（SGLang 与 vLLM 已扫完，不宜重扫）。
