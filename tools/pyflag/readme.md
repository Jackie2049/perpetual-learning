# pyflag — 扫描"配置项有没有生产者"（布尔开关静默失效）

## 用在哪

一个布尔开关在源码里有**消费者**（函数形参 + `if flag:` 分支），却找不到任何
**生产者**（调用点传值），那它在本快照里不可达：用户设了也没用。这类问题不看
代码很难发现——开关名在 config、trainer、engine 里各出现一次，grep 起来"到处都在用"。

工具只做只读静态分析（AST），不导入、不运行目标仓库。C++ 版是正则级结构分析，
不做符号解析，见下文。

## 怎么用

```bash
python3 scan_unwired_flags.py /path/to/repo               # 打印 JSON 报告
python3 scan_unwired_flags.py /path/to/repo --json <path>  # 同时落盘到指定路径
python3 scan_unwired_flags.py /path/to/repo --json         # 裸用 → scan-<repo>.json（写到脚本目录）
python3 -m unittest test_scan_unwired_flags -v             # 14 个用例，含 7 条易错点
```

C++ 仓库用 `scan_cpp_flags.py`，见下文「C++ 版」。

**2026-11-05 修**：`--json <path>` 原来会把路径当第二个位置参数，工具直接打用法并 `exit 2`
（`args` 收集了所有非 `--` 开头的词）。现在解析器显式跳过 `--json` 后面那个词。

自动发现的开关：形参带 `= True/False` 默认值、且名字以 `use_` / `enable_` /
`disable_` / `no_` / `with_` 开头。

## 输出怎么读

每条 flag 给四个数和一个判定：

| 字段 | 含义 |
|---|---|
| `reads_nontest` | 非测试代码里出现该名字的**行数**（粗粒度，只作参照） |
| `consumers_nontest` | 把它当形参、且在函数体内**按它分支**的函数 |
| `producers_nontest` | 真的把值**传进去**的调用点（关键字或位置实参） |
| `silently_inert` | 有消费者、无生产者 |
| `mode` | `never` = 默认 False 且无人传 ⇒ **分支永不执行**；`always` = 默认 True 且无人传 ⇒ **分支总是执行**；`mixed` = 多个消费者默认值不一致 |

`mode` 这个区分很关键：同样是"没人传"，`never` 是功能静默失效，`always`
是功能静默生效（用户以为能关，其实关不掉），两者是完全不同的问题。

## 四个易错点（都真实踩过，测试里各有一条）

1. **构造器的调用点写的是类名**。消费者是 `Actor.__init__`，调用点是 `Actor(...)`。
   不处理会把所有类构造全部漏判。
2. **Ray 的 `.remote(...)` 包了一层**。`engine.init_group.remote(..., use_ray=x)`
   的真实被调方法是 `init_group`，不是 `remote`。
3. **同名局部变量覆盖形参**。函数体内 `use_ray = getattr(...)` 之后
   `if use_ray:` 读的是局部变量，不再构成"按该形参分支"。
4. **形参默认值不是生产者**。`def f(use_x=False)` 这一行里出现了名字，但没人调用它。

## 已知边界（结果必须带着这些读）

- **只有静态可达性判断**，不做数据流/跨进程分析。通过配置文件字典间接传参
  （`**cfg`、`getattr(ns, flag)`）**检测不到**，会漏报——本工具报"有生产者"时可信，
  报"无生产者"时仍需人工确认没有间接路径。
- **同名不同函数会串**。只按名字匹配调用点，不做符号解析。
- `reads_nontest` 是行数不是引用数，别当精度看。
- 测试/示例/文档目录按路径排除（`tests/`、`examples/`、`docs/` 等），
  但目录命名不统一时会漏。

## 已跑过的快照（2026-11-02）

| 仓库 | commit | 发现开关 | 判为失效 | 人工复核结论 |
|---|---|---|---|---|
| verl | `6879e1ab07` | 37 | 8 | `use_prefix_grouper`、`use_tiled_mlp` **确认为真**；`use_chunking_entropy` 也在同一批不可达路径上 |
| SkyRL | main | 39 | 7 | `use_liger_kernel`、`use_remove_padding` **确认为真** |
| OpenRLHF | main | 5 | 0 | 修掉三处假阳性后归零（详见测试用例） |
| rLLM | `4e828c8` | 11 | 2 | `use_tci` 是**零调用者的死代码**（不是用户可配开关）；`use_async` 是**假阳性**——形参默认 `False` 但文档写 `None`、演示块传 `True`，扫不到 `None` 这条路径。人工复核后另立 B 类候选 [tool-use-async-autodetect](../../open-source/rllm/tool-use-async-autodetect.md) |

报告存档：`scan-verl-project-verl-6879e1ab07.json`、`scan-NovaSky-AI-SkyRL.json`、
`scan-OpenRLHF-OpenRLHF.json`、`scan-rllm-4e828c8.json`。

## 已跑过的快照（2026-11-04 扩池）

| 仓库 | 快照 | 发现开关 | 判为失效 | 人工结论 |
|---|---|---|---|---|
| SGLang | `27aedf7f` | **386** | **34** | **总量不能直接用**：排名前三的 `enable_cfg_parallel`／`use_decode_cuda_graph`／`use_zigzag_page_table` 都是**属性传参**（第五类），`mode=mixed` 只是多个消费者默认值不一致。逐条取证后只有一条真失效：`fused_moe_router_shim` 的 `enable_deterministic_inference`（见 [TM-013](../../queue_tech-miner.md)） |
| TRL | main（2026-11-04） | **6** | **0** | 全仓 `use_*/enable_*/disable_*` 形参默认值只有 **1 处** ⇒ **这个框架不适合当 pyflag 对象**，勿再扫 |

**这一批最大的方法教训**：仓库越大、读到的路径越间接，`silently_inert` 里的**属性传参误报占比越高**。SGLang 386→34 看着像"问题很多"，实际逐条看是"开关很多、间接传参很多"。⇒ **报结论前先做两问分桶**（是不是某个 config 字段？有没有位置参数调用？），报总量没有意义。

**第二批（2026-11-04，同一快照 `27aedf7f`）：把 34 条**全部归档分类**，真缺陷只剩 2 条。**
用 [build_triage.py](../../experiments/sglang-pyflag-triage/build_triage.py) 按**消费者所在位置**分桶：
内核启动器 2 / 配置 dataclass 3 / `__init__` 里存 self 8 / 源码分支 21；再用
[verify_findings.py](../../experiments/sglang-pyflag-triage/verify_findings.py) 把 34 条逐条绑到可 grep 的事实（形参的每一处读取分「分支 / 转发 / self 属性」），并做 3 条负向对照。结论：

| 标签 | 条数 | 含义 |
|---|---|---|
| `KERNEL_CONSTEXPR` | 10 | Triton/cute 核的 `constexpr`，不传即默认，两条路径等价 |
| `INTERNAL_OR_DEFAULT` | 12 | 真正生产者是模块级常量 / `**kwargs` / 默认值，不是前端选项 |
| `ATTR_PASSING` | 6 | 属性传参（第五类误报） |
| `CONFIG_DEFAULT` | 3 | HF 配置 dataclass 字段 |
| `INTERNAL_PRODUCER` | 1 | `use_dsa`：子类内部传 `use_dsa=True` |
| `TM013_DUP` | 1 | 第一批那条 |
| **`TOPIC_ESCAPE_FAILS_OPEN`** | **1** | **新发现：`use_logits_buffer` 默认值在唯一调用点不可达**（[证据](../../open-source/sglang/evidence/logits-buffer-escape-hatch.md)） |

**新学到的两类"工具看不见的生产者"**（已能机械判出，不必逐条读）：
1. **回调参数**：`cp_strategy.run_attention(q, forward_batch, dev, _inner_fn, backend=…)` 把内部函数
   当参数传下去，`srt/layers/cp/zigzag.py` 再 `use_zigzag_page_table=True` 调回来。⇒ 名字的生产者
   藏在一个**不透明实参**里。判据：消费者所在函数的名字是否作为实参出现在别的调用里。
2. **调用点用「别的东西」代替了形参的语义**：`_get_logits(..., use_logits_buffer=True)` 的注释说
   「形状不匹配就跳过」，调用点却按「块数」传值。⇒ **形参有生产者、但语义被掏空**，
   这是 pyflag 结构上看不见的一类，只能读注释与调用点。
3. **生产者把值钉成了字面量**（2026-11-05，TM-021 补）：MLC-LLM 的 `enable_debug` 被扫成
   `producers_total=1`（生产者是 `cli/serve.py` 的 `enable_debug=parsed.enable_debug`），看着正常。
   但同一名字的**另一处**生产者在 `router/router.py:82` 是 `PopenServer(..., enable_debug=True, ...)`
   —— `Router` 的形参表里没有它、router 的 CLI 也没有 ⇒ **这条部署路径上恒真且用户关不掉**。
   ⇒ 判据：同一形参的多个生产者里只要有一处是**字面量**，那条路径就不受配置控制；
   `producers_total > 0` 只说明"有人传"，说明不了"配得动"。这一类要按**生产者的实参形态**分桶才看得见
   （本工具现在只报 `how: keyword|positional`，不区分字面量与变量——**尚未自动化**）。
   见 [证据页](../../open-source/mlcllm/evidence/router-debug-endpoints-always-on.md)。
4. **`functools.partial` 隔了一层（2026-11-05，TM-022 补）**：AReaL 的三个 loss 开关
   （`use_sapo_loss` / `use_cispo_loss` / `use_decoupled_loss`）被报 `producers_total=0`，
   实际生产者在 `actor.py:612` 的 `loss_fn = functools.partial(grpo_loss_fn, ..., use_sapo_loss=self.config.use_sapo_loss, ...)`。
   扫描器收集调用点时只看 `Call.func` 的名字是不是 `grpo_loss_fn`，而这里是
   `partial` 的**第一个位置实参**。⇒ 判"没生产者"之前先看：这个函数有没有作为
   `partial` / `lambda` / 装饰器的第一个实参出现过。**尚未进扫描器**（要做就得把
   `partial(f, ...)` 展开成一次 `f(...)` 调用，位置实参的对应关系也会跟着变）。
   见 [AReaL 批](../../experiments/areal-pyflag-triage/README.md)。

**跑 SGLang 的耗时**：`python3 scan_unwired_flags.py <sglang>` 约 **10.5 分钟**（8000+ 文件），且 `--json` 落盘到**脚本所在目录** `tools/pyflag/scan-<root>.json`，不是被测仓库里——第一次找错了地方。

## 已跑过的快照（2026-11-05 扩池第二批：vLLM）

| 仓库 | 快照 | 发现开关 | 判为失效 | 人工结论 |
|---|---|---|---|---|
| vLLM | `da0b6ac2` | **263** | **12** | 逐条取证后**真缺陷 1 条**（`enable_kv_events`，[TM-015](../../queue_tech-miner.md)）；1 条模板硬编码（`no_thinking`）；其余 10 条四类结构原因。**同一批里也证明了它会误报**：`use_page_cache` 被报成 `never`，实际经 `self.use_page_cache` 属性转发接通 |

扫描耗时 **2 分 42 秒**（5976 个 `.py`）。读数归档 `scan-vllm-da0b6ac2.json`，逐条判定见
[experiments/vllm-pyflag-triage](../../experiments/vllm-pyflag-triage/README.md)（`verify_findings.py` 37 checks、8 条注入全部命中目标条）。

## 已跑过的快照（2026-11-05 扩池第三批：LMDeploy）

| 仓库 | 快照 | 发现开关 | 判为失效 | 人工结论 |
|---|---|---|---|---|
| LMDeploy | `3228395546` | **38** | **5** | 逐条取证后**真缺陷 0 条**：1 条 `**kwargs` 转发误判（`disable_fastapi_docs`，通路已在 CPU 上跑通）、1 条 import 期硬件探测全局（`use_fa3`）、2 条测试/内部形参（`use_proxy`／`with_version`）、1 条库参数经 `__call__` 透传可达（`use_tqdm`）⇒ **负结论，不对外**（[TM-016](../../queue_tech-miner.md)） |

扫描耗时 **16.5 秒**（1113 个 `.py`）。读数归档 `scan-lmdeploy-main.json`，逐条判定见
[experiments/lmdeploy-pyflag-triage](../../experiments/lmdeploy-pyflag-triage/README.md)（`verify_findings.py` 39 checks / selftest 8/8、
`probe_kwargs_route.py` 7/7 含 2 条负向对照）。

**按 SGLang → vLLM → LMDeploy 三次的同一形状收拢**：

| 大仓库（>1000 文件） | 开关 | `silently_inert` | 真缺陷 | 误报/结构原因 |
|---|---|---|---|---|
| SGLang `27aedf7f` | 386 | 34 | 2 | 32（5 类） |
| vLLM `da0b6ac2` | 263 | 12 | 1 | 11（4 类） |
| LMDeploy `32283955` | 38 | 5 | **0** | 5（5 类） |

⇒ 真缺陷率 **< 6%**；LMDeploy 这一批是**全 0**（详见 [证据页](../../open-source/lmdeploy/evidence/inert-flags-triage-2026-11.md)）。
前两批是「**属性转发**」与「**config 字典键**」占多数，第三批换成「**`**kwargs` 兜底**」「**别名调用**」与
「**测试目录／模块级全局**」。**共同点是：扫描器的输出只能当候选入口，分桶（是不是 config 字段 /
有没有属性转发 / 签名有没有 `**kwargs` / 调用点是别名还是原名 / 在哪一层消费）比总量更有信息量。**

## 已跑过的快照（2026-11-05 扩池第四批：TensorRT-LLM）

| 仓库 | 快照 | 发现开关 | 判为失效 | 人工结论 |
|---|---|---|---|---|
| TensorRT-LLM | `f6405a5965` | **352** | **35** | 逐条归因后**真缺陷 1 条**：`use_split_scale_stages`——形参在、两处 `elif` 在、同族四个兄弟 constexpr 在调用点都显式传了 `True`，只有它没传，而兄弟分支恒先命中 ⇒ 两处 `elif` 永不执行；合入 PR 当天（`4bbe1af6`）即如此，非回归。其余 34 条分 17 类结构原因（`self.` 属性 5、同文件构造器 6、死代码 3、计算值 2、`**dict` 2、构建脚本 2、配置字典 2、`getattr` 2、量化内部 2，其余各 1）。**这一条是新增的 `shadowed_arms` 自己扫出来的**，见第十一类 |

扫描耗时 **12 分 39 秒**（6084 个 `.py`），读数归档
[scan-trtllm-f6405a5965.json](scan-trtllm-f6405a5965.json)，逐条归因见
[experiments/trtllm-pyflag-triage](../../experiments/trtllm-pyflag-triage/README.md)
（`verify_findings.py` 28 checks / selftest 6/6，`probe_split_scale.py` 9/9 / selftest 2/2，
`classify.py` 35 条全覆盖）。

**四批扩池的同一形状**：

| 仓库 | 文件 | 开关 | 失效 | 真缺陷 | 真缺陷率 |
|---|---|---|---|---|---|
| SGLang `27aedf7f` | — | 386 | 34 | 2 | 5.9% |
| vLLM `da0b6ac2` | 5976 | 263 | 12 | 1 | 8.3% |
| LMDeploy `32283955` | 1113 | 38 | 5 | 0 | 0% |
| TensorRT-LLM `f6405a59` | 6084 | 352 | 35 | **1** | **2.9%** |

⇒ 失效条目里真缺陷的比例稳定在 **0–8%**。四批合计 **64 条候选 → 4 条真缺陷**。
**总量没有任何判据价值，分桶才是产出**；而且每换一个框架，占多数的那几类就换一批。

**归档命名注意（已修）**：`--json` 原来固定写 `scan-<目录名>.json`，而目标写成 `.` 时目录名是空串
⇒ 落成 `scan-.json` 并**互相覆盖**。第四批上真的踩了：两次扫描结果都写进同一个 `scan-.json`。
现在 `--json <path>` 写到指定路径，裸 `--json` 写 `scan-<repo|repo>.json`；`readme` 与
`scan_unwired_flags.py --json ../scan-trtllm-<sha>.json` 都可用。

## 第八类易错点：签名里没有该键 ≠ 收不到（`**kwargs` 兜底）

`def launch_server(num_nodes, node_rank, model_path, backend_config, proxy_url, server_port, **kwargs)`
—— 具名形参里**没有** `disable_fastapi_docs`，只看签名会判"断在这里"。但函数体里
`cur_server_kwargs.update(kwargs)` 把它原样收下、再 `serve(model_path, **kwargs)` 交给真正的消费者。
LMDeploy 的 `disable_fastapi_docs` 就是这样：`probe_kwargs_route.py` 把这三段真实源码在 CPU 上串起来跑，
`True` 时 FastAPI 三个 URL 全变 `None`、`False` 时 `docs_url='/'`。

⇒ **判「收不到」之前先看签名有没有 `**kwargs`，再看 kwargs 有没有被转发到消费者。**
`silently_inert=True` 只能说明"没有按同名关键字传"，说明不了"值到不了"。

## 第九类易错点：调用点写的是别名

`from lmdeploy.serve.openai.api_server import serve as run_api_server` 之后，调用点写的是
`run_api_server(`。按原名 `serve` 找调用点读到 **0 处**——同一份文件里明明传了值。
LMDeploy 的 `disable_fastapi_docs` 第一版判据当场停在这上面（A4/A5 两条红）。

⇒ 找调用点前先把 `from ... import x as y` 的别名表建出来，所有 `y(` 都要算成对 `x` 的调用。
同类还有 `**kwargs` 兜底与 `getattr(ns, name)`——三者共同点：**名字与调用形式不是一一对应的**。


## 第十类易错点：`**dict` 展开——发送端兜底

第八类是**接收端**兜底（签名有 `**kwargs`）；这是它的镜像：**发送端把值塞进字典再展开**。

```python
prepare_kwargs = {"batch_size": ..., "use_causal_mask": key.mask_type == "causal", ...}
prepare_routes = _PrepareBsrRoutes(**prepare_kwargs)
```

TensorRT-LLM 的 `use_causal_mask` 就这样被判成 `silently_inert`。字典字面量里的键是**字符串**，
AST 里 `ast.Call.keywords` 是空的，任何按关键字实参找生产者的写法都读不到。

⇒ 判「没有生产者」时，除了看 `**kwargs`（接收端），还要看本文件／本仓有没有
`{"<name>": ...}` 这样的字典键，以及该字典是否被 `**` 展开到消费者。

## 第十一类易错点：兄弟开关恒先命中——分支写了但永远走不到

这是本工具**唯一一类能产出高价值真缺陷、但结构上完全看不见**的情况：

```python
if cutlass.const_expr(use_b_byte_offset):
    ...
elif cutlass.const_expr(use_split_scale_stages):   # ← 永远轮不到
    ...
```

`use_split_scale_stages` 有声明、有分支、确实没生产者，看起来和普通误报一样。区别在于
**同一 `if/elif` 链上前面的 arm 在所有真实调用点都恒真** ⇒ 这段代码不但没人传值，连带着
`elif` 之后的兜底也一并失效。

⇒ 单看「有没有生产者」不够，还要看**同一个 if/elif 链**。**这一步 2026-11-05 已自动化**：
报告的 `shadowed_arms` 字段列出「某条 arm 被同链前面的兄弟条件遮蔽」的位置，判据是

- arm 的条件正好是**一个**名字（可包一层 `cutlass.const_expr(...)`）；
- 同一 if/elif 链前面有另一条 arm，条件是**同一个函数**的另一个形参；
- 该形参在**某个调用点**被传字面量 `True`，且**没有任何调用点**传可能为 False 的值
  （字面 `False`、裸变量名、任意表达式都算可能为 False）；
- `f(flag=flag)` 只在函数体没重新赋值该形参时才算"纯转发"，否则是计算值。

TensorRT-LLM `f6405a5965` 上全仓命中 15 条、非测试 9 条，人工复核 **1 条假阳性**
（同一函数多个调用点里有一个省略了兄弟参数，而省略走的是默认值——见下）。
**已知假阳性必须当候选读**，不要直接当结论。

关于默认值：调用点**省略**某个参数时走的是默认值。`False` 默认会折进"可能为 False"；
`True` 默认不折进"钉死"，因为那属于"这段代码天生就走不到"，pyflag 已经在 `mode=always`
里报过，重复计数没有收益。

**已实现**：`scan_unwired_flags.py` 的 `shadowed_arms()`，对**已经判为 `silently_inert` 的 flag**
再查一遍同一 `if/elif` 链。只在一个前提下报警：arm 的条件**恰好就是这个 flag 名**、前面某条 arm
测的是同函数的**另一个形参**、且该形参**至少一个调用点传了字面量 `True`、没有任何调用点传字面量
`False` 或别的字面量**（`has = bias is not None` 这种计算值、以及只做透传的 `f(x=x)` 调用点都不算
决定）。宁可漏报：不做多跳前向解析，命中为空是预期的假阴性而不是 bug。`--json` 输出多一个
`shadowed_arms` 键，与 flag 表并列。

⇒ 这条判据的价值在于**它抓到的那一类结构上完全看不见**：前四批 63 条候选里没有一条能用
「有没有生产者」判出来（真缺陷只有 4 条），而这条判据第一次跑就命中了 `use_split_scale_stages`。

**在真快照上的读数（TensorRT-LLM `f6405a5965`，`git status` 干净）**：两个文件各命中 2 条，
合计 4 处死分支——主文件 `:2668`／`:2675`（`use_split_scale_stages`）与孪生文件 `:3050`／`:3057`，
分别被 `use_sfb_stage_byte_offset`／`use_sfa_stage_byte_offset` 挡。主文件另命中 2 处
`use_split_operand_stages`（`:2661`／`:2692`），但这个 flag 本身**不在** `silently_inert` 名单里，
扫描器只对 inert 入选者升级，所以最终归档里只有 4 条 `use_split_scale_stages`。

**这条判据只认一种机制**：调用点把兄弟**字面量置真**。本例 PV 调用点 `:4522` 是另一种——
`stage_sfa_from_smem=False` 让整个 `if` 块进不去，与兄弟无关，记在 `F1h.pv_dead_by_enclosing_guard`。
⇒ **命中为空不代表分支可活**，只代表"不是兄弟挡路"；反之命中也不代表其它调用点没有别的死法。

## 第七类易错点：`text.count(name)` 会把 docstring、告警文案、`if name:` 一起算进去

判定「某处没有传这个值」时最自然的写法是 `src.count("enable_kv_events=") == 0`。但**同一个名字**在
形参声明（`enable_kv_events: bool = False,`）、docstring 续行、`if enable_kv_events:` 分支、
告警文案（`"enable_kv_events is set on secondary tier ..."`）里都会出现。vLLM 那一轮第一版判据用
`count(...) == 3` 判「每个 tier 各 3 处」，实际各 4 处（多一处告警文案），当场报 MISMATCH。

⇒ **「有没有生产者」必须逐行分类判**：先排除 `def` 行、docstring、`if flag:`、字符串字面量，剩下的才算赋值。
这与第四类（形参默认值不是生产者）是同一个坑的另一种触发方式。

## 自检的第二个坑：注入必须让**对应那条判据**变红

vLLM 那一轮的第一版自检写成 `ok = False if inj else ok`——8 条注入「全部被抓住」，但**一条判据都没参与**，
红的是同一个 verdict 变量。改法是把注入绑到具体 check 上（`inj_fail(name, cond)`），
并在自检输出里打印**红的是哪几条**：若红的不是目标那条，说明这条判据根本没在看那个东西。

## 第五类易错点：生产者是**属性**（或位置参数），不是关键字

`enable_sample_support_replay`、`enable_pd` 在 SkyRL 上都报「无生产者」，但两者都**能生效**：

```python
sample_support_replay = self.cfg.algorithm.enable_sample_support_replay   # 读配置字段
self.model(..., enable_sample_support_replay=sample_support_replay)       # 传给形参
```

工具按形参名找调用点，看到的是「形参 `enable_sample_support_replay` 没人传同名关键字」，
而真正被配置的是**同名的 dataclass 字段**。⇒ 报 silently_inert 的条目里，
**「名字同时是某个 config 字段」的那几条要先按字段查一遍读取点**，再判断是不是失效。

SkyRL 2026-11-04（`1089e18`）的 7 条里，靠这一条排除掉的是 `enable_sample_support_replay`
与 `enable_pd` 两条；`use_liger_kernel` 恰好相反——它在 config 里**没有**字段，
所以「无生产者」是真的。逐条判定与脚本见
[experiments/skyrl-pyflag-triage](../../experiments/skyrl-pyflag-triage/readme.md)。

同类补充：**位置参数也是生产者**。`Qwen4ExpFinalMixer` 用
`super().__init__(config, use_combine=False)` 传值，工具只认 `use_combine=...` 关键字，
于是把内部形参报成「无人传」。

## 第六类易错点：三态形参

`use_async: bool | None = None` 这类**三态形参**是扫描的盲区。工具只认 `= True/False`，
所以把默认写成 `False`、却在文档里宣称"默认 None、自动探测"的代码，会被报成
"默认 False 且无人传 ⇒ 分支永不执行"，看起来像漏报，实际是**默认值与文档不符**这一
不同的问题。rLLM 的例子正是如此：真缺陷不在"没人传"，而在"传了 `None` 也判不对"。

## C++ 版：scan_cpp_flags.py（2026-11-05 第五批：llama.cpp）

Python 版从**函数形参默认值**出发；C++ 里这类开关的主要形态是**结构体成员默认值**
（`struct X { bool f = false; }`），所以单开一个脚本，三步口径不变：
**字段声明 → 消费者（出现在 `if`/三目/`&&`/`||` 里）→ 生产者（被赋值 / 传成实参 / 聚合初始化）**。

```bash
python3 scan_cpp_flags.py --selftest                       # 7/7，含 3 条负向对照
python3 scan_cpp_flags.py /path/to/repo [--json <path>] [--min-members N]
```

读法与 Python 版一致：`unwritten` = 有消费者、无写入。**同理只是候选，不是缺陷清单。**

### 已跑过的快照

| 仓库 | 快照 | 文件 | 有分支字段 | `unwritten` | 人工结论 |
|---|---|---|---|---|---|
| llama.cpp | `3de52ac3` | 1574 | 1636 | **37** | 逐条归因后**真缺陷 0 条**（[证据](../../open-source/llamacpp/evidence/cpp-fields-triage-2026-11.md)）。同日 Python 版扫它的 250 个 `.py`：7 个开关 / 0 条失效 ⇒ **对 llama.cpp 用 Python 版没有信息量** |

耗时约 **4 分钟**。37 条里 15 条是 ggml 后端/内核参数结构体、4 条测试夹具、4 条同名局部变量，
其余各自能指到真实写入点。

### C++ 特有的三类写入形态（都是"报没人写、实际写了"的真反例）

1. **单行 `if` 里赋值**：`/**/ if (v == "jsonl") { params.x = true; }`。
   这一行既是条件行又是写入行；先判消费者会把整个字段记成"只读"。⇒ 必须先判
   "本字段在这行有没有被写"，再决定它是不是纯消费者。
2. **`/* name */ value` 聚合初始化**：`chat_params = { /* allow_image */ mtmd_cap.vision, … }`。
   剥掉注释后整行只剩 `mtmd_cap.vision,`，字段名根本不出现。⇒ 要回看**未剥注释的原始行**。
3. **位置实参**：`slots.push_back({"tower_pos", 1, 1, true})`（第四项是 `in_place`），
   而同结构体在另一个文件只传三项。⇒ 没有通用修法，只能人工复核——这也是"`unwritten`
   必须人读"的又一条理由。

### 第十二类易错点：引用绑定的写入（C++ 版）

`server-schema.cpp` 的 `field_bool(name, bool &)` 在构造时**按引用绑定字段**，真正的写入
发生在 `field_bool::eval` 里的 `val = data.at(n).get<bool>()`。源码里通篇只出现
`params.sampling.reasoning_control`，没有 `x.field = ` 这样的形式 ⇒ 扫描器报 `unwritten`，
实际完全接通。**这是第八类（签名里没有 ≠ 收不到）在 C++ 里的同族问题：绑定 ≠ 赋值。**

### 第十三类（C++）：只认 `=` 会漏掉 `++` / `--` / 构造函数初始化列表

`req_count++`、`n_attempts++`、`enabled_(enabled)`、`/*.shares_model =*/`（注释名带点号）
这四种写法在 llama.cpp 里都存在。本批是人工复核补上的，尚未进扫描器默认判据。

### 第十三类（Python 版）：工厂函数被当成消费者（2026-11-05 已修）

**症状**：MLC-LLM `ab90…`/`44fd06d0` 批，`enable_prefix_cache` 被报 `mixed` + `silently_inert=True`，
`producers_total=0`。翻个案：`python/mlc_llm/interface/router.py:17` 的 `serve()` 被扫成消费者——
它的函数体里确实**没有** `if enable_prefix_cache:`，可扫描器把"名字出现在函数体里"就算成消费，
而它出现在 `Router(..., enable_prefix_cache=enable_prefix_cache, ...)` 这个**实参位置**。

**这就是工厂／包装函数的形状**：`serve()` 只是把 CLI 的值原样递给下一跳，决策在 `Router` 里。
`probe_router_chain.py` 把三处真实源码在 CPU 上跑起来（真 `Router.__init__`、真
`interface.router.serve`、真 argparse）：`True` → 两台 `PopenServer` 都拿到 `radix`，
`False` → 都拿到 `disable`，CLI 两档都如实到达 `Router` ⇒ **接线是通的**。

**修法（已进工具）**：
- 名字作为**某个调用的关键字实参**出现也算消费（否则 `return Cls(f=flag)` 这类"选择"看不见）；
- 形参只被"传给下游、从不被 `if` / 赋值 / 属性访问读到"的消费者标 `param_default='forwarded'`，
  数据落 `deciding_consumers` 字段，`silently_inert` 只由**决策者**决定。

**第一版的同类误判（同一批，同一个根）**：反事实注入写成
`src.replace("enable_prefix_cache=enable_prefix_cache,", "")`，那条字符串同时命中转发行和**签名行**，
注入后 `Router(...)` 照样收到关键字 ⇒ 探针自己报了 `CHAIN_BROKEN`。是 `--selftest` 的红灯
暴露的："这是把读的路走成了死路"。⇒ **注入必须只改调用点**（按 AST 定位 `Router(...)` 再删实参）。

**同族：容器键位写入**。`kwargs["enable_disaggregation"] = True` 这种写法原来既不算生产者、
也不算消费（`If`/`Assign`/属性都不会匹配到它）。现在单独出 `container_writes` 字段**只作线索**，
**不进 `producers_total`**——字典里写了键不等于下游真读到，判"写了没人读"仍然成立。

### 已跑过的快照（2026-11-05 扩池第五批 + 第六批：llama.cpp C++ 版见上，MLC-LLM）

| 仓库 | 快照 | 统计口径 | 发现开关 | 判为失效 / `unwritten` | 人工结论 |
|---|---|---|---|---|---|
| MLC-LLM | `44fd06d04f` | Python `scan_unwired_flags.py` | 7 | 2（修复前）→ **0**（修复后） | 两条都是**工厂转发误判**，见第十三类（Python 版）。真缺陷 0 条 |
| MLC-LLM | `44fd06d04f` | C++ `scan_cpp_flags.py` | 89 个文件 / 22 个字段 | 1 | `PrefillInput::is_decode`：结构体字段 + hybrid 预填时置 `true` + 三处消费者，**接通** |

扫描耗时：Python **4.0 秒**（415 个 `.py`）、C++ **0.5 秒**（88 个 C/C++ 文件）。
C++ 侧只有 22 个"有消费者且按它分支"的结构体字段——MLC-LLM 的 C++ 部分是服务层
（engine / request_state / engine_actions），配置面主要在 Python 与 `mlc-chat-config.json`，
**C++ 版扫它的信息量本来就低**。

**第六批的后续（同日 TM-021）**：`enable_debug` 被本表判为"正常接线"（`producers_total=1`），
但顺着这个生产者往下查发现**另一处生产者是字面量 `True`**，而那条路径（`mlc_llm router`）
的形参表与 CLI 里都没有这个开关 ⇒ 恒开且关不掉。见上文「工具看不见的生产者」第 3 条
与 [证据页](../../open-source/mlcllm/evidence/router-debug-endpoints-always-on.md)。
⇒ **`producers_total > 0` 只能证明"有人传"，不能证明"配得动"。**

同批另有两个"扫描器看不见但接通"的形状，一并记在这里：

1. **属性包装**：`self.no_debug_config` 被赋值、后面 `if not self.no_debug_config:` 读——
   消费者是 `__init__`，生产者是构造实参，扫描器看得见，但"读取点"里那条 `self.` 属性读不算消费。
2. **编译期按模型接**：`PagedKVCache.create_generic(..., enable_disaggregation=...)` ——
   38 个模型里只有 **3 个**（llama / llama4 / nemotron）显式传，其余 35 个吃形参默认 `False`；
   `dispatch_kv_cache_creation.py` 里 `kwargs.pop(...)` 再 `kwargs[...] = ...` 的 `kwargs` 是
   **具名形参**（不是 `**kwargs`），只改这个字典自己。⇒ 不是运行时开关，属设计面。

## 训练侧（2026-11-05，扩池第七批：AReaL）

推理引擎六批扫完后换到**训练侧**，第一个对象是 AReaL `3b936798`（2026-10-19，1141 个 `.py`、
**32 秒**）：**42 个开关 / 7 条 `silently_inert`**。逐条归因后：**R1 真实接线 4 条**（全是同一个盲区：
`functools.partial`，见上文第 4 条）、**R2 真不可达 1 条**（`use_cpu`）、**R3 内部默认 2 条**
（`use_dist_ckpt` / `with_rng`，无配置字段）。⇒ **没有任何一条是"用户配了没反应"的开关**。

| 仓库 | 快照 | 文件 | 开关 | 失效 | 人工结论 |
|---|---|---|---|---|---|
| AReaL | `3b936798` | 1141 | 42 | 7 | R1 4（`partial` 盲区）/ R2 1（`use_cpu`）/ R3 2（内部默认）；真缺陷 0 条 |

**R2 那条值得单独记**：`archon/moe/kernels.py:155` 的 `generate_permute_indices(..., use_cpu=False)`
在 `:211` 有 `if use_cpu: fill_indices_cpu(...) else: fill_indices_wrapper(...)`，
而唯一调用点 `moe/utils.py:212` 只传 5 个位置实参、不传它，全仓（含 `tests/`）**零生产者**
⇒ CPU 实现从未被执行过。docstring `:172` 仍写着 "whether to use CPU implementation"。
但这条**不是用户配置**（没有 CLI、没有 config 字段），是开发期调试开关 ⇒ 对外价值低。
**后续批已把后半句查完**（[use-cpu-fallback](../../experiments/areal-use-cpu-fallback/README.md)）：
CPU 实现在上游缓冲口径下与 GPU 排布逐位相同（**不是**坏掉的回退），而 `kernels.py` 顶层
`import triton` ⇒ 缺 Triton 的机器整模块进不来，测试又用 `skipif(not HAS_TRITON)` 静默跳过。
⇒ 口径：**开关不可达 + 实现可用**。方法教训：判"回退路径坏了"之前先对齐调用方实际给的
缓冲/参数口径——本批三次误判有两次栽在这里。

**训练侧与推理引擎的形状差异**（下一批换框架时要注意）：训练框架的开关大多挂在
**config dataclass 字段**上（`cli_args.py` 里 `field(...)`），而不是函数形参；
本工具按"形参默认值"发现开关，所以**只能看见函数层面的那部分**——AReaL 42 个开关
相对它数百个配置字段是很小的覆盖。⇒ 训练侧若要系统覆盖，得另做"配置字段 → 消费点"的扫描，
不能照搬 pyflag 的口径。

## 判定的分量

"有消费者无生产者"是**必要条件**，不是充分条件。说"这个开关是死代码"之前，
至少要排除：① 间接传参（`**kwargs` / 配置对象属性）；② 插件或动态注册；
③ 下游 fork 的调用约定。本工具只负责把候选缩到几条，结论仍要人读代码。

## 附：scan_test_assertions.py — 测试能不能失败

另一个同类扫描器，起因是 rLLM 的 `tests/tools/test_tools.py` **有 0 个 assert**——它只 `print`，
失败也只打印一行。于是上游一个真实的分派缺陷被"测试通过"掩盖了很久。

```bash
python3 scan_test_assertions.py <repo_root> [--json]
```

输出四类：

| 字段 | 含义 |
|---|---|
| `files_no_assert` | 零断言且**看不出委托**——真候选 |
| `files_no_assert_but_may_delegate` | 零断言，但源码里有 `check_allclose` / `validate=` / `verify=` 等——**可能把校验委托出去了，需人读** |
| `files_assert_in_text_only` | 文本里有 "assert" 但 AST 里零个——多半在注释或字符串里 |
| `files_print_as_check` | `if ...: print("fail...")` 且该分支无 assert/raise |

2026-11-03 在三个框架上跑过：

| 仓库 | 测试文件 | 零断言 | 可能委托 | print 当判据 |
|---|---|---|---|---|
| rLLM `4e828c8` | 104 | 1 | 1 | **1** |
| SkyRL main | 337 | 6 | 1 | 1 |
| verl `6879e17` | 257 | 8 | 5 | 4 |

### 已知假阳性（必须人读复核）

- **`test_import.py` 这类是合法的**：import 成功即通过，本来就不该有断言。
- **委托型**：`test_correctness_on_gpu.py` 把 `check_allclose=True` 传给下游 worker，
  断言在别的模块里。扫描器只能标"可能委托"，不能判定。
- **框架型**：某些参数化 / fixture 断言写在 conftest 里。

### 这条扫描的定位

**它只筛候选，不下判决。** "零断言"是"可能掩盖缺陷"的必要条件，不是充分条件；
反过来，有断言的测试也可能断言写错。真正确认需要：**把被测代码改坏，看测试会不会红**。

### 一个真实的代价

rLLM 那个例子里，`test_python_tool` 之所以"过"，恰恰是因为缺陷吞掉了异常——
**测试不但没覆盖缺陷，还依赖缺陷才通过**。这类"靠 bug 才绿"的测试，比没有测试更危险，
因为它给出的是**错误的信心**。

## 已跑过的快照（2026-11-05 扩池第六批：MLC-LLM）

见上「第五批 + 第六批」那张表。结论一句话：**7 个开关 / 2 条候选 → 修复后 0 条；
22 个 C++ 字段 / 1 条候选 → 0 条**。两条 Python 候选都是**工厂转发**误判
（`interface.router.serve` 把 `enable_prefix_cache` 原样递给 `Router`、
`interface.serve` 把 `enable_tracing` 递给引擎类），修法见第十三类（Python 版）。
C++ 那条 `PrefillInput::is_decode` 是**聚合初始化里用注释写的字段名**（第二类写入形态）。

六批合计（Python 统计口径，第一个数是开关数、第二个是 `silently_inert`）：
SGLang 386→34、vLLM 263→12、TensorRT-LLM 352→35、LMDeploy 38→5、TRL 6→0、
MLC-LLM 7→2（修复后 0）。C++ 口径：llama.cpp 1636 字段→37、MLC-LLM 22→1。
**真缺陷合计 4 条**（SGLang 2、vLLM 1、TensorRT-LLM 1），其余全是结构原因或工具盲区。
读数归档：[scan-mlc-llm-44fd06d04f.json](scan-mlc-llm-44fd06d04f.json)（修复后）、
[scan-cpp-mlc-llm-44fd06d04f.json](scan-cpp-mlc-llm-44fd06d04f.json)；
修复前那份在 [experiments/mlcllm-pyflag-triage/raw/scan-py.json](../../experiments/mlcllm-pyflag-triage/raw/scan-py.json)。
