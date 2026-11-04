# pyflag — 扫描"配置项有没有生产者"（布尔开关静默失效）

## 用在哪

一个布尔开关在源码里有**消费者**（函数形参 + `if flag:` 分支），却找不到任何
**生产者**（调用点传值），那它在本快照里不可达：用户设了也没用。这类问题不看
代码很难发现——开关名在 config、trainer、engine 里各出现一次，grep 起来"到处都在用"。

工具只做只读静态分析（AST），不导入、不运行目标仓库。

## 怎么用

```bash
python3 scan_unwired_flags.py /path/to/repo          # 打印 JSON 报告
python3 scan_unwired_flags.py /path/to/repo --json   # 同时落盘 scan-<repo>.json
python3 -m unittest test_scan_unwired_flags -v       # 10 个用例，含 4 条易错点
```

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

**跑 SGLang 的耗时**：`python3 scan_unwired_flags.py <sglang>` 约 **10.5 分钟**（8000+ 文件），且 `--json` 落盘到**脚本所在目录** `tools/pyflag/scan-<root>.json`，不是被测仓库里——第一次找错了地方。

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
