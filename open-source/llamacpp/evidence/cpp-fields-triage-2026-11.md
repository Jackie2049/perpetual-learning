# llama.cpp 的布尔字段：37 条"没人写"里没有一条是缺陷

<!-- contribution-summary:start -->
**【仓库】** [llama.cpp](https://github.com/ggml-org/llama.cpp)（ggml 生态的 C/C++ 推理引擎，本地部署与量化的事实标准）

**【贡献点】** 把 pyflag 从 Python 扩到 C++，在 `3de52ac3` 快照上扫出 37 个"有分支、无人写"的结构体布尔字段，逐条归因后**真缺陷 0 条**；真正的产出是三类 C++ 特有的写入形态（单行 `if` 里赋值、`/* name */ value` 聚合初始化、位置实参），以及唯一被怀疑过的 `reasoning_control` 经复核确认**已接线**，见 [实验与读数](../../../experiments/llamacpp-pyflag-triage/README.md)。上游 issue 未提、PR 未提，**待用户授权**。

**【类型】** 问题诊断 / 方法沉淀

**【价值评估】** 3 / 10 — 证据链完整且可复现（脚本 15 checks + 注入全红），但结论是负的：没有可对外的缺陷，价值全在"扫描器该怎么写"这一层。

**【进展】** 2026-11-05 完成扫描与逐条归因，纯静态 + CPU 源码执行（未编译 llama.cpp、未起服务、未跑推理）。`reasoning_control` 的首版判定是**错的**（把守卫里的另一项当成了主因），已在同批推翻重写，见第五节。

**【大白话讲明白】**

llama.cpp 是很多人本地跑大模型的引擎，它的配置项里有一大批是"开关"性质的结构体字段。有一种很容易漏掉的 bug：某个开关有声明、代码里也真的按它分支，但整个代码库里没有任何一行给它赋过值——那这个分支就永远走不到，用户设了也没用。我们在 Python 框架里用 pyflag 抓过这类问题，这次换到 C++ 来试。

llama.cpp 的 Python 代码只有 250 个文件，而且是转换脚本为主，直接扫几乎没什么可看（7 个开关、0 条失效）；真正要扫的是 1574 个 C/C++ 文件。为 C++ 重写扫描器后，先报出 71 条，两轮修正后是 37 条。逐条看过以后，**没有一条是"开关坏了"**：多数是某 ggml 后端的内核参数（由设备探测填写）、测试夹具，或者单纯的局部变量同名。剩下 20 来条全部能指到真实写入点，只是那些写法是 C++ 特有的、按字段名做正则的扫描器看不见。

**【What–Why–How】**

- **What：** 对象是 llama.cpp `3de52ac3`（2026-11-04）；问题是有消费者、按它分支、但全仓无写入的结构体布尔字段。
- **Why：** 这类字段的危害是"配置看起来生效、实际不影响行为"，且不会报错。Python 版 pyflag 已覆盖四批框架（合计 64 条候选→4 条真缺陷），C++ 的字段形态不同，需要单独的方法。
- **How：** 新写 `tools/pyflag/scan_cpp_flags.py`（正则级结构分析 + 7 条夹具自检），扫完后逐条 grep 复核，把每条分到明确档位；判定脚本 15 checks，4 条注入全部如期变红。**边界**：正则级分析不做符号解析，同名不同结构体会串，报"没人写"仍需人工确认——本批 37 条就是这么逐条看掉的。
<!-- contribution-summary:end -->

## 一、为什么先扫 Python 会得到空结果

`scan_unwired_flags.py` 扫 `3de52ac3` 的 250 个 `.py`：**7 个开关、0 条 `silently_inert`**。这些文件是 `conversion/`（115 个，模型格式转换）、`gguf-py/`、`scripts/` 与 server 测试，本来就不是"性能开关"的分布形态。

结论很直接：**对 llama.cpp 用 Python 版 pyflag 没有信息量**，要看 C++ 面。这也意味着"已扫过五个框架"里，llama.cpp 需要按 C++ 重新计数。

## 二、C++ 版的判定口径

C++ 里这类开关的主要形态不是函数形参，而是**结构体成员默认值**：

```c++
struct c2w_state_slot {          // tools/mtmd/models/models.h:490-494
    std::string name;
    int64_t     ne0;
    int64_t     ne1;
    bool        in_place = false;   // ← 声明
};
...
if (slot.in_place) { ... }          // ← 消费（clip.cpp:5903 / 7860、drama3-gen.cpp:102）
```

所以三步是：**字段声明**（`struct X { bool f = false; }`）→ **消费者**（出现在 `if`/三目/`&&`/`||` 里）→ **生产者**（被赋值、被传成实参、聚合初始化）。三者都为 0 的才进候选。

扫描口径的两个硬约束：注释与字符串先剥掉（行号不变），声明行本身不算出现。前者是 Python 版第七类易错点的直接移植。

## 三、读数

| 项 | 值 |
|---|---|
| 快照 | `3de52ac38818978d6da8f386eda43a856e53714a`（2026-11-04 22:34 UTC，`ci: run self hosted cpu jobs…` #31439） |
| C/C++ 文件 | 1574（`.h/.hpp/.hh/.cpp/.cc/.cxx/.c/.cu/.cuh/.m/.mm`，排除 `.git`） |
| 有消费者且按它分支的字段 | 1636 |
| 判为 `unwritten`（有分支、无写入/传参） | **37** |
| 逐条归因后**真缺陷** | **0** |

37 条的档位分布：

| 档位 | 条数 | 说明 |
|---|---|---|
| ggml 后端 / gen-kernel 参数结构体 | 15 | Vulkan / OpenCL / Hexagon / WebGPU / SYCL 的内核参数，由设备能力探测或代码生成器填写 |
| 测试夹具 | 4 | 只在 `tests/` 下使用 |
| 局部变量同名 | 4 | `blocks_found_in_stage` / `is_dir` / `paren_depth` / `a_layer`，是函数内变量而扫描器按字段名匹配 |
| 构造函数初始化列表 | 2 | `_dotall(dotall)`、`enabled_(enabled)` |
| `++` / `--` 写入 | 2 | `req_count`、`n_attempts` —— 只认 `=` 赋值就会漏 |
| `/*.name =*/` 设计化初始化 | 2 | `shares_model`、`topk_blocks`，注释名带点号 |
| 位置实参聚合初始化 | 1 | `in_place` |
| JSON 解析 | 2 | `defer_loading`（chat.cpp:673-684）、`is_inference` |
| 第三方 vendored | 1 | cpp-httplib |
| `REMOVED_BY_DESIGN` | 1 | `filter_allowed_keys`——配套白名单在 #24739 被有意移除，上游 #27929 正在讨论是否恢复 |
| `DEAD_BRANCH_UPSTREAM` | 1 | `show_key_map_stats`——唯一写入机会是构造函数，而构造函数不收它 |
| `WIRED` | 1 | `reasoning_control`——见第五节 |

逐条表：`raw/classify.json`。

## 四、三类 C++ 特有的写入形态（本批真正的方法产出）

这三条都是"扫描器报没人写、实际写了"的真反例，已各自写成判据（`verify_findings.py` 的 B1/B2/B3），删掉对应代码行就必须变红。

**B1 单行 `if` 里赋值。** `common/arg.cpp:3862`：

```c++
/**/ if (value == "jsonl") { params.batched_bench_output_jsonl = true; }
else if (value == "md") { params.batched_bench_output_jsonl = false; }
```

这一行既是条件行又是写入行。第一版实现先判"这行在 `if` 里 ⇒ 只是读"，于是整个字段被记成没人写。**这是本批第一个自错**：命中 71 条候选里的 8 条（含 `batched_bench_output_jsonl` 这类明显活着的字段）。修法是把"本字段在这行有没有被写"提到消费者判定之前。

**B2 `/* name */ value` 聚合初始化。** `tools/server/server-context.cpp:1729-1736`：

```c++
chat_params = {
    /* allow_image           */ mtmd_cap.vision,
    /* allow_audio           */ mtmd_cap.audio,
    /* allow_video           */ mtmd_cap.video,
    ...
```

剥掉注释后整行只剩 `mtmd_cap.vision,`，字段名根本不出现。**这是本批第二个自错**：修法是回看原始文本（未剥注释）判断注释里是否恰好写了这个字段名。

**B3 位置实参。** `tools/mtmd/models/drama3-gen.cpp:12`：

```c++
slots.push_back({"tower_pos", 1, 1, true});   // name, ne0, ne1, in_place
```

同一个结构体在 `qwen3tts-gen.cpp:624` 只传三个（`{"tfm_pos", 1, 1}`），默认 `false` —— 所以**必须两个文件都看**才知道这个字段确实有人写。这条没有通用修法（位置语义要类型信息），只作为"报没人写要人工复核"的具体例证留下。

## 五、唯一被怀疑过的字段：`reasoning_control` 是接通的

这条值得单独写，因为**我第一版判错了**。

`reasoning_control` 是 server 的一个请求字段（#23971 引入），文档说它"为此 completion 开启实时推理中断"。扫描器报它 `unwritten`，理由是它唯一的写入看起来是 `server-common.cpp:1530` 的 `llama_params["reasoning_control"] = json_value(...)`——写的是 JSON，不是字段。

第一版探针把"写/不写这个字段"当成 `common_sampler_init` 那条守卫的唯一变量，读到 `True` 档建采样器、`False` 档不建，于是判定"该字段影响不了采样器"。**这是本批第三个自错，而且是最严重的一个**：守卫里还有 `!params.reasoning_budget_end.empty()` 这一项，而 `reasoning_budget_end` 的形状取决于模型模板有没有 thinking 标记。第一版把"有结束标记"和"没有"混在同一个变量里，读数正好反过来。

重写后的探针（`probe_reasoning_control.py`）把每个量显式列出来跑网格，7 条判据全过：

| 配置 | 建采样器 | 说明 |
|---|---|---|
| `control=true`，无预算（server 默认） | **是** | 该字段确实生效 |
| `control=false`，无预算 | 否 | 两档确有差别 |
| `control=true`，`budget=256` | 是 | 对照：预算本身就能建 |
| `control=false`，lazy grammar | 是 | 对照：lazy grammar 也能建 |
| `control=true`，模板无 thinking 结束标记 | 否 | 对照：守卫真的看那两个向量 |

链路是完整的：`server-schema.cpp:351` 用 `field_bool(name, bool &)` 绑定引用 → `field_bool::eval`（`:636-647`）在请求带该键时 `val = data.at(n).get<bool>()` → `sampling.cpp:315` 的守卫读它 → `rbudget` 建起来 → `server-context.cpp:3123` 调 `common_sampler_reasoning_budget_force()`。

**扫描器漏判的原因**：`field_bool` 的构造接的是 `bool &`，写入发生在另一个函数里通过引用完成，源码里出现的是 `params.sampling.reasoning_control` 而不是 `x.field = `。这是 Python 版第八类（`**kwargs` 兜底）在 C++ 里的同族问题：**签名里没有 ≠ 收不到**。

## 六、本批自错清单

| # | 自错 | 怎么被抓到 | 现状 |
|---|---|---|---|
| 1 | 先判消费者、后判赋值，单行 `if` 里的写被吞 | 71 条里 `batched_bench_output_jsonl` 明明有赋值却报没人写 | 已修，加夹具（selftest 第 6 条） |
| 2 | 剥注释后 `/* name */ value` 整行不含字段名 | 同上，`allow_image` 等 3 个能力位 | 已修，加夹具（第 7 条） |
| 3 | `reasoning_control` 结论方向反了 | 把守卫的每个量拆开跑网格，`end` 非空时 True 档是建的 | 探针重写，7 判据 |

另有两处纯实现层面的坑：`guard_to_python` 先替换 `!` 再替换 `.empty()` 会生成 `not x.not`；`_render` 用 C++ 花括号层级缩排时把 `return;` 的层级算错。两者都由脚本当场报错拦下，没有产出错误读数。

## 七、复现

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama-scan
python3 tools/pyflag/scan_cpp_flags.py --selftest              # 7/7
python3 tools/pyflag/scan_cpp_flags.py /tmp/llama-scan \
        --json experiments/llamacpp-pyflag-triage/raw/scan-cpp-llama.json   # 约 4 分钟
python3 experiments/llamacpp-pyflag-triage/classify.py /tmp/llama-scan     # 37 条全归因，exit 0
python3 experiments/llamacpp-pyflag-triage/verify_findings.py /tmp/llama-scan   # 15 checks 0 failed
python3 experiments/llamacpp-pyflag-triage/verify_findings.py --selftest /tmp/llama-scan
python3 experiments/llamacpp-pyflag-triage/probe_reasoning_control.py --selftest
python3 experiments/llamacpp-pyflag-triage/probe_reasoning_control.py /tmp/llama-scan
```

**验证边界**：全部是静态分析与 Python 里对抽出的源码片段执行；**没有编译 llama.cpp、没有起 server、没有跑推理、没有上 GPU**。`probe_reasoning_control.py` 的守卫是把它转成 Python 表达式求值的，转译规则写在 `guard_to_python()` 里，四类替换逐条可见。
