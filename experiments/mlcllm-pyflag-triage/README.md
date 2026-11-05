# MLC-LLM 扩池：7 个 Python 开关 + 22 个 C++ 字段，**没有一条真缺陷**

对 [MLC-LLM](https://github.com/mlc-ai/mlc-llm) `44fd06d0`（2026-10-30）跑 pyflag 的 Python 版和 C++ 版，
两条候选逐条查证后**真缺陷 0 条**。本批真正产出的是**扫描器自己被同一种形状骗了两次**，
以及把那个根因修进工具（工厂函数被判成消费者、容器键位写入既不算生产者也不算消费）。

- 实验记录与读数：本文件
- 证据页：[open-source/mlcllm/evidence/pyflag-2026-11.md](../../open-source/mlcllm/evidence/pyflag-2026-11.md)
- 工具改动：[tools/pyflag/readme.md](../../tools/pyflag/readme.md) 第十三类（Python 版）

## 读数

| 项 | 值 |
|---|---|
| 快照 | `44fd06d04f22c87820e205b0f2b2e5e6a1f6ca85`（2026-10-30 01:57 -0700，[#3588](https://github.com/mlc-ai/mlc-llm/pull/3588)） |
| Python 文件 | 415（扫描 4.0 秒） |
| Python 开关 / `silently_inert` | **7 / 2（修复前）→ 7 / 0（修复后）** |
| C/C++ 文件 | 88（扫描 0.5 秒） |
| C++ `unwritten` 字段 | **1**（`PrefillInput::is_decode`，接通） |
| 真缺陷 | **0** |

初测那份扫描输出**原样保留**在 `raw/scan-py.json`（修复前），修复后的在
`raw/scan-py-fixed.json`；两者差异本身就是本轮的一条读数。

## 逐条归因（Python 7 条）

| 开关 | 扫描器（修复前） | 实况 | 档位 |
|---|---|---|---|
| `enable_prefix_cache` | `mixed` / `silently_inert=True` / 生产者 0 | CLI → `interface.router.serve` → `Router` → `PopenServer(EngineConfig(prefix_cache_mode=...))` **四跳齐全** | R1 真实接线 |
| `enable_tracing` | `mixed` / `silently_inert=True` / 生产者 0 | CLI `--enable-tracing` → `interface.serve` → `MLCEngine`/`AsyncMLCEngine` → `super()` → `EngineState(if enable_tracing: EventTraceRecorder())` | R1 真实接线 |
| `enable_debug` | 无 | `interface.serve` → `MLCEngine(enable_debug=...)` | R1 |
| `enable_disaggregation` | 无 | 38 个模型建 KV cache，**只有 3 个**（llama / llama4 / nemotron）显式传，其余 35 个吃默认 `False` | R4 编译期按模型接 |
| `disable_instrument` | 无 | `testing/debug_chat.py` 的 CLI 辅助，生产者是 `parsed.disable_instrument` | R5 内部辅助 |
| `no_debug_config` | 无 | `bench/api_endpoint.py` 内部，生产者是 `no_debug_config=True` | R5 |
| `with_scale` | 无 | `gemma4_audio` 内部用例 | R5 |

档位全覆盖（`classify.py` 有候选没档位就 `exit 1`）：R1 3 / R2 0 / R3 0 / R4 1 / R5 3。

## 两条候选都是**工厂转发**：扫描器把"透传"读成了"决策"

`python/mlc_llm/interface/router.py:17` 的 `serve()` 函数体里**没有** `if enable_prefix_cache:`，
可扫描器把它算成消费者——因为这个名字出现在
`Router(..., enable_prefix_cache=enable_prefix_cache, ...)` 的**实参位置**。
`serve()` 是工厂：它只是把 CLI 的值原样递给 `Router`，决策在 `Router.__init__:86`。

`probe_router_chain.py` 把三处真实源码在 CPU 上跑起来（真 `Router.__init__`、真
`interface.router.serve`、真 argparse 参数表，第三方模块用替身，替身清单进读数）：

| 镜头 | 读 True | 读 False |
|---|---|---|
| C1 真 `Router.__init__` → 每个 `PopenServer` 的 `prefix_cache_mode` | `radix, radix` | `disable, disable` |
| C2 真 `interface.router.serve` → `Router(...)` 收到的关键字 | `True` | `False` |
| C3 真 argparse | `--enable-prefix-cache` → `True` | 缺省 → `False` |

`verdict: WIRED`，exit 0；上游哪天拆掉这一跳则 exit 2（探针自己报警）。

## 本批最要紧的一处自错：探针第一版给了**反向读数**

第一版探针的基线就错了：真 `interface.router.serve` 跑出来
`router_kwargs_enable_prefix_cache = <absent>`，我一度准备把它写成一条真缺陷。

**错在替身，不在源码。** 为了让 `serve()` 跑完，第一版把 `uvicorn` 换成了空模块，而 `serve()`
最后一行是 `uvicorn.run(app, ...)` ⇒ `AttributeError`，异常在 `Router` 已经构造完之后、
`serve()` 返回之前抛出：那一轮里的 `Router(...)` 其实**带了关键字**，但异常让 `serve()` 返回不了，
探针把失败轮记成 `<absent>`。**错在"把异常轮当成正常轮"，不在源码。** 补上
`uvicorn.run = lambda *a, **k: None` 后同一份源码读出 `True / False`。

`--selftest` 的红灯救回来了：反事实注入该让读数翻转，读数却没动——因为基线本来就是
`<absent>`，"再删一次"当然还是 `<absent>`，注入空转。现在注入按 AST 找 `Router(...)` 这次调用
再删实参（`drop_router_kwarg`），并加了"注入必须真的改动源码"的前置检查。
⇒ **这和 TM-017 那次 `sed` 事故是同一类**：判据绿不代表读的是原文件；这里更狠，
**判据红也不代表源码有问题，先怀疑工具**。

## 扫描器的根因（已修，工具 + 单测 + 回归）

1. **名字作为某个调用的关键字实参**出现也算消费。原来只认 `if` 测试里的名字，
   于是 `return Cls(f=flag)` 这类"在工厂里选"的写法整段看不见。
2. **只转发的形参不构成决策**。形参只被"传给下游、从不被 `if` / 赋值 / 属性访问读到"的消费者
   标 `param_default='forwarded'`，落进新的 `deciding_consumers` 字段；
   `silently_inert` 只由**决策者**决定。
3. **容器键位写入**（`kwargs["enable_disaggregation"] = True`）单独出 `container_writes` 字段，
   **只作线索、不进 `producers_total`**——字典里写了键不等于下游真读到，所以"写了没人读"仍然成立。

回归：verl `6879e1ab07`（8 条）、SkyRL（7 条）、OpenRLHF（0 条）三份既有归档的
`silently_inert` **逐条相同**（37/39/5 个开关数量不变）；`test_scan_unwired_flags.py`
**16 passed**（新增 3 条：工厂外壳不算决策者、真在 `if` 里用的仍算、容器写入是线索不是生产者）。

## C++ 侧：22 个字段、1 个 `unwritten`，是接通的

`cpp/serve/engine_actions/batch_prefill_base.h:26` 的 `PrefillInput::is_decode` 默认 `false`，
扫描器没找到写入点。实际写入是**聚合初始化的第四项**：
`batch_prefill_base.cc:238` 在 hybrid 预填模式下把 decode 的请求也塞进 prefill 批，
写成 `{rsentry, num_tokens_for_next_decode, 0, /*is_decode=*/true}`——`/*is_decode=*/` 是注释，
剥掉后整行只剩一个 `true`，正是 TM-018 记的**第二类 C++ 写入形态**。三处消费者
（`new_request_prefill.cc:107`、`disagg_prepare_recv.cc:113`、`disagg_remote_send.cc:115`）都真读它。

C++ 版只有 22 个"有消费者且按它分支"的结构体字段——MLC-LLM 的 C++ 是服务层
（engine / request_state / engine_actions），配置面在 Python 与 `mlc-chat-config.json`，
**C++ 版扫它的信息量本来就低**，这一点先记下来，别让下一批拿"0 条"当方法失效。

## 脚本

| 文件 | 作用 | 怎么跑 |
|---|---|---|
| `classify.py` | 7 条逐条归档位 + 理由；有候选没档位 `exit 1` | `python3 classify.py raw/scan-py.json --json raw/classify.json` |
| `verify_findings.py` | 21 checks 钉到具体文件行（含快照/干净树前置 `S0`）；`--selftest` 4 条注入各让目标判据变红 | `python3 verify_findings.py <root>` / `--selftest <root>` |
| `probe_router_chain.py` | 三处真实源码在 CPU 上跑 `enable_prefix_cache` 的四跳；`WIRED` → exit 0，拆掉 → exit 2 | `python3 probe_router_chain.py <root>` / `--selftest <root>` |
| `probe_container_write.py` | 编译期 `enable_disaggregation`：3 传 / 35 省；dispatch 里的 `kwargs` 是具名形参 | `python3 probe_container_write.py <root>` / `--selftest <root>` |

## 快照副本

`raw/snapshot/` 存三份原样副本（`router/router.py`、`interface/router.py`、`cli/router.py`，
共 24 KB），`probe_router_chain.py` 优先读它们 ⇒ **只有本仓库的机器也能跑 `--selftest`**
（8/8），不依赖 `/tmp` 里的克隆。完整快照仍需要克隆才能跑 `verify_findings.py`
（它要读 21 个判定点，覆盖模型目录与 C++ 侧）。

## 复现

```bash
git clone --depth 1 https://github.com/mlc-ai/mlc-llm /tmp/mlc-scan
cd <repo>
python3 tools/pyflag/scan_unwired_flags.py /tmp/mlc-scan --json /tmp/scan-mlc.json   # 4 秒
python3 tools/pyflag/scan_cpp_flags.py    /tmp/mlc-scan --json /tmp/scan-mlc-cc.json # 0.5 秒
python3 experiments/mlcllm-pyflag-triage/verify_findings.py /tmp/mlc-scan
python3 experiments/mlcllm-pyflag-triage/probe_router_chain.py /tmp/mlc-scan
```

## 边界

纯静态分析 + 在 Python 里执行抽出的源码片段。**未安装 mlc_llm、未建 TVM、未启动任何服务器、
未上 GPU/NPU、未触上游**。上游查重：`enable_prefix_cache` / `create_generic` /
`disaggregation kv_cache` 三组关键词在 mlc-llm issue 里**均无结果**（`gh search issues`，2026-11-05）——
不是"有人报过我们复现"，而是本来就没有人报过这条。
