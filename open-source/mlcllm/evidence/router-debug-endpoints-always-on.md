# MLC-LLM：`mlc_llm router` 起的子服务器永远带 debug 端点，用户关不掉

<!-- contribution-summary:start -->
**【仓库】** [MLC-LLM](https://github.com/mlc-ai/mlc-llm)（MLC 社区的通用 LLM 部署引擎）

**【贡献点】** `mlc_llm router` 给它拉起的每个子服务器传 `enable_debug=True`（[router.py:82](https://github.com/mlc-ai/mlc-llm/blob/44fd06d0/python/mlc_llm/router/router.py#L82)），而 `Router.__init__` 的形参表与 `cli/router.py` 的 10 个参数里**都没有** debug 开关 ⇒ 用 router 部署时 `/debug/reset_engine`、`/debug/cuda_profiler_*`、`/debug/dump_event_trace`、`/debug/dump_engine_metrics` 恒可用且无法关闭；同仓 `mlc_llm serve` 有这个开关（默认关），两条部署路径不一致。[实验](../../../experiments/mlcllm-router-debug-hardcoded/README.md)

**【类型】** B 类（主动发掘）：四跳真实源码在 CPU 上跑出的行为不一致，**未提交上游**

**【价值评估】** 5 / 10 — 分支活的、下游真分叉、用户关不掉，可证伪且行号可机读；但端点在源码里被标为 debug 用途、默认只绑 `127.0.0.1`，**不能写成安全漏洞**，更准确的说法是"部署路径上少了一个开关"。

**【进展】** 2026-11-05 在 `44fd06d0` 上完成四跳探针（`--selftest` 5 项 / 6 条注入全过）+ 重复 flag 的 argparse 实测（5/5）；上游查重（`enable-debug` 25 条、`PopenServer` 9 条、`router debug endpoint` 3 条）**无重复**。**未起服务、未发 HTTP、未上 GPU**。

**【大白话讲明白】**

`mlc_llm` 有两种跑法：一种是单机直接 `mlc_llm serve` 起一个服务；另一种是 `mlc_llm router`，它作为一个总调度，在每张卡上各拉起一个子服务，再把请求分发给它们。

这些子服务里有一组"调试专用"的接口：看事件轨迹、开关 CUDA 性能分析器、导出内部指标，还有一个会**把引擎整个重置、清掉运行数据**的接口。

开关本身是好的：`serve` 那边确实按它分支，关掉就不挂这些接口、开就挂上。问题在于 `router` 拉起子服务时，把它**写死成"开"**——而 router 自己的命令行里根本没有这个开关。所以只要用 router 部署，这些调试接口就在每个子服务上一直开着，admin 想关也关不掉。

对比一下就更清楚：同一个仓库，`mlc_llm serve` 有 `--enable-debug`（默认关），要不要开是你的事；`mlc_llm router` 没有，等于替你选了"开"。

**这不等于安全漏洞**：子服务默认只绑本机回环地址，要跨机访问得自己指定 `--endpoint-hosts`。本文只说"这条部署路径上少了一个开关，且和隔壁路径不一致"。

**【What–Why–How】**

- **What：** `Router.__init__` 构造 `PopenServer(..., enable_debug=True, ...)` 写死；`Router` 形参无此项、`cli/router.py` 无此选项 ⇒ router 部署路径上 debug 端点恒开。
- **Why：** 下游 `interface/serve.py:122-126` 真的按它分支（挂 `debug_entrypoints.app`），`openai_entrypoints.py:139/140/206/314` 也读它决定是否接受调试配置 ⇒ 两档行为确实不同。历史：`PopenServer` 的默认值在 `0b2cbb2c`（#2417，2024-05-25）从 `False` 改成 `True`，`router.py` 由 `5c9ebcb5`（#3064，2024-12-16）新建时就是 `enable_debug=True`，此后无人改动 ⇒ **不是回归，落地即如此**。
- **How：** 四跳全部取真实函数体在 CPU 上执行（`PopenServer`、`requests`、`fastapi` 等外部依赖用替身，清单进读数），同一闭包里的 `prefix_cache_mode` 作对照项证明探针活着；另用快照里真实的 `mlc_llm.support.argparse` 验证"`--enable-debug` 被追加两次"幂等无害。**边界**：未起服务、未发 HTTP 请求；端点清单是读 `debug_entrypoints.py` 得来的，没有真跑过。
<!-- contribution-summary:end -->

2026-11-05，M1。来源：TM-019（MLC-LLM pyflag 扩池第六批）在做"形参有生产者 ≠ 语义对"复核时的
顺带发现——扫描器把 `enable_debug` 判为正常接线（`producers_total=1`），这一轮顺着那条生产者往下走，
发现它接到的位置是**恒真**的。

> **一眼结论。** `mlc_llm router` 会在每张卡上拉起一个子服务器。那个子服务器带有一组
> **debug 专用端点**（`/debug/dump_event_trace`、`/debug/cuda_profiler_start|stop`、
> `/debug/reset_engine`、`/debug/dump_engine_metrics`）。开关本身是好的、下游也确实按它分支——
> 但 router 在构造子服务器时把它**写死成 `True`**，而 router 自己的命令行里**没有任何 debug 开关**。
> 结果是：只要用 `mlc_llm router` 部署，每个 endpoint 就都开着 debug 端点，没有任何办法关掉。
>
> 对照：同仓 `mlc_llm serve` 的 CLI 有这个开关（默认关），所以"要不要开调试端点"在
> `serve` 路径上是用户的选择，在 `router` 路径上不是。

## 一、四跳读数（每跳都执行真实源码）

`probe_router_debug.py` 不 grep，取真实函数体在 CPU 上执行，替身只放在外部依赖上：

| 跳 | 取的是什么真实源码 | 两档（`enable_prefix_cache=True/False`） |
|---|---|---|
| C1 | `Router.__init__` 全部函数体，含内部 `start_server` 闭包 | 两个档都构造出 `PopenServer(enable_debug=True, ...)` |
| C2 | `PopenServer.start()` 全部函数体（`subprocess.Popen` 换只记 argv 的桩） | `True` 档 argv 含 `--enable-debug`；`False` 档不含 |
| C3 | `interface/serve.serve` 的 `enable_debug` 消费段 | `True` → `server_context.enable_debug=True` 且挂上 `debug` 路由；`False` → 都不发生 |
| CLI | `cli/router.py` 与 `cli/serve.py` 的 `add_argument` 表 | router：**0 个** debug 开关；serve：`--enable-debug` |

**C1 的对照项**：同一个闭包里还有 `prefix_cache_mode="radix" if enable_prefix_cache else "disable"`。
它随档变化（`radix` / `disable`）⇒ 探针确实跑起来了，"`enable_debug` 两档都是 `True`"不是读数死了。

## 二、下游确实按它分叉（所以这不是废开关）

`interface/serve.py:122-126`：

```python
server_context.enable_debug = enable_debug

if enable_debug:
    app.include_router(debug_entrypoints.app)
    logger.info("Enable debug endpoint and debug_config in requests...")
```

而 `server_context.enable_debug` 还被 OpenAI 入口读：`openai_entrypoints.py:139/140/206`
用它决定 `request_final_usage_include_extra` 与 `request_include_debug_config`，
并在请求带了 debug 配置而服务端没开时拒绝（`:206`、`:314`）⇒ **两档行为真不一样**。

## 三、钉在哪

`router/router.py:77-88`：

```python
server = PopenServer(
    model=model,
    model_lib=model_lib,
    host=hosts[i],
    port=ports[i],
    enable_debug=True,        # ← 写死
    device=f"cuda:{self.device_id_starts[i]}",
    mode="server",
    ...
)
```

`Router.__init__` 的形参表里**没有** `enable_debug`；`cli/router.py` 的 10 个参数
（`model`、`--model-lib`、`--router-mode`、`--router-host`、`--router-port`、
`--endpoint-hosts`、`--endpoint-ports`、`--endpoint-num-gpus`、`--enable-prefix-cache`、
`--pd-balance-factor`）里也没有。

## 四、历史归属：和 router 一起进来，不是后来改坏的

| 时间 | 提交 | 事件 |
|---|---|---|
| 2024-05-24 | `cfc05978`（#2414） | `PopenServer` 加 `enable_debug` 形参（默认 `False`），`start()` 拼 `--enable-debug` |
| 2024-05-25 | `0b2cbb2c`（#2417） | 默认改成 `True`；**同一提交里又加了一处 `if self.enable_debug: cmd += ["--enable-debug"]`** —— 于是 `start()` 会追加两次 |
| 2024-12-16 | `5c9ebcb5`（#3064 MicroServing） | 新建 `router/router.py`，`PopenServer(..., enable_debug=True, ...)` 从第一天就是这样 |

⇒ **不是回归**：router 落地时就是钉死的，此后没有一次提交动过这一行。
（`git log -S'                enable_debug=True,'` 在 `router.py` 上只命中 `5c9ebcb5`。）

## 五、顺带量出来的一条：`--enable-debug` 被追加两次

`popen_server.py:70-71` 与 `:114-115` 各有一份，两处都是 `if self.enable_debug:`。
`0b2cbb2c` 引入了第二份、之后一直没删。

**它单独看没有后果**：`probe_duplicate_flag.py` 用快照里**真实的** `mlc_llm.support.argparse`
（只覆盖了 `error` 的 `ArgumentParser` 子类）解析三种 argv——只出现一次 / 出现两次 / 一次都没有：
`store_true` 累加幂等，三次都正常退出、`enable_debug` 分别是 `True/True/False`，
`model` 位置参数也都没被吃掉。⇒ **重复本身不是缺陷，只是多一个 token**；
记下来是为了不让"两处追加"被误读成"会导致报错"。

## 六、影响面（只写能核实的）

**能写**：用 `mlc_llm router` 部署时，**每个** endpoint 子进程都带 `--enable-debug` ⇒
`interface/serve.serve` 会挂 `debug_entrypoints.app` ⇒ 以下端点在这条部署路径上恒可用：

| 端点 | 做什么 |
|---|---|
| `POST /debug/reset_engine` | **重置引擎、清掉全部运行数据与 metrics** |
| `POST /debug/cuda_profiler_start` / `_stop` | 起停 CUDA profiler |
| `POST /debug/dump_event_trace` | 导出 event trace |
| `POST /debug/dump_engine_metrics` | 导出引擎 metrics |

**不能写**：这不是"安全漏洞"。是否构成风险取决于部署边界——endpoint 默认绑
`127.0.0.1`（`PopenServer.host` 默认值），跨机访问要在 `--endpoint-hosts` 里显式指定。
本文只说"这条部署路径上没有开关、且它与同仓 `serve` 路径不一致"。

**边界**：**未启动任何服务、未发任何 HTTP 请求、未上 GPU**。全部是取真实源码文本在 CPU 上执行 +
静态读取 CLI 表；`debug_entrypoints.py` 的端点清单是读源码得来的，没有真跑过。

## 七、不算缺陷的三条（查过，写在这里免得下一轮重复）

1. **`Router` 内部转发 `--enable-debug` 到子进程不是多余的**：子进程是全新的 `python -m mlc_llm serve`，
   它的 `enable_debug` 只能从命令行来。
2. **重复追加无害**（第五节，已用真 argparse 验）。
3. **`chat.py` 的 `PopenServer(model)` 也吃默认 `True`**，但那是 docstring 里的示例用法、
   非默认 CLI 路径，不在本结论的范围内。

## 核对产物

| 文件 | 作用 | 结果 |
|---|---|---|
| [`probe_router_debug.py`](../../experiments/mlcllm-router-debug-hardcoded/probe_router_debug.py) | 四跳读数 + 6 项判定 | `finding_holds: true`，`--selftest` **5 项 / 6 条注入**全过 |
| [`probe_duplicate_flag.py`](../../experiments/mlcllm-router-debug-hardcoded/probe_duplicate_flag.py) | 重复 flag 用真 argparse 解析 | `--selftest` **5/5** |
| `raw/snapshot/` | 被引用的 7 个源文件的逐字节副本 | 与 `44fd06d0` 的 `git status` 干净树逐字节相同 |

**复现**：`python3 probe_router_debug.py raw/snapshot`（不需要克隆，快照自带）。

## 待用户决定

**不对外**。要对外的话，最小动作有两个方向，都需逐项授权：

- **(a) 只改注释/文档**：在 `router/router.py` 那行 `enable_debug=True` 上写清"router 路径恒开"，
  成本最低、零行为变化。
- **(b) 把开关接出来**：`Router.__init__` 加 `enable_debug: bool = True`（保默认不变）、
  `cli/router.py` 加 `--enable-debug/--no-enable-debug`、`interface/router.py` 转发。
  这会让默认部署的行为**不变**，但用户可以关——需要维护者确认"router 是否需要始终开着 debug"。
- **(c) 顺手删掉重复那次追加**：纯清理，无行为变化。

按 TM 渠道属 **B 类（主动发掘）**。(a)+(c) 风险最低，(b) 更有用但要动 4 个文件的签名。
