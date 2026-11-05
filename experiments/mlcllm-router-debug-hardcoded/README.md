# MLC-LLM `mlc_llm router` 的子服务器恒开 debug 端点（TM-021）

在 [MLC-LLM](https://github.com/mlc-ai/mlc-llm) `44fd06d0` 上，`mlc_llm router` 拉起子服务器时把
`enable_debug=True` 写死，而 router 自己没有任何 debug 开关 ⇒ 这条部署路径上 debug 端点恒开、
用户关不掉；同仓 `mlc_llm serve` 有这个开关（默认关）。

- 证据页：[open-source/mlcllm/evidence/router-debug-endpoints-always-on.md](../../open-source/mlcllm/evidence/router-debug-endpoints-always-on.md)
- 上游状态：**未提交**，待用户逐项授权（查重见下）

## 读数

| 项 | 值 |
|---|---|
| 快照 | `44fd06d04f22c87820e205b0f2b2e5e6a1f6ca85`（2026-10-30，`git status` 干净） |
| C1（`Router.__init__` → `PopenServer` 实参） | 两档 `enable_prefix_cache` 下 `enable_debug` **都是 `True`** |
| C1 对照项（同一闭包的 `prefix_cache_mode`） | `radix` / `disable` **随档变化** ⇒ 读数不是死的 |
| C2（`PopenServer.start` → argv） | `True` 档含 `--enable-debug`（**2 次**）；`False` 档不含 |
| C3（`interface/serve` 的消费段） | `True` → `server_context.enable_debug=True` + 挂 `debug` 路由；`False` → 都不发生 |
| CLI 表 | `cli/router.py` **0 个** debug 开关；`cli/serve.py` 有 `--enable-debug` |
| 重复追加是否有害 | **无害**：真 `ArgumentParser` 下 `once` / `twice` / `zero` 三形状都正常，`twice` 与 `once` 解析结果相同 |
| `finding_holds` | **true** |

## 脚本

| 文件 | 作用 | 怎么跑 |
|---|---|---|
| `probe_router_debug.py` | 四跳读数 + 6 项判定；`--selftest` 6 条注入 | `python3 probe_router_debug.py raw/snapshot` |
| `probe_duplicate_flag.py` | 用快照里**真实的** `mlc_llm.support.argparse` 解析重复 flag | `python3 probe_duplicate_flag.py raw/snapshot` |

`raw/snapshot/` 是 7 个被引用源文件的**逐字节副本**（`router/router.py`、
`serve/server/popen_server.py`、`interface/{router,serve}.py`、`cli/{router,serve}.py`、
`support/argparse.py`），探针只从它取源码 ⇒ 没有克隆也能跑、也不会被上游改动带偏。

## 复现

```bash
git clone https://github.com/mlc-ai/mlc-llm /tmp/mlc-scan && cd /tmp/mlc-scan && git checkout 44fd06d0
cd <repo>/experiments/mlcllm-router-debug-hardcoded
python3 probe_router_debug.py /tmp/mlc-scan            # 或 raw/snapshot（无需克隆）
python3 probe_router_debug.py --selftest raw/snapshot  # 5 项 / 6 条注入
python3 probe_duplicate_flag.py --selftest raw/snapshot
```

## 判定项与它们的含义

| 判定键 | true 表示 | 若变 false 说明 |
|---|---|---|
| `router_pins_enable_debug_true` | router 两档都传 `True` | 上游把开关接出来了 |
| `control_prefix_cache_mode_live` | 对照项随档变化 | **读数死了**，其余判定全部无效 |
| `subprocess_gets_flag` | `True` 档有 flag、`False` 档没有 | 中间某跳断了 / 探针没跑 |
| `duplicate_enable_debug_append` | `start()` 追加了两次 | 重复那次被删了 |
| `downstream_really_forks` | 下游真按它挂路由 | 分支被拆或改成恒真 |
| `router_cli_can_turn_it_off` | CLI 有开关**且**值真转发进 Router **且** Router 内不是字面量 | 用户有办法关 |
| `finding_holds` | 上述（除 `router_cli_can_turn_it_off`、`duplicate_*`）全成立且用户关不掉 | 结论失效 |

## 本批 5 处自错（都由实测抓到，判据已写进代码注释）

1. **连着 `def start(...)` 签名拼函数体** ⇒ 新函数里嵌了内层 `def`，外层函数体是空的，读数恒空。
2. **`class _Subprocess: Popen = _Popen`** —— 类体赋值在类作用域求值，看不到外层函数的局部名。
3. **把函数体截到 `self._proc = subprocess.Popen(` 之前** —— argv 恰恰只在那里拼，截掉后
   读数成了"两支都没有 `--enable-debug`"，**看着像结论成立**。现在整段跑、只桩 `Popen`/`requests`。
4. **`Popen.poll()` 返回 `0`** —— `start()` 判 `is not None` 就当成"子进程已退出"，当场抛
   `The server fails to launch`。要返回 `None`。
5. **`router_cli_can_turn_it_off` 第一版只看 CLI 有没有 debug 开关** —— 自检第 4 条注入
   （只加 `--no-enable-debug`、不转发）当场暴露：加了参数却不接进 `Router(...)`，
   行为上等于没有。改为"CLI 有开关 ∧ 真转发进 `Router` ∧ Router 内不是字面量"。
   第 6 条注入（真接通）随后通过。

> 第 3 条是这批最要紧的：**它给出的是"看起来对"的假读数**。若没有"抓不到 argv 就抛异常"
> 这条守卫，探针会安静地报一个不存在的结论。

## 查重（2026-11-05，只读 GitHub API）

| 关键词 | 命中 | 结论 |
|---|---|---|
| `"enable-debug"` | 25 | 无一条说 router 路径关不掉 |
| `PopenServer` | 9 | 全是历史修复/重构，无本问题 |
| `router debug endpoint` | 3 | 无关（metrics 可解析、embedding phase 2、跨引擎编排提问） |

## 边界

**未启动任何服务、未发任何 HTTP 请求、未装 mlc_llm、未建 TVM、未上 GPU/NPU、未触上游。**
全部是"取真实源码文本在 CPU 上执行" + 静态读 CLI 参数表。`debug_entrypoints.py` 的端点清单
来自读源码，没有真跑过；端点在源码里被标为 debug 用途、`PopenServer.host` 默认 `127.0.0.1`，
**本文不主张它是安全漏洞**。
