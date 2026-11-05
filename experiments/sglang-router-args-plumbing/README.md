# SGLang 路由网关的 Python 启动器：108 个配置字段，**没有一个被悄悄丢掉**

对 [sgl-project/sglang](https://github.com/sgl-project/sglang) `1942c5f0`（2026-11-04）里
`sgl-model-gateway/bindings/python/src/sglang_router/` 做一次静态通路核对。

动机是接上 [TM-021](../mlcllm-router-debug-hardcoded/README.md) 那类形状：**"用户配了但没到
Rust 侧"**。路由网关有 Python 配置层（`RouterArgs`，108 个字段）和 Rust 核心（`Router::new`，
90 个形参），中间靠 `Router.from_args` 逐个转换——这类"两层各有一份字段表"的地方最容易漏。

- 证据页：[open-source/sglang/evidence/sgl-router-args-plumbing.md](../../open-source/sglang/evidence/sgl-router-args-plumbing.md)
- 上游状态：**不对外**（没有缺陷）

## 结论

| 问题 | 答案 |
|---|---|
| `RouterArgs` 有字段被静默丢掉吗？ | **没有**。108 个字段的去处全可数：86 个直接进 `Router(**kwargs)`，22 个先被转成 Rust 参数（policy/backend/TLS/JWT/Oracle/Redis/Postgres…）、或本就只在 Python 侧用（`mini_lb`、`test_external_dp_routing`） |
| 有 `TypeError` 风险（传了 Rust 不认的关键字）吗？ | **没有**。传给 `Router(...)` 的 86 个关键字**全部**在 pyo3 签名的 90 个形参里 |
| 只由 Rust 默认值提供、Python 侧没读过的字段？ | **0 个** |
| 那 4 个 `tokenizer_cache_*` 呢？ | `Router::new` 的**签名里没有**它们，但 `to_router_config()` 里真读了（`:626-630`）⇒ 用"构造函数形参表"比对会误报 4 条，用"整个 config 组装段"比对才是 0 |

⇒ **这是一次负结论**：分层设计在这里是完整的。留给下一批的是方法：
**比对的另一端要取"值真正被消费的那段代码"，不是"构造函数的形参表"**。

## 读数

| 项 | 值 |
|---|---|
| 快照 | `1942c5f0950b9030c61d4c6a670f8ccd02f9aeaa`（2026-11-04 22:17 -0500） |
| `RouterArgs` 字段 | **108** |
| 进 `Router(**kwargs)` | **86** |
| 先转换 / 只在 Python 侧用 | **22**（其中 19 个在 `to_router_config()` 里另有去处，3 个是本侧专用） |
| Rust `Router::new` 形参 | **90**（pyo3 signature 默认值 90 个，与 `fn new` 的形参**逐一相同**） |
| 传给 Rust 但 Rust 不认的关键字 | **0** |
| 判据 | `probe.py` **8 条全 PASS**（含一条"读不到就有去处"的对照） |

## 本批的"看起来像结论"两次（都被对照抓出）

1. **`start()` 切片切错**：`to_router_config()` 与 `start()` 是紧邻的两个方法，按"遇到下一个
   `    }` 收尾"切片会把整个 config 组装段漏掉 ⇒ 读数凭空多出 22 个"无去处"字段。
   改成"从 `pub fn to_router_config` 到 `server::startup(`"再切，才拿到真实段。
2. **对照臂自己写错**：第一版对照想模拟"字段没被声明"，但只把它们从 `fields` 里删掉、
   忘了同步从"移除清单"和 `passed` 里删，四条字段仍被当成"有人传"，判据永远不报红。
   最终对照改成**只改读取侧**（把这四处 `self.tokenizer_cache_*` 改名），结论清晰：
   判据能分辨"有去处"与"没去处"。

## 脚本

| 文件 | 作用 | 怎么跑 |
|---|---|---|
| `probe.py` | 8 条判据（字段计数／签名两份清单一致／无多余关键字／组装段可达性／对照） | `python3 probe.py` |
| `raw/router_args.py` | `RouterArgs` 与其 argparse 定义（逐字节快照） | — |
| `raw/sr_router.py` | Python 侧 `Router.from_args`（`fields_to_remove` 名单在这里） | — |
| `raw/bind_lib.rs` | Rust pyo3 绑定（`#[pyo3(signature)]` + `to_router_config`） | — |
| `raw/test_pyo3.py` | 上游自带的绑定测试（`pool_*`／`tcp_keepalive`／`enable_wasm` 的 getter 断言） | — |

## 边界

- **github.com:443 本轮不可达**（`curl 28`，连续重试均失败），文件是走 `api.github.com` 的
  contents API 逐个取的；因此**没有整仓 AST 扫描**，只核这 4 个文件。
- 纯静态阅读，**未编译 Rust、未启动网关、未发请求、未触上游**。
- "没有静默丢弃"只覆盖 Python 启动器 → Rust 构造函数这一段；Rust 内部把字段落到
  `RouterConfig` 之后的消费路径不在本批范围。
