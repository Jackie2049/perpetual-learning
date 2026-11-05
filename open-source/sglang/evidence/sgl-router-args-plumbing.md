# SGLang 路由网关：Python 配置层 108 个字段，没有一个被悄悄丢掉

<!-- contribution-summary:start -->
**【仓库】** [sgl-project/sglang](https://github.com/sgl-project/sglang)（推理引擎；本次看的是它内置的 `sgl-model-gateway` 路由网关）

**【贡献点】** 对 `sgl-model-gateway/bindings/python/src/sglang_router/` 做 Python → Rust 的配置通路核对：`RouterArgs` **108 个字段**里 86 个直接进 `Router(**kwargs)`、22 个先被转换或只在 Python 侧用（`mini_lb` 等），**没有一个被静默丢弃**；传给 Rust 的关键字**全部**在 pyo3 签名内，无 `TypeError` 风险。[实验与判据](../../experiments/sglang-router-args-plumbing/README.md)

**【类型】** 问题诊断（B 类：主动发掘，负结论）

**【价值评估】** 2 / 10 — 没有可对外的缺陷。留下的是一条方法：核对"配置是否到达下层"时，**比对的另一端要取"值真正被消费的那段代码"**，而不是构造函数形参表——本批正是靠这一点把 4 条假缺陷收回。

**【进展】** 2026-11-05 完成。`probe.py` 8 条判据全 PASS；快照 `1942c5f0`；纯静态、未编译 Rust、未启动网关、未触上游。

**【大白话讲明白】**

SGLang 的路由网关有两种实现：老的纯 Python 迷你负载均衡器（`mini_lb`），和新的 Rust 核心。启动时 Python 读命令行参数、整理成一个配置对象，再把它交给 Rust ——相当于把一整张表抄了一遍。

这种"两边各有一份字段表"的地方最容易出错：Python 有 108 个字段，如果某个没被抄过去，用户改了它也不会有任何反应，而且不会报错。

这一次逐个数下来：**没有一个漏掉**。86 个原样交给 Rust；剩下的 22 个里，大多数被"翻译"成了 Rust 那边另一种写法（比如把字符串 `"cache_aware"` 翻成枚举值、把一堆 JWT 配置打包成一个结构体），少数几个本来就只有 Python 版迷你负载均衡器才用得上。

有一处差点误判：4 个"分词器缓存"相关的字段在 Rust 构造函数签名里找不到。乍看像是漏了，翻到下面组装配置的那一段才发现——它们是**直接写进配置结构体**的，只是不经过构造函数。**所以检查"有没有漏"，不能只看构造函数的参数表。**

**【What–Why–How】**

- **What：** `RouterArgs`（108 字段）→ `Router.from_args` 的 `fields_to_remove` 名单（22 个）→ `Router(**args_dict)` 与 `to_router_config()`。
- **Why：** 前一轮在 MLC-LLM 上刚记下"恒为真的硬编码开关"，路由网关是同类形状的另一面：两份字段表之间可能漏。
- **How：** 三段文本切片 + pyo3 签名解析（两路互相校验：`#[pyo3(signature)]` 的默认值清单与 `fn new` 的形参逐一相同），逐字段判"有没有去处"。**边界**：只覆盖 Python 启动器 → Rust 构造函数这一段；Rust 内部之后的消费不在范围内。
<!-- contribution-summary:end -->

## 一、三段事实

| 段 | 位置（`1942c5f0`） | 读数 |
|---|---|---|
| 字段来源 | `sglang_router/router_args.py` | `RouterArgs` **108** 个字段 |
| 过滤名单 | `sglang_router/router.py` 的 `fields_to_remove` | **22** 个先被拿掉（转换成 Rust 参数或纯 Python 侧用） |
| Rust 签名 | `bindings/python/src/lib.rs` 的 `#[pyo3(signature = (...))]` | **90** 个形参，与 `fn new` 的形参逐一相同 |

⇒ 传入的 86 个关键字全部命中签名；108 − 86 = 22 个都有去处（19 个在 `to_router_config()`
里被真读，`mini_lb`／`test_external_dp_routing` 只有 Python 侧用）。

## 二、那 4 个"看起来漏了"的字段

`tokenizer_cache_enable_l0` / `l0_max_entries` / `enable_l1` / `l1_max_memory`：
`Router::new` 的签名里**没有**它们，但它们出现在 `to_router_config()` 的
`.tokenizer_cache(config::TokenizerCacheConfig { ... })`（`lib.rs:626-630`）。

⇒ 若用"构造函数形参表"当比对基准，会误报 4 条；用"整个 config 组装段"才是 0。

## 三、为什么不对外

没有用户会被影响——问题不存在。这条的价值是方法：**核对"配置到达下层"时，另一端的取景
范围决定结论**，太窄（只看构造函数）会造假缺陷，太宽（整个文件）又会把注释和测试算进去。

## 核对产物

| 文件 | 结果 |
|---|---|
| [`probe.py`](../../experiments/sglang-router-args-plumbing/probe.py) | **8 条判据全 PASS**（含一条读取侧改名的对照） |
| `raw/` | 四个源文件的逐字节快照（`router_args.py`／`sr_router.py`／`bind_lib.rs`／`test_pyo3.py`） |

复现见 [实验 README](../../experiments/sglang-router-args-plumbing/README.md)。
