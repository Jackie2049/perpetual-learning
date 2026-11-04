# LMDeploy：5 条「静默失效」开关逐条查证，没有一条能写成缺陷（2026-11-05）

<!-- contribution-summary:start -->
**【仓库】** [LMDeploy](https://github.com/InternLM/lmdeploy)（InternLM 的推理部署工具链，TurboMind + PyTorch 双后端）

**【贡献点】** pyflag 在 `3228395546df`（2026-11-03）上扫出 **38 个开关 / 5 条 `silently_inert`**。逐条绑行号后**没有一条构成用户可见的缺陷**：1 条是 `**kwargs` 转发误判（链路实测是通的）、1 条是 import 期的硬件探测全局变量不是开关、2 条是仓库内/测试内的内部形参、1 条是 `setup.py` 的 docstring 与默认值不一致（无用户面）。[实验与复现](../../experiments/lmdeploy-pyflag-triage/README.md)

**【类型】** B 类（主动发掘）的**负结论**：按方法学收口这批候选，**不建议对上游提任何 issue**

**【价值评估】** 2 / 10 — 对外价值；唯一像缺陷的 `disable_fastapi_docs` 经 CPU 执行真实源码证明**两条路由都通**；唯一真实的文档问题是 `setup.py` 里一行 docstring。**本批的产出是方法读数而不是贡献候选。**

**【进展】** 2026-11-05 完成扫描 + 逐条取证（`verify_findings.py` 39/39）+ 通路正向对照（`probe_kwargs_route.py` 7/7，含两条负向对照）。**未提交上游，也未准备提交。**

**【大白话讲明白】**

pyflag 干的事是：找一个"代码里写着能开关、但从来没人给它传值"的布尔参数。上一批在 vLLM 上这么找，找到了一条能写的；这一批换到 LMDeploy，找到 5 条，逐条读下来**一条都不成立**。

最像的一条是 `--disable-fastapi-docs`：命令行里有这个参数，服务器里也有这个分支，但中间那个负责多卡起服务的函数 `launch_server` 的形参表里**没有这个名字**——只看签名会以为断在这儿。实际上它用的是 `**kwargs` 兜底，把用户给的值原样收下再转给真正的 serve 函数。本批把这三段真实源码**在 CPU 上串起来跑**：给 `True`，FastAPI 的三个文档地址全变 `None`；给 `False`，`docs_url` 是 `'/'`。链路是通的。

另外四条更简单：`use_fa3` 是 import 时探测显卡算力得出的一个模块级变量，根本不是给用户开的开关；`use_proxy` 的消费者和生产者都在 `autotest/` 目录里，扫描器按路径把它们排除了；`use_tqdm` 在仓库里的生产者只在 `benchmark/` 两个脚本，但 `Pipeline.__call__` 会把 `**kwargs` 透传给 `infer`，所以用户 `pipe(prompts, use_tqdm=True)` 是能用的；`with_version` 是 `setup.py` 自己的内部函数，docstring 写 `default=False` 而代码默认 `True`——这行不一致是真的，但没有任何用户会碰到。

**【What–Why–How】**

- **What：** 5 条 `silently_inert` 的判定分别是 `REACHABLE_BOTH_ROUTES`（disable_fastapi_docs）、`MODULE_GLOBAL_PROBE`（use_fa3）、`TEST_ONLY`（use_proxy）、`LIBRARY_PARAM_REACHABLE_VIA_CALL`（use_tqdm）、`SETUP_INTERNAL`（with_version）。
- **Why：** 共同根因是**工具的对象边界**——pyflag 只认「形参 + 按它分支」做消费者，于是把 `**kwargs` 兜底、模块级全局、测试目录里的形参、`__call__` 透传这几类全部算成了"没人传"。这是 SGLang（34→2）与 vLLM（12→1）之后第三次出现在同一处。
- **How：** `verify_findings.py` 逐条绑行号判（每条带负向对照），`probe_kwargs_route.py` 把 kwargs 收集段、转发行、api_server 分支三段**真实源码**在 CPU 上执行并检查 FastAPI 三个 URL。**边界**：纯静态 + CPU 源码执行，未装 lmdeploy、未起服务、未上 GPU；`launch_server` 的 `**kwargs` 只在 `proxy_url`/多卡路径上生效，单卡路径走的是 CLI 直调 `serve()`，两条都验过。
<!-- contribution-summary:end -->

## 证据链（全部可机读复跑）

| 位置 | 内容 |
|---|---|
| `lmdeploy/cli/utils.py:747` | `--disable-fastapi-docs` 定义（`store_true`，默认 False） |
| `lmdeploy/cli/serve.py:334` | 非 DP 路由：`run_api_server(` 是 `from ...api_server import serve as run_api_server` 的**别名** |
| `lmdeploy/cli/serve.py:366` | DP 路由：`launch_server(` |
| `lmdeploy/serve/openai/launch_server.py:108-114` | 签名 `(num_nodes, node_rank, model_path, backend_config, proxy_url, server_port, **kwargs)`——**具名形参里没有该键** |
| `lmdeploy/serve/openai/launch_server.py:159` | `cur_server_kwargs.update(kwargs)` |
| `lmdeploy/serve/openai/launch_server.py:66` | `_run_server` 内 `serve(model_path, **kwargs)` |
| `lmdeploy/serve/openai/api_server.py:274` | `disable_fastapi_docs: bool = False` |
| `lmdeploy/serve/openai/api_server.py:389` | `if disable_fastapi_docs:` → 三处 URL 置 `None`，否则 `FastAPI(docs_url='/')` |
| `lmdeploy/pytorch/backends/cuda/attention/__init__.py:15` / `:22` | `use_fa3` 模块级初值与探测赋值（仅此两处真赋值） |
| `lmdeploy/pipeline.py:101` / `:120` / `:314` | `infer(use_tqdm=False)` → `tqdm.tqdm(...) if use_tqdm else None`；`__call__` 透传 `**kwargs` |
| `setup.py:63` / `:124` | `with_version=True` 默认值，docstring 却写 `default=False` |

## 为什么这批没有一条能对外

`disable_fastapi_docs` 是唯一有说服力的候选，历史也支持"曾经断过"的猜测：这个参数由 PR #2540（2024-10-08）引入时，只接了非 DP 那条 CLIs 直调；`launch_server` 直到 #3414（2025-05-19）才出现，**但它一出生就是 `**kwargs` 转发**（`cur_server_kwargs.update(kwargs)` 与 `serve(model_path, **kwargs)` 在引入提交里都已存在）。所以严格说：**这条链路上从来没有过"值被吞掉"的版本**。

上游查重也印证没人报过这个问题：#2540 与 #4263 两个 PR 都是在**加**这个功能（后者 closed 未合并），没有任何 issue 说它不生效。

## 方法读数（本批真正的产出）

| 大仓库（>1000 文件） | 开关 | `silently_inert` | 真缺陷 |
|---|---|---|---|
| SGLang `27aedf7f` | 386 | 34 | 2 |
| vLLM `da0b6ac2` | 263 | 12 | 1 |
| **LMDeploy `32283955`** | **38** | **5** | **0** |

三批的失效来源各不相同，但**排除它们的判据是同一套**：

1. **`**kwargs` 兜底**（本批新增的一条）：签名里没有该键 ≠ 收不到。判据是先看签名有没有 `**kwargs`，再看 kwargs 有没有被转发到真正的消费者。
2. **别名调用**（本批新增）：`from X import f as g` 之后调用点写的是 `g`，按 `f` 找会读到 0 处。扫描器与检查器都要处理。
3. 模块级全局 / 测试目录 / `__call__` 透传（前两批已有）。

## 本批自己踩的坑（都已写进判据或 readme）

1. **多行调用只看首行关键字**：`launch_server(\n  a=1,\n  use_x=True)` 的首行没有关键字实参，第一版据此判"没传"，A4/A5 两条当场红。
2. **「检查器能不能看见」写成了「能不能看见某个键」**：自证 check 一度要求签名解析器看见 `disable_fastapi_docs`——而这条判定的结论恰恰是"签名里没有它"，自证与结论自相矛盾，selftest 第二条注入直接报红。
3. **同一 check 混了两件事**：「别名找得到」与「调用点传了值」写在一条判据里，注入改坏别名时它跟着变色，selftest 判"红错地方"。拆成两条才对。
4. **`x = ...` 的三种假阳性**：形参默认值、关键字实参、续行实参，都会被行级正则当成赋值。`use_fa3` 的赋值点统计因此在 `[15, 22, 116, 123, 128]` 与 `[15]` 之间来回翻——最后按「本行前面是否已有 `(` / `,`」「只有缩进且上一非空行以 `,` / `(` 结尾」两条排除，才稳定在 `[15, 15, 22]` 的正确集合上。
5. **先剥三引号再剥注释**：docstring 收尾那行的井号会被当成注释起点、截掉整行，三引号配不上对，整段匹配失效。
