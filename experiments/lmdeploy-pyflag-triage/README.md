# LMDeploy pyflag 逐条取证（2026-11-05，M1）

## 一句话

pyflag 在 LMDeploy `3228395546df`（2026-11-03）上扫出 **38 个开关 / 5 条 `silently_inert`**，
逐条绑行号后 **没有一条构成用户可见缺陷**；结论是**负结论**，不建议对上游提 issue。
证据页：[open-source/lmdeploy/evidence/inert-flags-triage-2026-11.md](../../open-source/lmdeploy/evidence/inert-flags-triage-2026-11.md)。

## 怎么复跑

```bash
# 取源码（按 commit 钉住，别用 refs/heads/main）
curl -sSL -o /tmp/lmdeploy.tar.gz \
  https://codeload.github.com/InternLM/lmdeploy/tar.gz/3228395546dfb2f6427faf39ec0c0c285f4c378a
tar -C /tmp -xzf /tmp/lmdeploy.tar.gz && mv /tmp/lmdeploy-3228395546dfb2f6427faf39ec0c0c285f4c378a /tmp/lmdeploy-main

# 原始扫描（16 s，1113 个 .py；`--json` 落在 tools/pyflag/ 下）
python3 tools/pyflag/scan_unwired_flags.py /tmp/lmdeploy-main --json

# 逐条取证（39 checks）+ 自检（8 条注入）
python3 experiments/lmdeploy-pyflag-triage/verify_findings.py
python3 experiments/lmdeploy-pyflag-triage/verify_findings.py --selftest

# 通路正向对照（CPU 上跑三段真实源码）+ 负向对照
python3 experiments/lmdeploy-pyflag-triage/probe_kwargs_route.py
python3 experiments/lmdeploy-pyflag-triage/probe_kwargs_route.py --selftest
```

## 读数

`3228395546df` · 1113 个 `.py` · 扫描 **16.5 s** · **38 个开关 / 5 条 `silently_inert`**（`never` 3 / `mixed` 2）。

| 开关 | pyflag 的 `mode` | 判定 | 一句理由 |
|---|---|---|---|
| `disable_fastapi_docs` | never | `REACHABLE_BOTH_ROUTES` | `launch_server` 具名形参没有它，但 `**kwargs` 收下并转发给 `serve()`；CLI 另有直调 `serve()` 的路由 |
| `use_fa3` | mixed | `MODULE_GLOBAL_PROBE` | import 期按 SM/CUDA 探测的模块级全局；全文件只有 `:15` 初值与 `:22` 探测两处真赋值 |
| `use_proxy` | never | `TEST_ONLY` | 消费者与 2 个生产者都在 `autotest/`，`lmdeploy/` 包内 0 处 |
| `use_tqdm` | never | `LIBRARY_PARAM_REACHABLE_VIA_CALL` | 仓内生产者只在 `benchmark/`，但 `Pipeline.__call__` 透传 `**kwargs` → `infer` |
| `with_version` | mixed | `SETUP_INTERNAL` | `setup.py` 内部 helper；docstring 写 `default=False`，代码默认 `True` |

**5 条全部不进对外材料。**

## 两个脚本各证明什么

### `verify_findings.py`（39 checks，`--selftest` 8/8）

每条判定由**具名 check** 得出、带**负向对照**（比如 A2c 要求 `else` 分支真的 `docs_url='/'`，
否则"能关"这件事本身不成立；C4c 要求 `lmdeploy/` 包内 0 处，否则"只在测试里"可能只是"没扫到"）。
`--selftest` 每次在一份**拷贝**上注入一处篡改，要求**指定那条** check 变红；
红的是别的 check 也算失败（唯一白名的一处是别名改名会连带让 A6 红——那是同一个事实的两个切面）。

### `probe_kwargs_route.py`（7/7，含 2 条负向对照）

`verify_findings.py` 证明的是静态链路，容易被读成"签名里没有 = 断在这里"。本探针把
**kwargs 收集段**、**`_run_server` 转发那一行**、**api_server 里 `if disable_fastapi_docs:` 那 7 行**
三段**真实源码**分别取出来在 CPU 上串起来跑（不 import lmdeploy、不装依赖、不起服务）：

| 给 CLI 的值 | kwargs 收集段 | `serve()` 收到 | FastAPI 三个 URL |
|---|---|---|---|
| `True` | `True` | `True` | 全 `None`（docs/redoc/openapi 都关） |
| `False` | `False` | `False` | `docs_url='/'`（照常开着） |

负向对照：① 把收集段换成空 dict ⇒ `serve()` 收不到这个键（`<missing>`）；
② 把 api 分支换成"永远 `docs_url='/'`" ⇒ `docs_disabled` 判假。两条都必须生效，否则探针在自欺。

## 历史归属（只说到能做到的一步）

- `--disable-fastapi-docs` 由 PR #2540（`6f34738db0`，2024-10-08）引入，当时只接了 CLI 直调 `serve()` 那条路。
- `launch_server` 由 PR #3414（`776aa1b686`，2025-05-19）引入，**引入即 `**kwargs` 转发**——同一个文件里
  `cur_server_kwargs.update(kwargs)` 与 `_run_server` 的 `serve(model_path, **kwargs)` 都在，
  签名从那时起就是 `(num_nodes, node_rank, model_path, backend_config, proxy_url, **kwargs)`。
- ⇒ 不存在"值曾经被吞掉、后来修好"的版本。**没做全仓 bisect**，只核了这两个提交与当前 main。

## 本批自己踩的五个坑（都进判据或证据页了）

1. 多行调用的关键字实参只看首行 ⇒ 读到空集，A4/A5 当场红。
2. 自证 check 要求签名解析器"看见 `disable_fastapi_docs`"，与结论（签名里没有它）自相矛盾。
3. 「别名找得到」与「调用点传了值」写进同一条判据 ⇒ 注入时连带变红，无法判断红对了没有。
4. `x = ...` 的三种假阳性（形参默认值 / 关键字实参 / 续行实参）让 `use_fa3` 的赋值点集合反复横跳。
5. 先剥三引号再剥注释 ⇒ docstring 收尾行的 `#` 被当注释截断，三引号配不上对。

## 边界

- **纯静态 + CPU 源码执行**：未 `pip install lmdeploy`、未启动任何服务、未上 GPU/NPU、未触上游。
- `launch_server` 的 `**kwargs` 路径只在 `dp > 1` 或设了 `proxy_url` 时走到；单卡路径走 CLI 直调 `serve()`。两条都验了，
  但**两个 `resolve` 都没在真机上跑过**（本机 Apple Silicon，LMDeploy 需要 CUDA）。
- 归档文件：`pyflag-inert.json`（原始读数）、`findings.json`（39 checks）、`kwargs-route.json`（通路读数）。
