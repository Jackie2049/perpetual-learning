# 独立复核 TM-009：rLLM `Tool.__call__` 的自动探测分支

<!-- contribution-summary:start -->
**【仓库】** [rllm-org/rllm](https://github.com/rllm-org/rllm)（RL 训练框架，5905 stars）

**【贡献点】** 复核 TM-009 的两条论断并补充第三点。**结论：两条论断成立，且危害比原记录描述的更实际**——只实现同步 `forward` 的工具走自动探测分支时**不报错**，而是静默返回 coroutine（基类 `async_forward` 转调 `self.forward`），调用方不 await 就拿到一个未执行的对象。**本地复核，未对外发布**。取证：[../../open-source/rllm/tool-use-async-autodetect.md](../../open-source/rllm/tool-use-async-autodetect.md)

**【类型】** 问题诊断（复核 + 危害面补充）

**【价值评估】** 6 / 10 — 原记录的价值评估维持不变（生产路径不传该参数，现有训练不受影响）；本次复核新增的是"失效形态"而非"影响范围"，不改变可发表性判断。

**【进展】** 已完成三条论断的源码核对，并补做原记录标记为"未做"的上游查重（`async_forward` 零命中，`use_async` 仅命中一条无关 PR）。**未发布任何上游内容**。

**【大白话讲明白】**

rLLM 的工具基类支持两种实现：同步的 `forward` 和异步的 `async_forward`。文档说"不指定 `use_async` 时会自动挑一个能用的"。

实际有三个问题：一是那个"自动挑"的分支根本没被走到（默认值是 False 不是文档说的 None）；二是判断"有没有异步实现"的那行代码永远返回"有"；三是——这次新发现的——因为基类的异步实现默认就是去调同步实现，所以**只写同步的工具被误判成异步时不会报错**，而是返回一个"还没执行"的对象。调用方如果不 await，就什么也没拿到，也不会有任何提示。

好在这个参数目前生产代码里没人传，所以现在的训练不受影响。

**【What–Why–How】**

- **What：** 复核 `rllm/tools/tool_base.py` 的 `Tool.__call__`（137 行）在 `use_async` 未指定时的实际行为。
- **Why：** 第 137 行签名是 `use_async=False`，而 141–142 行 docstring 写「If use_async is None (default)」——**文档与签名矛盾**，自动探测分支默认不可达。同时第 154 行 `has_async = inspect.isfunction(self.__class__.async_forward) or self.__class__.async_forward is Tool.async_forward`：`Tool.async_forward` 是真实定义的 `async def`，`inspect.isfunction` 为真，故 `has_async` 对**任何**子类恒为 True（含只实现同步的那些）。后果不是抛错：基类 `async_forward`（127 行）体内 `return self.forward(*args, **kwargs)`，所以误判会静默返回 coroutine。
- **How：** 对 main（`4e828c8` 同批快照）做 AST 与源码核读；用最小类层次验证两个谓词的真值表（`Tool`／`AsyncOnly`／`SyncOnly` 三者 `has_async` 全为 True）；上游查重走 GitHub Search API。**边界**：未在真实 rLLM 运行时验证（缺 `pydantic` 依赖），三条论断均为**静态源码 + 谓词真值表**层面；未运行任何训练。**不主张**"训练结果有问题"——生产路径不传该参数，这一点原记录已核。
<!-- contribution-summary:end -->

## 一、复核结论

| 论断 | 复核结果 |
|---|---|
| ① 签名默认 `False`，docstring 却写 `None (default)`，自动探测分支默认不可达 | **成立**（137 行 vs 141–142 行） |
| ② `has_async` 对任何子类恒为 True | **成立**（`Tool.async_forward` 是真实 `async def`，`inspect.isfunction` 为真） |
| ③ **新增**：误判的后果是静默返回 coroutine，不是报错 | **成立**（基类 `async_forward` 转调 `self.forward`） |

## 二、谓词真值表（最小类层次，非手写替身逻辑）

```
Tool       has_async=True  has_sync=False
AsyncOnly  has_async=True  has_sync=False
SyncOnly   has_async=True  has_sync=True     ← 只实现同步，仍被判为有异步
```

第三行是关键：**只实现 `forward` 的工具，`has_async` 依然是 True**。

## 三、为什么危害形态比原记录更实际

原记录写的是"自动探测分支默认不可达"。但真正会咬人的场景是**分支可达时**（调用方显式传 `use_async=None`）：

```python
class SyncOnlyTool(Tool):
    def forward(self, x): return f"sync:{x}"

t = SyncOnlyTool()
t("hello", use_async=None)   # 不抛错，返回 coroutine —— 因为
                             # Tool.async_forward 体内是 return self.forward(...)
```

调用方若按同步习惯直接使用返回值，拿到的是**未执行的 coroutine**，没有异常、没有日志。

## 四、上游查重（原记录标记为"未做"）

| 查询 | 结果 |
|---|---|
| `repo:rllm-org/rllm async_forward` | **0 命中** |
| `repo:rllm-org/rllm use_async` | 1 命中，为无关 PR（#460 Daytona sandbox backend） |

→ **上游无人认领该缺陷**。

## 五、边界

- **未在真实运行时验证**：`import rllm.tools.tool_base` 需要 `pydantic`，本机未安装；三条论断基于静态源码与谓词真值表，未跑真实工具调用。
- 未运行任何训练或评测；**不主张**"训练结果有问题"（生产路径不传该参数）。
- 只覆盖 `4e828c8` 同批快照；上游改动后结论需重核。
- **未发布任何上游内容**。
