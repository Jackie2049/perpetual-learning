# rLLM 工具调用的 `use_async=None` 自动探测从未生效

<!-- contribution-summary:start -->
**【仓库】** [rLLM](https://github.com/rllm-org/rllm)（Agent 强化学习框架）

**【贡献点】** `Tool.__call__` 文档承诺 `use_async=None` 会自动选择同步/异步实现，但签名默认是 `False`、且 `has_async` 判定对任何子类恒为 True，导致自动探测分支不可达、`use_async=True` 的守卫也不触发。[源码](https://github.com/rllm-org/rllm/blob/4e828c8/rllm/tools/tool_base.py#L137-L171) · [文档](https://github.com/rllm-org/rllm/blob/4e828c8/docs/api/tools.mdx#L85-L90) · [复现脚本](../../experiments/rllm-tool-use-async-probe/repro.py)

**【类型】** 问题诊断（B 类：主动发掘，附最小复现）

**【价值评估】** 6 / 10 — 缺陷确定、可复现、修法局部；但当前生产路径不传 `use_async`，实际影响面主要是文档契约与自定义工具作者，因此不主张它是性能或正确性回归。

**【进展】** 已对 main `4e828c8`（2026-10-08）完成静态取证与真实代码复现，负向对照通过。**对外动作（issue／PR）未做，待用户逐项授权。**

**【大白话讲明白】**

rLLM 的工具类可以只写同步版本，也可以只写异步版本。它的文档说：调用时如果 `use_async` 不传（或传 `None`），框架会自动判断用哪个——有异步就用异步，没有就用同步。

实际情况是：不传时默认值是"用同步"，文档说的自动判断根本走不到；而即使你显式写 `use_async=None`，框架的"有没有异步版本"这一判断在任何子类上都返回"有"。于是一个**只写了同步版本的工具**，被要求走异步时不会报错，而是返回一个「还没真正执行的异步任务对象」——调用方以为拿到了结果，实际上什么也没做。本该拦下这种情况的那句 `NotImplementedError` 也就永远不会触发。

**【What–Why–How】**

- **What：** `rllm/tools/tool_base.py` 的 `Tool.__call__` 中三处行为与文档不符：签名的 `use_async=False`、`has_async` 的判定表达式、以及依赖该判定的 `NotImplementedError` 守卫。范围限于工具调用分派，不涉及训练或推理内核。
- **Why：** `has_async = inspect.isfunction(self.__class__.async_forward) or self.__class__.async_forward is Tool.async_forward`。`Tool.async_forward` 本身是一个**普通 `async def` 函数**，所以 `inspect.isfunction(...)` 恒为 True；后半段 `X is X` 也恒为 True。两者都用不到"子类是否覆盖了它"这个信息，于是表达式对任何子类都返回 True。一旦 `has_async` 恒真，`use_async=True` 分支的 `else: raise NotImplementedError` 成为不可达代码，自动探测分支也总是选 `async_forward`。
- **How：** 用真实的 `rllm.tools.tool_base.Tool` 构造两个子类（只实现 `forward` / 只实现 `async_forward`），在 Python 3.12 下逐项断言。判据是"能否区分是否覆盖"，正确写法应为 `type(self).async_forward is not Tool.async_forward`；负向对照显示改用该写法后，自动探测能正确选到同步实现、且 `use_async=True` 会按设计抛出异常。
<!-- contribution-summary:end -->

## 缺陷实际表现

对 main `4e828c8` 跑 [repro.py](../../experiments/rllm-tool-use-async-probe/repro.py)（Python 3.12.13，venv 见下文）： 

```
[probe] Tool.__call__ signature default: use_async=False
  -> documented 'default is None' does NOT match the signature default False
[probe] has_async computed for a sync-only subclass: True
  -> the expression is True for every subclass, so it cannot discriminate
[probe] sync-only tool, default call t(3) -> 6
[probe] sync-only tool, documented auto-detect t(3, use_async=None) -> coroutine
[probe] sync-only tool, t(3, use_async=True) -> coroutine (guard expected to raise)
[probe] async-only tool, auto-detect awaited -> 6
```

三条读数：

1. **默认路径是好的**：不传 `use_async` 时走 `forward()`，同步工具正常返回 `6`。这是框架内部唯一的调用方式，所以现有训练/评测流程不受影响。
2. **文档承诺的自动探测返回协程**：`use_async=None` 返回 `coroutine` 而非结果。若调用方在这个返回值上取 `.output`／`.error` 会得到 `AttributeError`；若直接丢弃，则工具**静默未执行**。
3. **守卫失效**：`use_async=True` 在纯同步工具上应抛 `NotImplementedError`（该错误消息就写在同一个 `if` 里），实际返回协程。

## 影响面

- 仓库内真实工具：`grep "def forward" rllm/tools/` 命中 **11** 处，`grep "def async_forward"` 命中 **2** 处（`mcp_tool.py`、`firecrawl_tool.py`）。即绝大多数工具是同步工具，恰好是自动探测失灵时后果最重的那一类。
- 框架内部调用点：`grep -rn "use_async=" rllm/` 的命中**全部位于各文件的 `__main__` 演示块**，生产路径不传该参数。这是本条被评为 6 分而非更高的直接原因。
- `tests/tools/test_tools.py:202` 用 `use_async=True` 调 `python_tool`，但该函数只 `print` 结果、不含断言，因此缺陷长期未被测试发现。

## 复现环境

```bash
uv venv --python 3.12 /tmp/rllm-py312
uv pip install --python /tmp/rllm-py312/bin/python pydantic click rich PyYAML tqdm pandas httpx openai requests
cd /path/to/rllm && git checkout 4e828c8
PYTHONPATH=. /tmp/rllm-py312/bin/python /path/to/repro.py   # exit 0 = 缺陷仍存在
```

系统 Python 3.9.6 无法导入 rllm（源码使用 PEP 604 的 `str | None` 标注），必须 ≥3.10。

## 修复方向（未提交，供授权后使用）

最小改动是把判定改成"看子类是否覆盖"：

```python
has_async = type(self).async_forward is not Tool.async_forward
```

其余三处配套改动缺一不可，否则文档仍然不实：

1. 签名默认值 `use_async: bool | None = None`，与文档和 docstring 一致。
2. `use_async is True` 时若 `has_async` 为假，让 `NotImplementedError` 真正可达（当前它不可达）。
3. `tests/tools/test_tools.py` 中对 `use_async=True` 的用例补断言——现状只打印不断言，等于没有回归保护。

负向对照（`/tmp/fixed_tool.py`，与 `repro.py` 同一判据）显示改后三条行为均正确：

```
fixed, sync-only, default     -> 6
fixed, sync-only, None        -> 6
fixed, sync-only, True        -> NotImplementedError (guard fires correctly)
```

## 边界与不确定项

- **不主张现有训练受影响**：生产路径不传 `use_async`，默认同步分支是正确的。若把本项描述成"训练结果有问题"就是过度声明。
- **未核实下游 fork 的调用约定**：本仓库之外若有代码传 `use_async=None` 并依赖当前（错误的）行为，改签名默认为 `None` 会让它们开始走自动探测。这一点需要在 PR 描述里说明。
- **未做上游 issue 检索**：本轮未查询 GitHub issue/PR 是否已有人认领该缺陷；对外前必须补这一步，避免重复。
- **未验证 `mcp_tool.py` 的异步路径**：它是真实实现了 `async_forward` 的两个工具之一，自动探测失效对它反而是"碰巧正确"；未对它做端到端调用验证。

## 状态

M1 2026-11-02 本地取证完成，**对外动作未做**。判定为 B 类候选，价值集中在"文档与代码不一致 + 守卫不可达 + 测试无断言"，适合作为一条独立的、规模小的 issue（附复现脚本），是否提出待用户决定。
