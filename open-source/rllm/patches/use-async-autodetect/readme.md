# TM-009 候选补丁：`has_async` 判定修正

<!-- contribution-summary:start -->
**【仓库】** [rLLM](https://github.com/rllm-org/rllm)

**【贡献点】** `Tool.__call__` 里 `has_async` 对任何子类恒为 True，导致自动探测分支永远选异步、`use_async=True` 的守卫不可达；改一行判定式 + 三条回归测试。证据：[取证](../../tool-use-async-autodetect.md) · [补丁](use-async-autodetect.patch) · [可达性探针](../../../../experiments/rllm-tool-use-async-probe/prod_reachability.py)

**【类型】** 缺陷修复（B 类候选的修复形态），本地就绪、**未对外**。

**【价值评估】** 6 / 10 — 缺陷确定、修法一行，但生产路径不触达，不因加了补丁而上调。

**【进展】** 2026-11-02 对 main `4e828c8` 完成：补丁 + 回归测试 + 正负向对照。**对外动作未做，待用户逐项授权。**

**【大白话讲明白】**

框架想知道某个工具「有没有写异步版本」。原来的判断是「异步方法是个函数吗」或者「它是不是基类那个方法」——可基类那个方法本来就是普通函数，第二条又是拿它跟自己比，所以答案永远是「有」。改成「它是不是基类那个方法」（不是——才说明子类自己写了）就对了。

**【What–Why–How】**

- **What：** `rllm/tools/tool_base.py::Tool.__call__` 第 153 行的 `has_async` 判定式。
- **Why：** `inspect.isfunction(Tool.async_forward)` 恒真（基类是普通 `async def`），`X is Tool.async_forward` 在 `X` 就是基类方法时也恒真，两半都用不到「子类是否覆盖」这一信息。
- **How：** 改为 `self.__class__.async_forward is not Tool.async_forward`，与相邻 `has_sync` 的写法对齐；补三条测试覆盖「同步工具自动探测」「同步工具显式要异步」「异步工具自动探测」。
<!-- contribution-summary:end -->

## 补丁内容

[use-async-autodetect.patch](use-async-autodetect.patch)（基于 `4e828c8`，`git apply` 干净）：

```diff
-        has_async = inspect.isfunction(self.__class__.async_forward) or self.__class__.async_forward is Tool.async_forward
+        has_async = self.__class__.async_forward is not Tool.async_forward
```

加 `tests/tools/test_tools.py` 三条测试：

| 测试 | 断言（常量，不从被测输出推导） |
|---|---|
| `test_auto_detect_prefers_sync_implementation` | `_SyncOnlyTool.forward(3) * 2`，`tool(3, use_async=None) == 6` |
| `test_use_async_true_requires_async_forward` | 同步工具 `use_async=True` 抛 `NotImplementedError` |
| `test_auto_detect_uses_async_implementation` | 异步工具 `use_async=None` → `asyncio.run(result) == 6` |

## 验证（Python 3.12.13，`/tmp/rllm-py312`）

```
[修复后] 3 passed                       # 三条新测试
[未修复] 2 failed, 1 passed             # 负向对照：前两条被抓到
```

第三条在未修复时也通过，属预期——纯异步工具在旧逻辑下正好"歪打正着"，所以它单独不构成判据，必须与前两条一起跑。

## 已知副作用：会让一条既有测试变红（这条要写进 PR 描述）

`tests/tools/test_tools.py::test_python_tool` 里对**纯同步**的 `PythonInterpreter` 调用 `python_tool(code, use_async=True)`。旧逻辑下它静默返回一个未被 await 的协程，测试直接通过；修复后按设计抛 `NotImplementedError`，测试变红。

```
[修复后] tests/tools/test_tools.py::test_python_tool  FAILED  NotImplementedError: Tool PythonInterpreter does not implement async_forward() but use_async=True was specified.
[未修复] 1 passed
```

**这正是缺陷本身的证据**——那条测试过去之所以"过"，是因为守卫从来不生效。所以补丁不能只改判定式：要么把该调用改回同步路径，要么让该测试接受 `asyncio.run(...)` 的返回。本仓库的补丁**暂未处理这一处**，因为怎么改涉及上游测试意图（它想测"异步执行"却对着一个同步工具测），应由维护者定。

## 未做

- 未改签名默认值（`use_async=False`）。文档写"默认 None 自动探测"与签名矛盾，但把默认改成 `None` 会让所有同步工具默认走 `async_forward` 分支（现在修好后同步工具会走 `forward`，行为其实等价），仍属行为变化，建议单独讨论。
- 未跑 rLLM 全量测试，只跑 `tests/tools/test_tools.py`。
- 未推 fork、未开 PR、未评论。
