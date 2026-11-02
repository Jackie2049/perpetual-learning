# rLLM 工具 `use_async` 自动探测取证

<!-- contribution-summary:start -->
**【仓库】** [perpetual-learning](https://github.com/Jackie2049/perpetual-learning)（AI 技术探索与贡献引擎）

**【贡献点】** 复现 rLLM `Tool.__call__` 的 `use_async=None` 自动探测失效，作为 B 类候选的可执行证据。[取证结论](../../open-source/rllm/tool-use-async-autodetect.md)

**【类型】** 工具（最小复现脚本）

**【价值评估】** 6 / 10 — 脚本能在真实源码上自证缺陷存在并在上游修复后返回非零，但只覆盖工具调用分派，不含训练或端到端验收。

**【进展】** 已对 rllm main `4e828c8` 跑通（exit 0＝缺陷存在），负向对照另行验证。

**【大白话讲明白】**

这份脚本只做一件事：拿真实的 rLLM 工具基类，造一个"只有同步实现"的工具，然后按文档说的方式（`use_async=None`）调用它，看拿回来的是结果还是一个没执行的异步任务。如果拿回来的是后者，脚本以 0 退出并打印缺陷；如果上游已经修好，脚本会以非零退出，提示结论需要重核。

**【What–Why–How】**

- **What：** `repro.py`——针对 `rllm/tools/tool_base.py` 的 `Tool.__call__`，用真实类而非手写替身做四项断言。
- **Why：** 手写模型容易把"我以为的逻辑"当成缺陷，用真实类可以排除这种误判；同时检查签名默认值，因为文档说的默认 `None` 与代码默认 `False` 这一条只能从源码读。
- **How：** 直接 `import rllm.tools.tool_base.Tool`，构造 `SyncOnlyTool`／`AsyncOnlyTool`，断言默认路径可用、`None` 返回协程、`use_async=True` 的守卫不触发。上游修复后任一断言变化即以 1 退出。
<!-- contribution-summary:end -->

## 文件

| 文件 | 作用 |
|---|---|
| `repro.py` | 最小复现与断言；`exit 0` 表示缺陷仍在 |
| `prod_reachability.py` | 生产可达性：AST 调用点扫描 + 两套工具栈核对 + 行为追踪；`exit 0` 表示缺陷在且无生产路径进入自动探测分支 |
| `run-info.json` | 被测 commit、解释器版本、依赖清单与判据 |

## 生产可达性探针：`prod_reachability.py`

配套的第二支探针（2026-11-02 新增）回答"缺陷到底够不够得着生产代码"：

```bash
cd /path/to/rllm && git checkout 4e828c8
python3 /path/to/prod_reachability.py          # exit 0 = 缺陷在，且无生产路径进入自动探测分支
```

四段检查与本次读数：

| 检查 | 读数 |
|---|---|
| AST 扫描全包，`use_async=` 出现在 `if __name__ == "__main__":` 之外 | **0 处** |
| `__main__` 块内的 `use_async=` 调用 | 4 处，取值全是 `True` |
| `rllm/harnesses/tool_calling.py` 分派方式 | `tool.execute(sandbox, **args)`，不经过 `Tool.__call__` |
| 生产构造点 `code_reward.py` 的调用 | 默认 `use_async=False`，同步返回正常；改传 `None` 即返回 coroutine |

**两套工具栈互不相干**：`rllm/tools/` 的 `Tool` 子类走 `__call__` 三元路由；`rllm/harnesses/tools/` 的 `BashTool` 等是普通类，协议是 `.schema` + `.execute(sandbox, ...)`。agent 主流路径结构上绕开本缺陷。

## 候选补丁与回归测试

[open-source/rllm/patches/use-async-autodetect](../../open-source/rllm/patches/use-async-autodetect/readme.md)：改 `has_async` 一行 + 三条常量断言测试。修复后 3 passed；未修复 2 failed（负向对照）。**已知连带**：上游 `test_python_tool` 会由绿转红——它对纯同步工具传了 `use_async=True`，过去靠缺陷吞掉才"通过"。

## 复现

```bash
uv venv --python 3.12 /tmp/rllm-py312
uv pip install --python /tmp/rllm-py312/bin/python pydantic click rich PyYAML tqdm pandas httpx openai requests
cd /path/to/rllm && git checkout 4e828c8
PYTHONPATH=. /tmp/rllm-py312/bin/python repro.py
```

## 验证范围

**已验证**：在 rllm main `4e828c8` 上，用真实 `Tool` 类复现三条缺陷表现；另用改正后的等价实现做负向对照，确认脚本判据对"修好/没修好"是敏感的。可达性探针跑通（exit 0），候选补丁在 `tests/tools/test_tools.py` 上 3 passed、未修复时 2 failed。

**未验证**：未跑 rLLM 全量测试套件；未做端到端 Agent rollout；上游是否已有人认领已于 2026-11-02 补查（无人认领）。
