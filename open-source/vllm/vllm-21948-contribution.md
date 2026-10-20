# vLLM：补齐多回答并行采样的参数测试

<!-- contribution-summary:start -->
**【仓库】** [vLLM](https://github.com/vllm-project/vllm)（大模型推理框架）

**【贡献点】** 补齐多回答并行采样的参数测试。[Issue #21948](https://github.com/vllm-project/vllm/issues/21948)

**【类型】** 测试改进

**【价值评估】** 5 / 10 — 边界测试可防止参数组合回归；旧问题当前状态与测试缺口待核对。

**【进展】** 历史准备材料：已规划 n>1 的测试场景，未记录测试贡献已提交或合并。

**【大白话讲明白】**

同一个提示词可以一次生成多个回答。某些参数在只生成一个回答时正常，生成多个时却可能组合出错。我们准备围绕这些组合补测试，先保证行为符合接口约定。

**【What–Why–How】**

- **What：** 并行采样 n>1 与相关采样参数的兼容性。
- **Why：** 单回答测试不能覆盖多回答的返回结构、数量与约束。
- **How：** 对照 issue 复现输入，补参数化测试和错误提示断言；先重新核对现有测试避免重复。
<!-- contribution-summary:end -->

## 详细记录

> 以下保留原始分析与历史进展；当前结论见上方速览。

> Parallel Sampling Parameter Tests for n>1
> Issue: https://github.com/vllm-project/vllm/issues/21948
> Labels: bug, good first issue, keep-open
> 状态: 准备中 (需要用户审查并提交)

## Issue 概要

为 vLLM V1 的 `n>1` (并行采样) 功能添加参数化测试:
- 测试 `AsyncLLM` 和 `LLMEngine` 两个入口
- 覆盖所有 `output_kind` 值: `DELTA`, `FINAL_ONLY`
- 可能发现 `LLMEngine + DELTA + n>1` 的 bug

## 当前测试状态

| 文件 | n>1 测试 | output_kind 覆盖 | 缺失 |
|------|---------|-----------------|------|
| test_parallel_sampling.py | 单元测试 | DELTA + FINAL_ONLY | 无集成测试 |
| test_llm_engine.py | 有(n=2-20) | 仅 FINAL_ONLY | DELTA 测试 |
| test_async_llm.py | 有(n=3) | DELTA + FINAL_ONLY | 更多参数化 |

## 贡献计划

### Step 1: 在 test_parallel_sampling.py 添加 CUMULATIVE 测试
```python
def test_parent_request_to_output_cumulative() -> None:
    """Test CUMULATIVE output_kind with n>1 parallel sampling."""
    # CUMULATIVE 模式应该累积所有已完成的 child outputs
```

### Step 2: 在 test_llm_engine.py 添加参数化测试
```python
@pytest.mark.parametrize("output_kind", [
    RequestOutputKind.DELTA,
    RequestOutputKind.FINAL_ONLY,
])
@pytest.mark.parametrize("n", [2, 3, 5])
def test_parallel_sampling_output_kinds(self, output_kind, n):
    """Test n>1 sampling with different output kinds."""
```

### Step 3: 在 test_async_llm.py 添加参数化测试
```python
@pytest.mark.parametrize("output_kind", [
    RequestOutputKind.DELTA,
    RequestOutputKind.FINAL_ONLY,
])
def test_parallel_sampling_async(self, output_kind):
    """Test async n>1 sampling with different output kinds."""
```

### Step 4: 运行测试验证
```bash
.venv/bin/python -m pytest tests/v1/engine/test_parallel_sampling.py -v
.venv/bin/python -m pytest tests/v1/engine/test_llm_engine.py -v -k parallel
.venv/bin/python -m pytest tests/v1/engine/test_async_llm.py -v -k parallel
```

## 注意事项

1. **AGENTS.md 规定**: 纯 AI PR 不允许, 需要人工审查者理解并捍卫每一行改动
2. **贡献流程**: 先检查重复 → fork → PR → CI 通过
3. **Commit 信息**: 必须包含 AI 辅助声明
4. **Pre-commit**: 必须通过所有 lint 检查

## 相关代码路径

- `vllm/v1/engine/parallel_sampling.py` — ParentRequest 实现
- `vllm/sampling_params.py` — RequestOutputKind 定义
- `vllm/v1/engine/__init__.py` — EngineCoreRequest
