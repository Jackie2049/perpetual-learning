# verl：修正检查点清理开关的类型标注

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（后训练框架）

**【贡献点】** 修正检查点清理开关的类型标注。[源码目录](https://github.com/verl-project/verl/tree/main/verl/workers/engine/veomni)

**【类型】** 代码维护

**【价值评估】** 2 / 10 — 静态类型一致性问题，运行时影响小，不作为单独 PR。

**【进展】** 暂缓：已确认 bool 开关被标成 int，等待相关文件的实质改动一起修正。

**【大白话讲明白】**

一个开关实际接受“是或否”，代码提示却写成了整数。程序通常仍能运行，但会误导阅读者和静态检查工具。这属于可以顺手修的小问题。

**【What–Why–How】**

- **What：** VeOmni load_checkpoint 中 del_local_after_load 的类型契约。
- **Why：** 该处 int 标注与 True 默认值及其他后端 bool 声明不一致。
- **How：** 与其他后端签名对齐为 bool；本项仅静态核验，不需要虚构 GPU 结果。
<!-- contribution-summary:end -->

## 详细记录

> 以下保留原始分析与历史进展；当前结论见上方速览。

> 状态：`verified-static`（类型标注错误确认；纯静态，无需上机）
> 首次发现：2026-08-03
> 来源：静态扫描

## 1. 问题

`verl/workers/engine/veomni/transformer_impl.py:567`：

```python
def load_checkpoint(
    self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
) -> None:
```

`del_local_after_load: int = True` —— 类型标注写成 `int`，默认值却是 `True`（bool）。应为 `bool`。

## 2. 对比正确签名

同参数在其他文件都是 `bool`：

| 文件 | 签名 |
|---|---|
| `checkpoint/checkpoint_manager.py:143` | `del_local_after_load: bool = False` |
| `engine/base.py:272` | `del_local_after_load: bool = True` |
| `workers/engine_workers.py:443` | 透传 |

veomni 的 `int = True` 是唯一错标。

## 3. 影响

- Python 运行时**不强制**类型，所以功能上无 bug（`True`/`1` 都可 truthy 传给下层）。
- 但违反类型契约，误导 mypy/静态分析/IDE 提示；且与同代码库其他 `del_local_after_load: bool` 签名不一致，读者会困惑。

## 4. 修复

单行：

```python
# verl/workers/engine/veomni/transformer_impl.py:567
-        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
+        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True, **kwargs
```

## 5. 评估

- 影响=2（类型契约，非功能 bug） 成本=1（单行） 接受度=4（低风险，但太琐碎 maintainer 可能觉得不值单独 PR） 契合度=4 → 性价比 12
- **建议**：不单独提 PR。**攒着，等同一文件有实质改动时顺带修**，或并入某个更大的 veomni/type-consistency PR。单提一个单行类型标注，性价比不划算、可能被视作刷 commit。
- 验证：纯静态，无需 GPU。

## 6. 待办

- [ ] 攒着，不作为独立 PR；挂到下次 veomni 相关改动
