# verl #8560 review 草稿（**未发送**）

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（主流 RL 训练框架，本仓历史上做过多次贡献）

**【贡献点】** 给 [PR #8560](https://github.com/verl-project/verl/pull/8560)（修 [issue #8559](https://github.com/verl-project/verl/issues/8559) 的 `...` 参数崩溃）准备一条 review 建议：当前判据只特判 `Ellipsis`，`set`／`bytes`／`complex` 走同一条路径仍会 `TypeError` 并丢掉整轮工具调用。[证据](../evidence/qwen3-ellipsis-json-serialization.md)

**【类型】** 代码评审草稿（未发布）

**【价值评估】** 5 / 10 — 事实可机读、可在 PR 自己分支上复现；但 PR 已存在且判定权在作者与维护者，我方只能提供判据，不能代替决策。

**【进展】** 草稿已就绪，**未发送**；发送需用户逐项授权。附带可直接叠在 `ccf60061` 上的补丁（删冗余特判 + 通用判据 + 5 条测试，该 HEAD 上 10 passed）。

**【大白话讲明白】** 这个 PR 把「参数是 `...` 时别崩」修好了，但同一个原因还会让参数为 `{1, 2}`、`b'abc'`、`1+2j` 时崩掉整轮工具调用。与其一个一个加特判，不如问一句「转出来的东西能不能写成 JSON」，不能就用原始文本。我把这个改动和测试做好了，作者愿意的话可以直接参考；不接受也不影响当前 PR 修掉最主要的那一种。

**【What–Why–How】**

- **What：** `convert_param_value` 的 `literal_eval` 结果是否可 JSON 序列化，PR 只覆盖了 `Ellipsis` 一种。
- **Why：** `ast.literal_eval` 的接受集合包含 set/bytes/complex，`json.dumps(param_dict)` 在同一函数结尾，失败会冒到 `extract_tool_calls` 的 `except` 并被吞成「无工具调用」。
- **How：** 在 `ccf60061` 上实测三类比原样崩；给出 `json.dumps` 探测版补丁与 5 条测试，双向负向对照均通过。**边界**：未跑真实模型；触发面取决于工具 schema 的参数类型声明。
<!-- contribution-summary:end -->

## 拟发送内容（在 #8560 上留一条 review comment，**未发**）

> The fix covers `...`, but the same `literal_eval` → `json.dumps` path also crashes on other Python literals that JSON cannot represent. On this branch (`ccf60061`):
>
> ```
> {1, 2}   -> TypeError: Object of type set is not JSON serializable
> b'abc'   -> TypeError: Object of type bytes is not JSON serializable
> 1+2j     -> TypeError: Object of type complex is not JSON serializable
> ```
>
> Minimal check (CPU only, no tokenizer):
>
> ```python
> parser._parse_xml_function_call("list_tool><parameter=items>{1, 2}</parameter>", [tool_with_array_param])
> ```
>
> Since a failure here always reaches `extract_tool_calls`' `except Exception` and drops **every** tool call of the turn, it may be worth replacing the `Ellipsis` special case with a serializability check:
>
> ```python
> raw_param_value = param_value
> try:
>     param_value = ast.literal_eval(param_value)
> except Exception:
>     ...
>     return param_value
> try:
>     json.dumps(param_value)          # e.g. `...`, sets, bytes, complex
> except TypeError:
>     return raw_param_value
> return param_value
> ```
>
> That keeps `(1, 2) -> [1, 2]` working (a type whitelist would break it) and needs no new imports. I have this plus a parametrised test and an end-to-end one (`extract_tool_calls` on a two-call response) prepared and verified against `ccf60061` — happy to share the diff if useful. No action expected on your side; the `...` case is already covered by this PR either way.

## 旁注：另一条，可能不值得发

`extract_tool_calls` 用 `except Exception` 把**任何**解析错误变成「本轮没有工具调用」，
使得单个参数的问题代价是整轮。改成「逐调用容错、保留已解析成功的调用」属于设计变更，
不是这个 bug 的必要部分——已在本目录 [实验 README](../../experiments/verl-8559-qwen3-ellipsis/README.md#五验证) 里记为**范围外**。
