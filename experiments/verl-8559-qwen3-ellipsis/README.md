# verl #8559：Qwen3 XML 工具解析器遇到 `...` 参数值整轮丢调用（本地复现 + 候选补丁）

仓库：[verl-project/verl](https://github.com/verl-project/verl) `954bd28f`（2026-11-05，fork `Jackie2049/verl` main）
issue：[#8559](https://github.com/verl-project/verl/issues/8559)（2026-11-05 06:57 UTC，作者 super-yuki，**无人认领、零评论**）
渠道：A 类（修已知 bug）——复现 + 根因 + 回归用例 + 候选补丁，全部本地/fork 准备，**未触上游**。

## 结论先说

issue 描述的现象**逐条复现属实**，而且**不止 `...`**：`ast.literal_eval` 接受的 Python 字面量里，
凡是 JSON 没有表示的（`...`、`set`、`bytes`、`complex`）都会走到同一条崩溃路径，
一次就把**本轮所有**工具调用全部丢掉。

**issue 作者 86 秒后自己开了 PR [#8560](https://github.com/verl-project/verl/pull/8560)**（`ccf60061`，`REVIEW_REQUIRED`，只有 CodeRabbit 过过），
它只特判 `Ellipsis`。在**该 PR 自己的分支上**实测：`...` 已修复，**`set` / `bytes` / `complex` 仍按原样崩**（探针 exit 0）。
本目录因此准备两份本地材料：通用的 13 行修复（[fix.patch](fix.patch)），以及直接叠在 #8560 上、
去掉其冗余特判的版本（[on-pr8560.patch](on-pr8560.patch)）。**对外动作全部未做。**

## 一、缺陷链（为什么一次失败丢的是整轮）

`verl/experimental/agent_loop/tool_parser.py` 里 `Qwen3XMLToolParser._parse_xml_function_call`：

```python
param_value = ast.literal_eval(param_value)      # :292  "..." → Ellipsis
...
return FunctionCall(name=..., arguments=json.dumps(param_dict, ensure_ascii=False))   # :329  TypeError
```

`ast.literal_eval` 的语义是「只解析 Python 字面量」，**不保证结果是 JSON 可序列化的**。
上游在 #6542（`2084d8f9`）把 `eval` 换成 `literal_eval` 时，注释写的正是
"literal_eval only parses Python literals (lists, tuples, dicts, numbers, ...)"——安全目标达成了，
但**「Python 字面量集合」比「JSON 值集合」大**这一条没被覆盖。

崩溃发生在 `json.dumps(param_dict)`，即在**整个函数调用的所有参数都解析完之后**。
上层 `extract_tool_calls` 的 `except Exception`（`:373-375`）记一条 `logger.exception` 后 **`return text, []`**
⇒ 本轮**已经解析成功的其它工具调用、以及同一响应里的后续工具调用**全部一起丢掉，
模型看到的是一轮「没有调用工具」的普通回复。

## 二、读数

`probe_issue_8559.py` 直接调真实的 `_parse_xml_function_call`（`tokenizer=None`），对 6 种非 JSON 字面量 + 6 种 JSON 可表示字面量各跑一遍：

| 形参类型 `array` 下的原始值 | 修复前 | 修复后 |
|---|---|---|
| `...` | `TypeError: Object of type ellipsis is not JSON serializable` | `{"items": "..."}` |
| `{1, 2}` | `TypeError: Object of type set ...` | `{"items": "{1, 2}"}` |
| `b'abc'` | `TypeError: Object of type bytes ...` | `{"items": "b'abc'"}` |
| `1+2j` | `TypeError: Object of type complex ...` | `{"items": "1+2j"}` |
| `(1, 2)` | `{"items": [1, 2]}` | `{"items": [1, 2]}`（不变） |
| `frozenset([1])` | `{"items": "frozenset([1])"}`（literal_eval 本来就拒） | 同左 |
| 6 条 JSON 对照（list/dict/int/bool/None/字符串） | 6/6 正常 | 6/6 正常 |

读数：[修复前](raw/probe-before-fix.json)（exit 0）／[修复后](raw/probe-after-fix.json)（exit 2，全部可往返）。
6 条 JSON 对照是**正向对照**：它证明读数不是因为解析器整个坏了而「什么都崩」。

**只有 Qwen3 一支有这个问题**：全仓 `ast.literal_eval` 只出现在这一处（`grep` 命中 1 处代码 + 1 处测试注释）；
Hermes / GPT-OSS / GLM / Seed / MiniMax / Kimi 直接在模型输出上做 `json.loads`，
Gemma4 自己按 `int → float → str` 逐级退化，都不经过 literal_eval。

## 三、候选补丁

[fix.patch](fix.patch)（2 文件 +49 行，其中源码 +13）：

```python
raw_param_value = param_value
try:
    param_value = ast.literal_eval(param_value)
except Exception:
    logger.warning(...); return param_value
# literal_eval also accepts literals with no JSON representation
# ("...", sets, bytes, complex); they would make the json.dumps below
# raise and drop the whole tool call, so degenerate to the raw string.
try:
    json.dumps(param_value)
except TypeError:
    logger.warning(...); return raw_param_value
return param_value
```

设计取舍：

1. **判据是「能不能序列化」，不是「类型白名单」**——白名单（`isinstance(x, (list, dict, int, float, str, bool)) or x is None`）
   会把 `tuple` 也挡掉，而 tuple 现在能正常变成 JSON 数组（`(1,2)` → `[1,2]`），去掉它属于顺带的行为变化。
   探针实测 `json.dumps` 与「白名单 ∪ tuple」在 12 个用例上等价。
2. **退化目标是原始字符串**，与 issue 的 Expected behavior 一致，也与同函数里
   「参数名不在 schema 里 → 直接返回字符串」「literal_eval 抛异常 → 退化字符串」两条现成路径一致。
3. **`except TypeError` 而不是 `except Exception`**：`json.dumps` 只对不可序列化对象抛 `TypeError`
   （`ValueError` 是循环引用/NaN 一类，literal_eval 的产物构造不出来），窄异常不掩盖别的问题。
4. **不动 #6542 的安全属性**：仍然只走 `ast.literal_eval`，`eval` 回归用例照旧，代码执行测试未改且仍绿。

## 四、验证

环境：worktree `/tmp/verl-954bd2`（`954bd28f`）+ `/tmp/verl-8559-venv`（Python 3.12.13、pytest、ruff；CPU、未装 vLLM/SGLang、未跑训练）。

| 项 | 修复前 | 修复后 |
|---|---|---|
| `pytest tests/experimental/agent_loop/test_qwen3_tool_parser_on_cpu.py` | **5 failed / 5 passed**（[log](raw/pytest-before-fix.txt)） | **10 passed**（[log](raw/pytest-after-fix.txt)） |
| `probe_issue_8559.py` | exit 0（缺陷在） | exit 2（全部可往返） |
| `ruff check` / `ruff format --check` | — | 两文件均通过 |

**负向对照**：新增的 5 条测试（4 个参数化 + 1 条端到端）在**未打补丁的源码**上全部红，
打的正是要防的那个行为；原有 5 条在修复前后都绿 ⇒ 补丁没有改动别处的语义。

端到端那条专门盯「丢整轮」：桩一个 `decode()` 返回含两个 `<tool_call>` 的文本，
修复前 `extract_tool_calls` 返回 `[]`，修复后返回两条，第二个调用的参数原样保留（`{"n": 3}`）。

## 五、边界

- **未跑模型**：没有真实 Qwen3-Coder 输出，缺陷由 issue 的日志 + 直接调解析器复现，不是端到端 rollout 复现。
- **只覆盖 `_parse_xml_function_call`**：`extract_tool_calls` 的 `except Exception` 吞异常是**设计如此**（单轮容错），
  本补丁不改它；改它会牵动「部分工具调用是否该保留」的语义，超出 bug 范围。
- **形参类型是 `string`/`str` 时根本不进 literal_eval**（`convert_param_value` 第一条分支直接返回原串），
  所以真实影响面取决于工具 schema 里 `bbox_2d` 一类参数怎么声明——`array`/`object`/数字/布尔才可能触发。
- 未查 issue 之外是否已有同问题 PR（只读了 issue 本身零评论、零 assignee；查重见下）。

## 六、与 PR #8560 的关系（决定对外怎么提）

| 项 | #8560（作者 super-yuki，`ccf60061`） | 本目录 [fix.patch](fix.patch) |
|---|---|---|
| 判据 | `parsed_value is Ellipsis` 特判 | `json.dumps(parsed)` 失败即退化 |
| `...` | 已修 | 已修 |
| `set` / `bytes` / `complex` | **仍崩**（在 `ccf60061` 上实测，探针 exit 0） | 已修 |
| 回归测试 | 1 条（`...`） | 5 条（4 参数化 + 1 端到端防「丢整轮」） |
| 源码改动 | +4 −1 | +13 |

在 `ccf60061` 上把本目录的测试换上去、源码保持 PR 原样跑：`3 failed / 7 passed`，
红的正是 `{1, 2}`／`b'abc'`／`1+2j` 三条，`...` 那条绿——**说明差异只在「通用 vs 特判」，不是评审风格意见**。

[on-pr8560.patch](on-pr8560.patch) 是叠在 `ccf60061` 上的完整版本：删掉已成冗余的 `Ellipsis` 特判、
换成 `json.dumps` 探测 + 5 条测试，在该 HEAD 上 `ruff check`／`format --check` 通过、`pytest 10 passed`、探针 exit 2。

## 七、对外状态

`gh search issues` / PR 只读查询见 [evidence 页](../../open-source/verl/evidence/qwen3-ellipsis-json-serialization.md)。
可做的三个动作**均未执行、待用户逐项授权**：(a) 在 #8560 上留一条只读 review 建议（指出 set/bytes/complex 仍在）；(b) 给 #8559 补一条同类复现评论；(c) 在用户 fork 上另开分支推同类修复。
按仓库铁律，**修改已有 PR 的代码（含推其 head 分支）也属对外动作**，本轮未碰。

## 复现

```bash
git -C <verl> worktree add /tmp/verl-954bd2 954bd28f        # 或任意含 #6542 之后的 main
uv venv --python 3.12 /tmp/verl-8559-venv
uv pip install --python /tmp/verl-8559-venv/bin/python -q pytest numpy torch tensordict codetiming \
    orjson wandb datasets cachetools uvicorn fastapi hydra-core omegaconf pyyaml aiohttp ray pillow transformers
cd /tmp/verl-954bd2
PYTHONPATH=/tmp/verl-954bd2 /tmp/verl-8559-venv/bin/python probe_issue_8559.py    # exit 0 = 缺陷在
/tmp/verl-8559-venv/bin/python -m pytest tests/experimental/agent_loop/test_qwen3_tool_parser_on_cpu.py -q
git apply fix.patch && /tmp/verl-8559-venv/bin/python -m pytest tests/... -q             # 10 passed
```
