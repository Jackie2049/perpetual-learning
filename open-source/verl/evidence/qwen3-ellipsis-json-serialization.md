# verl：Qwen3 XML 工具解析器遇到非 JSON 字面量（`...`/set/bytes/complex）整轮丢工具调用

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（主流 RL 训练框架，本仓历史上做过多次贡献）

**【贡献点】** 多轮工具调用用 `multi_turn.format=qwen3_coder` 时，模型把参数值吐成 `...` 会触发 `TypeError: Object of type ellipsis is not JSON serializable`，异常从 `extract_tool_calls` 冒出去后**该轮所有工具调用全部丢掉**。[Issue #8559](https://github.com/verl-project/verl/issues/8559)（无人认领）、[PR #8560](https://github.com/verl-project/verl/pull/8560)（作者同一天自开，只修 `...`，`REVIEW_REQUIRED`）。**在 #8560 自己的分支上实测 `set`／`bytes`／`complex` 仍按原样崩** ⇒ 修法应从「特判 Ellipsis」改成「字面量能解析但没法序列化就退回原字符串」。[实验、补丁与读数](../../experiments/verl-8559-qwen3-ellipsis/README.md)

**【类型】** A 类（修已知 bug）：复现 + 根因 + 通用修复 + 回归用例，**本地/fork 准备，未提交上游**

**【价值评估】** 6 / 10 — 崩溃确凿、CPU 秒级可复现、一次事故丢整轮工具调用（作者日志 38 步里出现 2 次）；但已有同作者 PR 占位，本项的价值窄化为「把特判改成通用判据」，能否被采纳取决于 PR 作者与维护者。

**【进展】** 2026-11-05 在 fork main `954bd28f` 上复现并定位；通用补丁源码 +13 行、新增 5 条测试，`pytest` 从 `5 failed / 5 passed` 变为 `10 passed`，`ruff check`／`format --check` 通过；在 `ccf60061`（#8560）上，PR 原样源码跑同一套测试 `3 failed / 7 passed`。**对外（issue 评论、PR review、fork 分支）均未执行，待用户逐项授权。**

**【大白话讲明白】** Qwen3 系列模型调用工具时，参数是写成 XML 文本交给解析器的。解析器为了把 `[1, 2, 3]` 这种文本变成真正的数组，会用 Python 的「安全字面量解析」把它转成对象——顺带地，`...`、`{1, 2}`、`b'abc'`、`1+2j` 这些** Python 合法但没有 JSON 写法**的字面量也会被转成对象。随后要把整包参数转成 JSON 字符串，一遇到这些对象就抛异常。异常被上层吞掉、返回「这一轮没有调用工具」，于是模型这一步白跑、奖励信号也跟着错。现在别人提的 PR 只挡住 `...` 这一种，另外三种照崩；修法其实只需一句判断：转出来的东西序列化不了，就把原始文本当字符串用。

**【What–Why–How】**

- **What：** `verl/experimental/agent_loop/tool_parser.py` 的 `convert_param_value` 在 `ast.literal_eval` 之后直接把结果放进 `param_dict`，由同一函数的 `json.dumps(param_dict)` 收尾；JSON 表达不了的 Python 字面量会让这一步抛 `TypeError`，上层 `except Exception` 返回 `text, []`。
- **Why：** #6542（`2084d8f9`）把 `eval` 换成 `literal_eval` 是为了禁掉代码执行，安全目标达成，但「Python 字面量集合 ⊋ JSON 值集合」这一条没被覆盖；全仓只有 Qwen3 这一支用 `literal_eval`（其他解析器直接 `json.loads`），所以影响面就是 `qwen3_coder` 这一条路径。
- **How：** 直接调真实 `_parse_xml_function_call` 对 12 种取值各跑一遍（6 种非 JSON + 6 条 JSON 正向对照）；修复后用 `json.dumps` 探测代替类型白名单，保证 `tuple → JSON 数组` 这类现有行为不变；双向负向对照（未修源码上新增测试全红；#8560 源码上恰好红掉三项）。**边界**：未跑真实模型，未做端到端 rollout 复现；形参声明为 `string` 时根本不过这一段，真实触发面取决于工具 schema。
<!-- contribution-summary:end -->

## 大白话

verl 的多轮 Agent 训练里，模型「调工具」这件事是从模型输出的**文本**里解析出来的。Qwen3-Coder 系列的格式长这样：

```xml
<tool_call>
<function=image_zoom_in_tool>
<parameter=bbox_2d>[120, 44, 300, 210]</parameter>
</function>
</tool_call>
```

解析器要把 `[120, 44, 300, 210]` 从字符串变成真正的数组，于是用了 `ast.literal_eval`。它的定位是「只解析 Python 字面量、不执行代码」，这是对的。但它接受的**范围比 JSON 大**：`...`（Ellipsis）、`{1, 2}`（set）、`b'abc'`（bytes）、`1+2j`（复数）都能解析成功，而 JSON 里没有这几种东西。

解析完全部参数之后，代码要把整个参数字典转成 JSON 字符串交给工具执行。**就在这一步炸了**——而且炸的不是一个参数，是**整个函数调用**；上层的 `except` 把异常吞掉、返回「这一轮没有工具调用」。

后果不是「这个工具没调成」，而是：
- 同一轮里**其它本来正常的工具调用也一起消失**；
- 模型看到的观察是「我明明调了工具，环境却说我没调」，训练信号与事实不一致；
- 作者日志里 38 个训练步中出现了 2 次。

## 一、缺陷链（行号基于 `954bd28f`）

```python
# verl/experimental/agent_loop/tool_parser.py
param_value = ast.literal_eval(param_value)            # :292  "..." → Ellipsis
...
return FunctionCall(name=..., arguments=json.dumps(param_dict, ensure_ascii=False))   # :329  TypeError
```

```python
# :366-375   上层把任何异常都变成「没有工具调用」
except Exception as e:
    logger.exception(f"Error in extracting tool call from response: {e}")
    return text, []
```

`_parse_xml_function_call` 的入口在 `:211`，`convert_param_value` 是它的内层闭包（`:224`），
所以这个失败**不区分是哪个参数、也不保留已经解析成功的其它调用**。

## 二、读数目测表

`experiments/verl-8559-qwen3-ellipsis/probe_issue_8559.py`，形参声明为 `array`：

| 原始值 | `954bd28f`（main） | `ccf60061`（PR #8560） | 通用修复 |
|---|---|---|---|
| `...` | TypeError（ellipsis） | `{"items": "..."}` | `{"items": "..."}` |
| `{1, 2}` | TypeError（set） | **TypeError（set）** | `{"items": "{1, 2}"}` |
| `b'abc'` | TypeError（bytes） | **TypeError（bytes）** | `{"items": "b'abc'"}` |
| `1+2j` | TypeError（complex） | **TypeError（complex）** | `{"items": "1+2j"}` |
| `(1, 2)` | `{"items": [1, 2]}` | `{"items": [1, 2]}` | `{"items": [1, 2]}` |
| `frozenset([1])` | `{"items": "frozenset([1])"}` | 同左 | 同左 |
| 6 条 JSON 对照 | 6/6 正常 | 6/6 正常 | 6/6 正常 |

读数归档：[main 修复前](../../experiments/verl-8559-qwen3-ellipsis/raw/probe-before-fix.json)（exit 0）、
[main 修复后](../../experiments/verl-8559-qwen3-ellipsis/raw/probe-after-fix.json)（exit 2）、
[#8560 原样](../../experiments/verl-8559-qwen3-ellipsis/raw/probe-on-pr8560.json)（exit 0，`['set','bytes','complex']`）、
[#8560 + 通用修复](../../experiments/verl-8559-qwen3-ellipsis/raw/probe-on-pr8560-plus-fix.json)（exit 2）。

**正向对照不可省**：6 条 JSON 字面量在两版上都正常，说明读数反映的是「这一类值」而不是「解析器整体坏了」。

## 三、影响面

- 只影响 `Qwen3XMLToolParser`（`multi_turn.format=qwen3_coder`／`qwen3.5`）。全仓 `ast.literal_eval` 只有这一处代码引用；
  Hermes / GPT-OSS / GLM / Seed / MiniMax / Kimi 直接对模型输出 `json.loads`，Gemma4 自己按 `int → float → str` 退化。
- 触发需要参数**声明类型不是 `string`**：`convert_param_value` 对 `string`/`str`/`text`/`varchar`/`char`/`enum` 直接返回原串，根本不进 `literal_eval`。
  所以真实触发面是「schema 里声明成 `array`/`object`/数字/布尔的参数，模型却吐了非 JSON 字面量」。
- 失败模式是**静默降级**（只剩一条 `logger.exception`，训练照常继续），不是 crash——这也是它存在很久没被发现的原因。

## 四、修复

```python
raw_param_value = param_value
try:
    param_value = ast.literal_eval(param_value)
except Exception:
    logger.warning(...); return param_value
try:
    json.dumps(param_value)                     # 判据：能不能序列化，不列类型白名单
except TypeError:
    logger.warning(...); return raw_param_value  # "..."、{1,2}、b'abc'、1+2j 退回原文本
return param_value
```

为什么不列白名单：白名单（`list/dict/int/float/str/bool/None`）会把 `tuple` 一起挡掉，
而 `(1, 2) → [1, 2]` 是现状行为，`json.dumps` 探测对它是放行的。探针 12 个用例两种写法等价，
但「能不能序列化」与 `json.dumps` 这一行**是同一个判据**，不会随 literal_eval 的接受范围漂移。

为什么不改上层 `except`：`extract_tool_calls` 的整轮容错是有意设计（模型输出本身就是不可信的），
改它要回答「部分调用是否该保留」，属语义变更，超出本 bug。

## 五、验证

| 项 | `954bd28f` 未修 | `954bd28f` + fix.patch | `ccf60061`（#8560）原样 | `ccf60061` + on-pr8560.patch |
|---|---|---|---|---|
| 新增/合并测试 | 5 failed / 5 passed | **10 passed** | 3 failed / 7 passed | **10 passed** |
| 探针 | exit 0 | exit 2 | exit 0 | exit 2 |
| `ruff check` / `format --check` | — | 通过 | — | 通过 |

日志：[pytest-before](../../experiments/verl-8559-qwen3-ellipsis/raw/pytest-before-fix.txt)、
[pytest-after](../../experiments/verl-8559-qwen3-ellipsis/raw/pytest-after-fix.txt)。
在 `ccf60061` 上跑出 `3 failed` 那一组，是把本目录的测试文件替换上去、源码保持 PR 原样得到的——
它证明**差异来自判据的一般性，而不是测试写法**。

## 六、对外状态（未执行）

| 动作 | 状态 |
|---|---|
| 在 #8559 补评论（指出 set/bytes/complex 同类） | **未执行**，待用户授权 |
| 在 #8560 留 review 建议（改成通用判据） | **未执行**，待用户授权 |
| 在用户 fork 上另开分支推同类修复 | **未执行**，待用户授权（按仓库铁律，推已有 PR 的 head 分支同样算对外动作） |
| 直接改 #8560 的代码 | **不做**，除非用户明确同意 |

草稿见 [review 草稿](../../open-source/verl/evidence/qwen3-ellipsis-review-draft.md)。

## 七、核验信息

- 源码版本：`954bd28f`（fork main，2026-11-05）、`ccf60061`（PR #8560 head）。
- 环境：Python 3.12.13（uv venv）、pytest、ruff；**CPU only**，未装 vLLM/SGLang，未跑训练，未连 GPU。
- 未验证：真实 Qwen3-Coder 输出端到端复现；issue 作者日志中的 38 步 2 次是**其自述**，本轮未独立复核。
