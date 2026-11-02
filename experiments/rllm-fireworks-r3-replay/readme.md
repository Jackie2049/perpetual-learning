# rLLM Fireworks R3 承载复现

<!-- contribution-summary:start -->
**【仓库】** [perpetual-learning](https://github.com/Jackie2049/perpetual-learning)（AI 技术探索与贡献引擎）

**【贡献点】** 复现 rLLM Fireworks 后端默认 `router_replay: R3` 时路由矩阵装不进 `tinker.ModelInput`，作为 B 类候选（TM-010）的可执行证据。[取证结论](../../open-source/rllm/fireworks-r3-routing-matrices.md)

**【类型】** 工具（最小复现脚本）

**【价值评估】** 6 / 10 — 脚本能在真实源码上自证缺口并在上游修好后返回非零；但只覆盖本地数据路径，不覆盖 Fireworks 服务端行为。

**【进展】** 已对 rllm main `4e828c8` 跑通（exit 0＝缺陷在），tinker 三个版本现象一致。

**【大白话讲明白】**

这份脚本问一件事：rLLM 号称把推理时的 MoE 选路"回放"到训练里，那份选路数据最后有没有一个地方可以放？脚本拿真实代码走一遍——采到数据、存进中间对象、转成训练数据包——然后检查训练数据包上到底有没有那个格子。没有，脚本就以 0 退出。

**【What–Why–How】**

- **What：** `repro.py`（两阶段）与 `carrier_check.py`（承载能力单测）。
- **Why：** 静态读源码只能证明"代码看起来这么写"，不能证明"运行时真的塞不进去"；用真实类跑一遍才能区分"文档与实现不一致"和"结构上不可能"。
- **How：** 阶段 A 用真实 `ModelOutput → Step → trajectory_to_datums`；阶段 B 用真实 `FireworksEngine` 配打桩 sampler，覆盖 `logprobs.content[*].routing_matrix` 的解析路径；承载检查单独测 `tinker.ModelInput` 的声明字段、工厂函数参数与赋值行为。上游一旦支持该字段，两个脚本都返回 1。
<!-- contribution-summary:end -->

## 文件

| 文件 | 作用 |
|---|---|
| `repro.py` | 两阶段复现；`exit 0` 表示缺陷仍在 |
| `carrier_check.py` | `tinker.ModelInput` 能否承载 + 两路径载荷类型对照；`exit 0` 表示缺口仍在 |
| `run-info.json` | 快照、环境、依赖、断言清单、已排除项与未验证项 |

## 怎么跑

```bash
uv venv -p 3.12 /tmp/rllm-r3-venv
uv pip install --python /tmp/rllm-r3-venv/bin/python \
  "tinker>=0.22.2" "tinker-cookbook>=0.4.1" "fireworks-ai[training]==1.2.20" \
  pydantic PyYAML omegaconf hydra-core codetiming numpy openai
# 在 rllm @ 4e828c8 的 checkout 根目录
PYTHONPATH=. /tmp/rllm-r3-venv/bin/python <this dir>/repro.py
PYTHONPATH=. /tmp/rllm-r3-venv/bin/python <this dir>/carrier_check.py
```

两个脚本都不需要 GPU、不需要 Fireworks 凭据；`fireworks-ai` 只为 import `FireworksEngine` 用。

## 记录下来的输出（2026-11-03，tinker 0.36.0）

```
[A] router_replay=disabled -> None
[A] R3 datum len=6 routing_matrices=['', '', '', 'm0', 'm1', 'm2']
[A] datum input type = tinker.ModelInput
[A] element types     = ['str']
[A] ModelInput declared fields = ['chunks']; can carry routing_matrices = False
[B] engine ModelOutput.routing_matrices = ['m0', 'm1', 'm2']
[B] workflow Step.routing_matrices       = ['m0', 'm1', 'm2']
[B] R3 datum routing_matrices            = ['', '', '', 'm0', 'm1', 'm2']
stage A defect: True   stage B defect: True
```

## 边界

- 打桩的部分：sampler（模拟 `DeploymentSampler.async_completions_stream` 的返回形状）、tokenizer、`parse_completion`（基类未实现，真实 tokenizer 会解析到家族解析器）。**矩阵提取本身走的是真实 `fireworks_engine.py` 代码。**
- 未跑真实 Fireworks 训练；服务端如何处理未声明的属性未验证。
- 曾怀疑"多步累积时矩阵布局错位"，实测两个形状都对齐，已排除，不写进结论。
