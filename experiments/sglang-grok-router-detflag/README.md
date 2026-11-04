# SGLang：Grok 路由的确定性逃生口没有接线（2026-11-04）

## 一句话结论

SGLang `27aedf7f`（2026-11-04）的 `fused_moe_router_shim` 上有一个形参
`enable_deterministic_inference: bool = False`，源码注释说它是为了
"avoid non-deterministic behavior"——**但从来没有任何调用点传过它**，
所以 `--enable-deterministic-inference` 在 Grok1/Grok2 的 MoE 路由上不生效，
不报错也不打日志。

## 做了什么

1. **静态取证**（[repro.py](repro.py)）：把「形参在签名里且默认 False」「`grok.py` 用
   `functools.partial` 只钉住第一个位置参数」「`grok.py` 全文不提这个开关」
   「`topk.py:3239` 的唯一调用点只传四个无关关键字」逐条钉住。
2. **行为分叉**：把 `router.py` 里三个函数的**真实源码**取出，桩掉 triton 与内核启动，
   在 CPU 上执行原文件的真实分支——照 `topk.py` 的调用走 **tensorcore**，
   显式传 `True` 走 **cudacore**。⇒ 这个形参不是死代码，是唯一的逃生口。
3. **历史归属**：逐版本取 `router.py` 核对，形参是 `9b8ebb2798`（2025-10-09，
   标题为 "move more files under srt/utils"）那次顺带加的，**同一提交没动调用点**
   ⇒ 不是重构回归，是加的时候就没接。

## 怎么跑

```bash
curl -sSL -o sg.tgz https://codeload.github.com/sgl-project/sglang/tar.gz/refs/heads/main
tar xzf sg.tgz
python3 repro.py ./sglang-main --json    # exit 0 = 缺陷在
python3 repro.py ./sglang-main --selftest  # exit 1 = 判定会翻转
```

## 读数

| 项 | 值 |
|---|---|
| 静态判据 | **7/7 PASS** |
| 照 `topk.py` 调用走到的内核 | `tensorcore` |
| 同形状强制 `True` 走到的内核 | `cudacore` |
| `--selftest` | exit 1（结论失效） |

## 边界（必须一起读）

- **纯静态 + CPU**，未装 SGLang、未跑推理、未上 GPU。本机是 Apple Silicon，triton 只有
  Linux 轮子（`uv` 报 `manylinux_2_28_*`），所以「真机跑一遍看 logits 是否逐位漂移」
  这件事**做不了**，本目录不声称做过。
- 只影响 **Grok1/Grok2 且 `num_local_experts > 0`**；不影响其它模型的确定性
  （DeepSeek-V2/V4、GPT-OSS、Kimi-K3、mega_gate 都是直读执行配置让路）。
- **两个内核的实际数值差异未测量**，只证明「会走到不同内核」。
- 不知道维护者是否**有意**把这条快捷路径设计成「一直用快的」。

证据页见 [open-source/sglang/evidence/grok-router-detflag.md](../../open-source/sglang/evidence/grok-router-detflag.md)。
