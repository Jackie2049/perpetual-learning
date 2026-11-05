# DeepSpeed pyflag 第九批：44 个开关 / 1 条候选；顺着它撞出一处 2023 年就存在的签名不匹配

快照 `7e58feb65f6bfb17671ef1b19dfbf123b569ad48`（2026-11-05），1167 个 `.py`，扫描 **29 秒**。

## 为什么选 DeepSpeed

前八批：SGLang／vLLM／LMDeploy／TensorRT-LLM／llama.cpp／MLC-LLM（推理六批）＋ AReaL（训练侧）＋ FastVideo（视频生成）。DeepSpeed 是训练与推理都覆盖的老牌框架，按扩池要求本轮只纳入它一个。

## 读数

**44 个开关 / 1 条 `silently_inert`。**

| 开关 | `mode` | 判定 | 理由 |
|---|---|---|---|
| `use_triton_ln` | always | `UNREACHABLE_DEAD_ELSE` | 三个函数的该形参都在形参表**末尾**，所有调用点按位置传且刚好停在它前一位 ⇒ 恒取默认 `True` |

DeepSpeed 的开关密度明显低于前面几批（1167 个文件只有 44 个），原因很可能是它的配置面主要在 **JSON 配置 + pydantic dataclass**（`deepspeed/inference/config.py`），而不是函数形参——pyflag 只认形参默认值。这条本身是方法读数：**pyflag 适合"形参即配置"的仓库，不适合 dataclass 驱动的仓库。**

## 真正有价值的发现不在扫描结果里

顺着 `use_triton_ln` 的唯一消费者往下读，在调用点上撞见一个更强的形态：**`fused_gemm_gelu` 的 8 个位置实参与 triton 版本 11 个必填形参不匹配**。

先 E2E 确认它真的会崩（4 档臂，A 臂 `TypeError`），再用 9 条可达性判据确认**用户撞不到**（`use_triton=True` 被限定为 BERT）。

详细分析与自错见 [证据页](../../open-source/deepspeed/evidence/triton-fused-gemm-gelu-argshift.md)。

## 四个探针

| 探针 | 判据 | selftest | 回答什么 |
|---|---|---|---|
| [probe_use_triton_ln.py](probe_use_triton_ln.py) | 17 | 5/5 | 扫描候选本身：该形参恒真、`else` 是死代码 |
| [probe_gelu_argshift.py](probe_gelu_argshift.py) | 9 | 5/5 | 两个签名对不对得上（AST + `inspect`） |
| [probe_gelu_e2e.py](probe_gelu_e2e.py) | 5 | 3/3 | 真实类体 + 真实函数执行，四档臂 |
| [probe_reachability.py](probe_reachability.py) | 9 | 4/4 | 这条路径在入口链上可不可达 |

合计 **40 条判据 / 13 条注入**，全部通过。

## 复现

```bash
git clone --filter=blob:none --no-checkout https://github.com/deepspeedai/DeepSpeed.git
cd DeepSpeed && git checkout 7e58feb65f6bfb17671ef1b19dfbf123b569ad48

for p in probe_use_triton_ln probe_gelu_argshift probe_gelu_e2e probe_reachability; do
  python3 $p.py --repo <repo> || echo "$p FAILED"
  python3 $p.py --repo <repo> --selftest || echo "$p SELFTEST FAILED"
done
```

**不需要克隆**：`raw/snapshot/` 有 10 个源文件逐字节副本，`--repo raw/snapshot` 四个探针读数相同（已实测）。

## 读数文件

| 文件 | 内容 |
|---|---|
| `raw/use-triton-ln.json` | 17 条判据 |
| `raw/gelu-argshift.json` | 9 条判据 |
| `raw/gelu-e2e.json` | 4 档臂 + 5 条判据 |
| `raw/reachability.json` | 9 条可达性判据 |
| `raw/selftest-*.txt` | 四组注入结果 |
| `raw/snapshot/` | 10 个源文件副本 |

## 边界

- **未安装 DeepSpeed、未上 GPU、未启用 triton**；"C++ 内核"是按 8 参数形态写的桩（未读 C++ 注册源码）。
- **未覆盖 DeepSpeed-MII**：若它绕过 `module_inject/containers` 的 BERT 限制，可达性结论可能变。
- 纯静态 + CPU 源码执行，**未触上游**。

## 结论口径

**对外 4/10，不对外。** 缺陷确凿但从未可达；写下它的理由是**方法**：见证据页第四节「先把'可达'和'存在'分开」与第五节 5 处自错。
