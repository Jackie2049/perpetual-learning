# llama.cpp pyflag 扩池（第五批）

对 [llama.cpp](https://github.com/ggml-org/llama.cpp) `3de52ac3` 跑一遍「配置项有没有生产者」，
结论是 **37 条候选、真缺陷 0 条**。本批真正产出的是 C++ 版扫描器和三类 C++ 特有的写入形态。

- 实验记录与读数：[README.md](README.md)（本文件）
- 证据页：[open-source/llamacpp/evidence/cpp-fields-triage-2026-11.md](../../open-source/llamacpp/evidence/cpp-fields-triage-2026-11.md)
- 扫描器：[tools/pyflag/scan_cpp_flags.py](../../tools/pyflag/scan_cpp_flags.py)

## 读数

| 项 | 值 |
|---|---|
| 快照 | `3de52ac38818978d6da8f386eda43a856e53714a`（2026-11-04 22:34 UTC） |
| C/C++ 文件 | 1574 |
| 有消费者且按它分支的字段 | 1636 |
| `unwritten` 候选 | 37 |
| 真缺陷 | **0** |
| Python 面的对照 | 250 个 `.py` → 7 个开关 / 0 条失效（没有信息量） |

档位分布见 `raw/classify.json`，三类 C++ 写入形态见证据页第四节。

## 脚本

| 文件 | 作用 | 怎么跑 |
|---|---|---|
| `classify.py` | 37 条候选逐条归因，每条给一句可复核的理由；有候选没档位就 `exit 1` | `python3 classify.py <llama.cpp 根>` |
| `verify_findings.py` | 15 条判据钉到具体文件行；`--selftest` 4 条注入全部要让目标判据变红 | `python3 verify_findings.py <root>` / `--selftest <root>` |
| `probe_reasoning_control.py` | 从快照抽 `common_sampler_init` 的守卫与 `common_reasoning_budget_apply` 真实源码，在 CPU 上执行；证明 `reasoning_control` **确实接线** | `python3 probe_reasoning_control.py <root>` / `--selftest` |

`raw/snapshot/common/` 是被引用的两个源文件的**逐字节副本**（`sampling.cpp`、
`reasoning-budget.cpp`），让 `--selftest` 在没有完整快照的机器上也能跑。

## 复现

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama-scan
cd <repo>
python3 tools/pyflag/scan_cpp_flags.py --selftest                     # 7/7
python3 tools/pyflag/scan_cpp_flags.py /tmp/llama-scan \
        --json experiments/llamacpp-pyflag-triage/raw/scan-cpp-llama.json   # 约 4 分钟
python3 experiments/llamacpp-pyflag-triage/classify.py /tmp/llama-scan
python3 experiments/llamacpp-pyflag-triage/verify_findings.py /tmp/llama-scan
```

## 边界

纯静态分析 + 在 Python 里执行抽出的源码片段。**未编译 llama.cpp、未起 server、未跑推理、未上 GPU/NPU、未触上游。**
扫描器是正则级结构分析，不做符号解析：同名不同结构体会串，报「没人写」仍需人工复核——
本批 37 条就是逐条看掉的（其中 4 条是同名局部变量）。
