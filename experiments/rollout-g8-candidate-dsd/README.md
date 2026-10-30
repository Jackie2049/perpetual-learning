# Candidate DSD — ngram-spec with Dynamic K schedule (engine-level)

**目的**：补 G3 备忘（[speculative-rl-performance-model-memo.md](../../notebook/speculative-rl-performance-model-memo.md)）§4 仍缺项「DSD 动态 K 行为（未测）」——vLLM 0.24.0 的 Dynamic Speculative Decoding（`num_speculative_tokens_per_batch_size` 批大小→K 调度表）在 ngram 弱提议器上的引擎级行为：动态下调 K（小批 K=2/3）与候选 C 固定 K=5 的吞吐/接受率对比。

**环境**：4090-1 GPU 5、venv `rollout-g8`（torch 2.11.0+cu129、vllm 0.24.0+cu129）、Qwen2.5-1.5B-Instruct、gsm8k 多样提示词、temp 0.7/top-p 0.9、max_tokens 128——与候选 C/C2 完全同栈。

**调度表**：`num_speculative_tokens_per_batch_size=[(1,8,2),(9,16,3),(17,64,5)]`——运行时批大小 b8→K=2、b16→K=3、b32+→K=5（DSD 文档主张长尾请求 K 上调；本实验验证小批下调 K 的效果）。

**执行**（2026-10-30 06:16–06:25 UTC）：先跑预热组合（in512 b8，不计时）使 Triton JIT kernel（`_topk_topp_kernel`/`_compute_slot_mapping_kernel`）编译完成，再跑 4 个计时组合（in{512,2048}×b{8,16}），全部 rc=0、无失败。另对 in512 b8 独立复跑 2 次（466.6/466.5/461.3 tok/s，±1% 内稳定）。

**结果（JIT-warm，spec/AR 以候选 C AR 基线为分母）**：

| 组合 | 动态 K | wall_s | decode tok/s | spec/AR | Mean acc len | A(p) |
|---|---|---|---|---|---|---|
| in512 b8 | 2 | 2.19 | 466.6 | 0.361 | 1.38 | [0.254, 0.123, 0, 0, 0] |
| in512 b16 | 3 | 2.31 | 885.8 | 0.281 | 1.36 | [0.221, 0.090, 0.053, 0, 0] |
| in2048 b8 | 2 | 2.45 | 418.5 | 0.310 | 1.17 | [0.131, 0.042, 0, 0, 0] |
| in2048 b16 | 3 | 2.69 | 761.3 | 0.277 | 1.23 | [0.148, 0.060, 0.023, 0, 0] |

**对照候选 C（固定 K=5，JIT 修正后）**：in512 b8 476.7（0.368）、b16 865.6（0.274）、in2048 b8 414.3（0.307）、b16 745.2（0.271）。

**结论**：
1. **动态 K 不改变 ngram 弱提议器的吞吐格局**：DSD（K=2/3）spec/AR 0.277–0.361 vs 固定 K=5 的 0.271–0.368，各组合差异 ±2% 内（in512 b8 -2.1%、b16 +2.3%、in2048 b8 +1.0%、b16 +2.2%）——小批下调 K 节省的 draft 计算量在此负载下被验证/前向结构成本淹没，仍全为净负（spec/AR < 1）。
2. **DSD 的 K 调度生效确认**：b8 组合 A(p) 只有前 2 位非零（K=2）、b16 前 3 位（K=3），引擎按运行时批大小正确取 K；Mean accept len 1.17–1.38 ≈ 固定 K=5 的 1.27–1.39 同量级。
3. **方法学修正（影响候选 C 归档）**：候选 C 归档 in512 b8 spec = 205.2 tok/s（wall 4.99s）是**首个 spec 组合的 Triton JIT 首编译污染**（`_topk_topp_kernel` 首次编译约 +2.8s）；JIT-warm 重跑同组合 = 476.7 tok/s（wall 2.15s）。候选 C 其余三组合（in512 b16 865.6／in2048 b8 414.3／in2048 b16 745.2）为后续 JIT-warm 组合，值有效。**轴 3 批规模锚点 in512 b8 的 0.159 应修正为 ~0.368**（与 b16 0.274 形成 b8→b16 小批串行化解除→平台缓降的新形态）；G9 负项表 spec 下限 205.2 应修正为 ~476.7。候选 C 归档 summary 的 in512 b8 spec 值保留原档、本 README 与专项文档加修正注记。
4. **A(p) 两态性复见**：DSD in512 b8 A(p1)=0.254 与候选 C 归档 0.207／重跑 0.269 均在批 29 已记的 ±0.03–0.06 单次方差带内——继续支持"候选 C 为单样本、精确值带方差注记"。

**归档文件**：raw/ 8 文件（4×stdout+4×stderr，stdout 含引擎 SpecDecoding metrics 行）＋run.log（含预热、组合行与验证重跑）＋summary.json＋anchor.json＋run-candidate-dsd.py＋run-all-dsd.sh。sha256 见 raw/ 下逐文件（本地与服务器 `~/venvs/lat_dsd_*` 拉取时一致）。

**执行/验证边界**：全部为本机受控 GPU 测量（GPU 5，结束释放回 1MiB），无上游写操作（铁律）；未下载新权重（EAGLE/Medusa 确认 HF 无现成 1.5B 权重，备忘 §4 判断继续成立）。
