# rollout-g8-candidate-c — G3 spec-for-RL 三字段采集（候选 C）

> 专项：[notebook/rollout-performance-research.md](../../notebook/rollout-performance-research.md) §13.15 G3 / §14 候选 C。
> 执行时间：2026-10-30 03:34–03:45 UTC（本地 11:34–11:45 CST），单 4090（GPU 5），约 11 分钟（8 引擎进程，torch.compile 缓存复用）。
> 结果账在专项 §14 第二十六批完成账；本目录为原始数据与复现脚本。

## 目的

候选 C 登记目标：为 G3 统一性能模型备忘（speculative-rl-performance-model-memo.md）采集**引擎级实测三字段**——接受率-位置曲线 A(p)（主）/ 批内分布 / 版本差（引擎级不可得，记 N/A）。本实验用 vLLM V1 内建 ngram proposer 测出**首个引擎级逐位置接受率曲线**，并给出 ngram 提议器在同栈同负载下的吞吐负项对照。

## 配置（anchor.json 有完整漂移记录）

- 引擎：vllm `0.24.0+cu129`（锚点偏移，登记锚点 eaf1627c 2026-10-27 main；6033 commits 漂移，同 V1 引擎代际，decode 吞吐不可跨版本外推——§14 第二十一批量化）
- 模型：Qwen2.5-1.5B-Instruct（/home/zxw/models）
- 硬件：4090-1 单卡（CUDA_VISIBLE_DEVICES=5），gpu_memory_utilization=0.40
- 提议器：ngram（prompt_lookup_max=5/min=1），num_speculative_tokens=5，零 drafter 下载
- 提示词：gsm8k train 前 200 题连接截断（离线读 HF 缓存 parquet，in=512 为 in=2048 前缀；回退随机 token）
- 矩阵：input_len {512,2048} × batch {8,16} × mode {ar, spec}，max_tokens=128，temp=0.7/top_p=0.9
- 采样计数：`disable_log_stats=False` + 生成后 `llm_engine.do_log_stats()` 强制 flush（offline LLM 默认 disable_log_stats=True，批 26 冒烟实测发现）

## 结果（summary.json 全量；decode = 批墙钟含 prefill，方向性口径）

| 组合 | A(p)（位置 1–5） | Mean accept len | 接受吞吐 tok/s | 批墙钟 decode tok/s |
|---|---|---|---|---|
| in512 b8 spec | 0.207, 0.088, 0.047, 0.034, 0.014 | 1.39 | 42.7 | 205.2 |
| in512 b16 spec | 0.212, 0.079, 0.045, 0.037, 0.019 | 1.39 | 172.1 | 865.6 |
| in2048 b8 spec | 0.144, 0.053, 0.031, 0.024, 0.022 | 1.27 | 72.2 | 414.3 |
| in2048 b16 spec | 0.148, 0.071, 0.045, 0.034, 0.028 | 1.33 | 151.9 | 745.2 |
| in512 b8 ar | — | — | — | 1294.6 |
| in512 b16 ar | — | — | — | 3156.1（15/16 全长度，1 条 EOS 早停，wall 口径约 +6% 高估） |
| in2048 b8 ar | — | — | — | 1348.9 |
| in2048 b16 ar | — | — | — | 2746.8 |

## 解读要点（回专项 §14/§13.15 G3 账）

1. **A(p) 逐位置衰减**：p1≈0.14–0.21 → p5≈0.014–0.028，Mean accept len 1.27–1.39 贴合经典 iid 界 1/(1−ᾱ)；长上下文（in2048）整体接受率低于 in512（内容更多样、ngram 前缀可预测性下降）。
2. **ngram 提议器在同栈为吞吐净负项**：接受率 ~5.5–10% 下 spec 批墙钟 205–866 tok/s 显著低于同配置 AR 1294–3156 tok/s（且 spec 禁用 async scheduling、prefill/decode 串行化）。draft 侧吞吐 543–2186 tok/s 接近自由，但低接受率使每验证轮收益不足——与 G3 模型轴 1 预测一致（ᾱ 低 → 1/(1−ᾱ) 大 → 无净收益）。**本负项记 G9 型证据**（提议器选择决定 spec-for-RL 收益方向；ngram=零成本弱提议器，不可推广为系统级结论）。
3. **口径边界**：A(p) 为引擎内建批量聚合（metrics.py `num_accepted_tokens_per_pos`，跨请求按步聚合），是引擎级参照而非某篇论文的轨迹级曲线；批墙钟含 prefill 且 AR/spec 同进程同配置可比（方向性）；版本差字段引擎级不可得（需跨权重 RL 采样，记 N/A）。
4. **控制验证**：批 26 冒烟阶段以循环重复句提示词测得 A(p)=1.000×5（平凡 100% 接受）——验证测量管线端到端正确；正式矩阵改用 gsm8k 多样文本后为非平凡曲线。

## 文件

- `run-candidate-c.py` / `run-candidate-c.sh` / `parse-candidate-c.py`：测量/编排/聚合脚本（服务器 `/home/zxw/venvs/` 同款）
- `raw/`：19 个原始文件（16×lat_{mode}_in{il}_b{batch}_{stdout,stderr}.txt + summary.json + anchor.json + run.log），sha256 与服务器逐文件一致
- 冒烟日志 `/tmp/candc-smoke*.txt`（服务器临时目录，未归档；关键数字已入账）

## 复现

```bash
# 服务器（4090-1，/home/zxw/venvs/，venv rollout-g8）
bash run-candidate-c.sh   # 8 组合约 11 分钟（compile 缓存热后）
```
