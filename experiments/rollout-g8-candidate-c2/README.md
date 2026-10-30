# Candidate C2 — G3 三轴补测：批规模扩展（b32/b64）＋强提议器（draft_model 0.5B）实测

承接候选 C（experiments/rollout-g8-candidate-c/，ngram A(p) 8 组合）的 G3 统一性能模型备忘
（notebook/speculative-rl-performance-model-memo.md）补测，2026-10-30 M5 收尾第二十九批。

## 目的
- **轴 3（批内分布/batch scaling）**：候选 C 只有 b8/b16，补 b32/b64 观察 spec/AR 批规模趋势
  （批规模能否挽救弱提议器净负）。
- **轴 1（提议器强度）**：ngram 是零成本弱提议器，补一个"学习式提议器"——vLLM
  draft_model（Qwen2.5-0.5B-Instruct 作为 drafter），对比 E[L]/A(p)/吞吐。

## 矩阵（16 组合）
- {ar, spec-ngram} × in{512,2048} × b{32,64}（8 组合，轴 3；b8/b16 用候选 C 归档）
- {ar, spec-draft} × in{512,2048} × b{8,16}（8 组合，同会话 AR 对照）

## 关键结果（详见专项 §13.15 G3 / §13.16 G9 / §16 第二十九批）
1. **提议器强度**：draft E[L]=4.12/4.13（in512）·2.48/2.63（in2048）≈ ngram 的 1.95–2.97×；
   A(p1) 0.80–0.82 vs ngram 0.14–0.21。draft 曲线单调缓降（0.82→0.49），无 ngram 自增强形态。
2. **吞吐净负维持**：draft spec/AR = 0.20–0.25（b8/b16），ngram = 0.16–0.31（b8–b64）——
   **强提议器 E[L]≈4.1 在本栈小批仍未转正**；反推 drafter 前向有效成本 c_draft/c_t≈1.1–2.3
   （非参数比的 1/3——小批内存受限＋双模型图切换串行），盈亏需 E[L]≈11.5–17.6。
3. **批规模（轴 3）**：ngram spec/AR in512 0.16→0.27→0.27→0.24、in2048 0.31→0.27→0.27→0.23
   （b8→b16 小批串行化解除、b16 后平台/缓降）；AR 本身近线性扩批（b8→b64 5.9–6.9×）——
   **批规模不挽救弱提议器**。
4. **方法学（方差）**：seed=0 引擎跨进程非 bit 确定；单次 A(p) p1 绝对方差 ±0.03–0.06
   （ngram in512 b8 观测 0.207/0.269 两态）；候选 C 归档值为单样本，批量 28 定量结论
   形状/方向稳健、精确值带方差注记。

## 文件
- raw/：32 文件（16 组合 × stdout/stderr）＋ summary.json（字符串 key）
- anchor.json：版本/提议器/口径锚点
- run-candidate-c2.py / parse-candidate-c2.py / run-all-c2.sh（服务器 /home/zxw/venvs/rollout-g8-candidate-c2/ 同款）
