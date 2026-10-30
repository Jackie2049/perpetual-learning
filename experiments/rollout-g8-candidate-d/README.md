# rollout-g8-candidate-d — 批内轨迹长度分布 → sync makespan（引擎级实测）

**目的**：填补 G3 统一性能模型备忘（`notebook/speculative-rl-performance-model-memo.md`）轴 3「批内轨迹分布」的数据缺口——此前全部引擎级测量均为**固定长度×batch 的同质批**构造（候选 C/C2/DSD）。本实验在**同一批内混入不同长度**的请求，量化批内长度分布对**批 makespan（=sync RL 的 T_rollout ≈ max_i T_i 的引擎级代理）**的影响，并检验 EfficientRollout「短响应先完成→active batch 缩小→长尾受益」论断在本栈的成立性。

**日期**：2026-10-30（M5 收尾阶段第三十二批）。**硬件**：4090-1 GPU 5（单 RTX 4090 24GB）。**软件**：vLLM `0.24.0+cu129`（anchor-drift，见 anchor.json）、Qwen2.5-1.5B-Instruct、gsm8k 派生提示词、temp 0.7/top-p 0.9、batch=8、max_model_len=4608（MIX 含 in=4096+128；与候选 C/B 的 4096 不同，同实验内一致即可，跨候选不直接比数值）。

## 矩阵与结果（summary.json，16 组合 = 4 comp × 2 mode × 2 iter）

| comp | 批内构成 | ar makespan (s) | spec makespan (s) | spec/AR |
|---|---|---|---|---|
| H512 | 8×in=512（均值 512） | 0.805 | 2.284 | 2.84× |
| H2048 | 8×in=2048（均值 2048） | 1.065 | 2.708 | 2.54× |
| MIX | [512,512,1024,2048,2048,2048,4096,4096]（均值 2048，与 H2048 同均值同总量） | 1.065 | 2.903 | 2.73× |
| RX8 | 8×in=2048，max_tokens 逐请求 [64,64,128,128,128,128,256,256]（均值 128） | 1.059 | 2.682 | 2.53× |

## 关键发现（轴 3 引擎级量化锚点）

1. **prompt 长度异质在 AR 下不产生 makespan straggler 税**：MIX/H2048 = **1.000**（1.0646 vs 1.0647 s）——同均值同总量下，批内长度跨度 512–4096 的混合批与全 2048 同质批 makespan 几乎逐位相同。连续批处理把各请求按 token 步进齐头推进，批 makespan 由**总工作量/解码速率**决定而非每请求长度的上端分位数——sync RL 经典「straggler 支配 makespan」在 L2 引擎级被吸收，属 L3 调度/系统级现象。
2. **response 长度异质（rollout 更相关场景）也不征税**：RX8/H2048 = **0.994**（AR）——逐请求响应预算均值 128、跨度 64–256 的混合批略**短**于全 128 同质批（短轨迹先完成、释放 KV 槽）。EfficientRollout「active batch 缩小→长尾受益」论断在引擎级有微小正方向（约 −0.6%），但 b8 已能容纳全部请求、释放槽位不显著加速剩余请求。
3. **spec 模式下 prompt 异质有 +7.2% 交互成本**：spec MIX/H2048 = **1.072**（2.903 vs 2.708 s）——ngram 提议+验证结构与异质批的 prefill/调度串行化（spec 禁 async scheduling）交互，异质成本在 spec 下显现；response 异质 spec RX8/H2048 = **0.990**（同样微负）。
4. **A(p) 与长度分布交互**：MIX 的 p1 略降（0.186 vs H2048 0.199）但 p4/p5 约 2×（0.029/0.026 vs 0.016/0.016）——批内长上下文（4096）请求的 ngram 后缀匹配在深位更好；RX8 p4/p5 反而降（0.006/0.004，短响应预算请求 ngram 上下文收益少）。交互存在但不足以翻正净负吞吐。
5. **spec/AR 2.5–2.8× 全净负**：与候选 C（0.16–0.37 口径已 JIT 修正）、C2、DSD 同栈结论一致——弱提议器每验证轮收益不足，且 spec 首 token 延迟整体抬升（禁 async scheduling 串行化，MIX first_s 1.33–1.73s vs AR 0.05–0.36s）。
6. **方法学**：seed=0 两 iter 复现良好（wall 差异 <0.5%）；per-request `e2e_latency` 在 offline 模式不填充（`first_token_latency` 有）——T_i 分布需 serve-mode 扩展，本实验以批 makespan 与逐请求 first-token 作为证据。

## 口径与边界

- 引擎级批墙钟含 prefill（AR/spec 同进程同配置方向性可比）；A(p) 为引擎内建逐位置聚合（跨异质请求）；spec 禁 async scheduling（ngram 限制）。
- 与族Ⅱb 各篇（Heddle 1.4–2.3×、SortedRL TPS 等）不做数字比值（§13.20 边界维持）；本实验是 G3 备忘轴 3 的**引擎级批内分布参照**（批规模方向锚点之外新增长度分布方向锚点）。
- 结论限定：单栈单点（1.5B/单卡/短响应 128），「引擎级吸收 straggler」不推广为 L3 系统级无 straggler。

## 复现

```bash
# 服务器 4090-1（GPU 5），venv rollout-g8（cu129 基线）
scp run-candidate-d.py run-all-d.sh parse-candidate-d.py 4090-1:/home/zxw/venvs/
ssh 4090-1 'cd ~/venvs && nohup ./run-all-d.sh > run-all-d.out 2>&1 &'
# 完成后（16 组合，约 10 分钟）：
ssh 4090-1 'mkdir -p ~/venvs/candidate-d-results && cp ~/venvs/lat_d_*.stdout.txt ~/venvs/lat_d_*.stderr.txt ~/venvs/candidate-d-results/ && cd ~/venvs/candidate-d-results && python3 ~/venvs/parse-candidate-d.py .'
```

原始结果（32 文件 lat_d_\<mode\>_\<comp\>_i\<iter\>.{stdout,stderr}.txt）与 summary.json/anchor.json 位于本目录 `raw/`，sha256 与服务器 `candidate-d-results/` 逐文件全匹配（32 文件集哈希 c4b602d2…、summary 73119a97…、anchor 92b231b5…）。
