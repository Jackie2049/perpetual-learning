# DeepSeek 开源生态活跃度与 PR 采纳调研（2026-10-20）

> 目的：评估在 DeepSeek 开源项目上提优化 PR 被采纳的可能性。用户核心问题。
> 数据：GitHub API 实测（2026-10-20）。结论仅限静态观测，不含真实实验。

## 一、DeepSeek org 全景（按 stars 排序的代码/infra 类）

| 项目 | 定位 | Stars | Forks | 最后 push | 近 30d commit | 全量 merged PR |
|---|---|---|---|---|---|---|
| deepseek-harness | 评测 harness | 255k | 30.8k | 10-17 | 活跃 | - |
| FlashMLA | MLA 内核 | 13k | 1.2k | 10-15 | 2 | **4** |
| DeepEP | EP 通信 | 10k | 1.5k | 10-16 | 3 | **19** |
| DeepGEMM | BLAS kernel | 9k | 1.4k | 10-20 | 3 | **18** |
| 3FS | 分布式文件系统 | 10k | 1.1k | 05-07 | 0 | 停滞 |
| DeepSpec | speculative decoding 全栈 | 7.2k | 676 | 07-09 | 0 | **停滞** |
| TileKernels | tilelang kernel 库 | 2k | 206 | 10-14 | 4 | **1** |
| DualPipe | 双向流水并行 | 3k | 335 | 01-14 | 0 | 停滞 |
| EPLB | EP 负载均衡 | 1.4k | 211 | 03-24 | 0 | 停滞 |
| DeepSelect | TopK kernel (DSA) | 497 | 46 | 09-30 | 低 | - |
| DeepJIT | xPU kernel JIT | 392 | 48 | 10-19 | 低 | - |
| DeepGEMM-Ascend | NPU GEMM | 591 | 52 | 10-14 | 低 | - |
| DeepEP-Ascend | NPU EP 通信 | 261 | 35 | 09-30 | 停滞 | - |

模型仓库（V3/R1/Coder/VL 等）为权重发布，不涉及 PR 采纳，排除。

## 二、PR 采纳核心发现（回答"优化 PR 能否被采纳"）

**结论：能做，但门槛高、周期长、且"优化类 PR"（尤其新增 kernel/重写）极难被采纳；"修复既有 bug/具体技术问题"才有较大概率。**

### 2.1 三个 kernel 库的合并 PR 作者分布（历史）

- **FlashMLA**（全量 merged=4）：janeyx99（PyTorch 成员，ABI 稳定）、PerryLink（128-bit PTX 修复）、interestingLSY（Ascend 950 release）、内部成员 release。→ 外部贡献极少，且多是知名/高价值小修复。
- **DeepGEMM**（merged=18）：Lucifer1004、AichenF、zhou9402、lfr-0531、zheanxu 等，含 SM100/SM90 kernel、MegaMoE 支持——外部/合作者贡献相对多。
- **DeepEP**（merged=19）：VesVlad、janbernloehr（DMA-BUF fallback）、abcdabcd987、Autumn1998 等——**外部 PR 采纳最活跃**。

### 2.2 外部优化 PR 的待遇（核心警示）

抽查 3 个"外部优化 PR"，全部 **0 comments、0 review_comments、多周无回应**：
- FlashMLA #233：sm86 MLA decode（RTX 3090）高价值实现，10-13 开，0 回应。
- FlashMLA #227：cb521 优化 prefill/FP8 decode，09-24 开，0 回应。
- DeepGEMM #474：SM89 FP8 Tensor Core 支持，10-14 开，0 回应。

### 2.3 open PR 平均年龄（= 响应延迟）

- **FlashMLA：185 天**（open PR 平均躺半年，47/50 超 14 天）
- **DeepEP：52 天**（41/50 超 14 天）
- **DeepGEMM：44 天**（43/50 超 14 天）

### 2.4 维护模式

- 核心维护者：DeepSeek 内部成员（FlashMLA 的 Shengyu Liu/Zeyu WANG/Jiashi Li/Perkz Zheng 等）。
- **"release 式"维护**：不定期批量同步内部代码（如 DeepGEMM 的 "Public release 10/14"），非持续响应社区。
- **无 CONTRIBUTING.md**（FlashMLA/DeepGEMM），社区贡献指引缺失。
- 被采纳的外部 PR 特征：**技术性强的小修复**（PTX 类型、ABI 稳定、num_splits 扩展、DMA-BUF），而非大幅优化重写。

## 三、对我们"提优化 PR"的启示

1. **DeepInfer 之外**：这些 kernel 库是"优化 PR 难被采纳"的高风险区——除非我们做的优化是**具体 bug 修复**且能被维护者认可。
2. **相对友好的目标**：DeepEP 采纳最活跃（外部 PR 多）；DeepGEMM 有较多外部/合作者贡献。
3. **避开**：FlashMLA 对大幅 kernel 优化基本不回应（半年躺着）。
4. **策略**：若要在 DeepSeek 生态提 PR，优先 DeepEP（通信层，我们有 DeepEP-Ascend 积累）和 DeepGEMM 的具体修复；且预期周期长。
5. **对比 verl/vLLM/AReaL**：这些是社区驱动、外部 PR 采纳率高得多；DeepSeek 是公司内部驱动、外部 PR 边缘化。**从"PR 被采纳"角度，verl/vLLM/AReaL 性价比远高于 DeepSeek 生态**。

## 四、局限

- 静态 API 观测，未跟踪微信/内部渠道；DeepSeek 可能在 issue 里认领但未同步到 PR。
- "外部"判定依赖 author 是否 DeepSeek 成员（public members 仅查到 zdaxie，其余靠 commit 归属推断，可能低估内部成员比例）。
- 采样窗口为 API 可见范围，可能未覆盖全部 open PR。
