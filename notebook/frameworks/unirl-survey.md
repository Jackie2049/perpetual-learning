# UniRL — 腾讯统一多模态 RL 框架调研

> 目标：调研腾讯 `Tencent-Hunyuan/UniRL`——是否开源、支持哪些硬件加速器、社区活跃度、**外部 PR 合入可能性**（用户核心关切：社区提的 PR 他们是否接受，还是只自己开发）。
>
> 调研时间：2026-10-22（GitHub API 实时核验）。
> 状态：**加入技术掘金高优关注项**（2026-10-22 用户定）。

## 一句话结论

> UniRL 是腾讯混元团队出品的 **Apache-2.0 开源、几乎日更的「统一多模态 RL」框架**（统一 AR 自回归 + diffusion 扩散模型的 RL），**目前只支持 NVIDIA GPU（Linux x86_64，CUDA 13），不支持昇腾 NPU**。**外部 PR 合入可能性高**：最近 100 个已关闭 PR 合并率 94%，且明确接受了来自 CMU/NTU/个人开发者的社区 PR——不是"只有腾讯自己在开发"。

## 项目概况

- **仓库**：`Tencent-Hunyuan/UniRL`（Tencent-Hunyuan 组织，腾讯混元团队）
- **描述**：UniRL is a Framework for **Unified Multimodal Model Reinforcement Learning**
- **官网**：https://unirl-project.github.io/unirl/
- **规模**：1023 stars / 102 forks / 30 contributors / 107 open issues
- **创建**：2026-06-08（较新）
- **语言**：Python
- **注意同名项目**：GitHub 另有 `showlab/UniRL`（VLM RL）、`G-U-N/UniRL`（ICML 工具箱），非腾讯出品，勿混淆。

## 是否开源 —— 是（Apache-2.0）

- LICENSE 明确："UniRL is licensed under the **Apache-2.0**"（Copyright 2026 Tencent）。
- GitHub 自动 license 检测显示 `NOASSERTION`，但那是文件头非标准模板所致，实际许可是 Apache-2.0，开源无疑。

## 硬件加速器支持 —— 只支持 NVIDIA GPU（Linux x86_64）

- **INSTALL.md 原文**："Python **>=3.12,<3.14**, Linux x86_64, **NVIDIA GPUs**"。
- README 全文**无任何 NPU / Ascend / 昇腾 / 910 提及**（精确搜索确认）。
- 依赖的推理/rollout 引擎：**SGLang、vLLM-Omni、fastvideo、VeOmni EP**——均为 NVIDIA CUDA 栈。
- **CUDA 版本**：CUDA 13（较新；但 README 未像 DeepInfer 那样限定具体 GPU 架构，普通 NVIDIA 卡应可跑，非 Blackwell-only）。
- **与硬件关注不重叠**：不涉及昇腾/910C（区别于 DeepInfer 的 Ascend 支持）。

## 活跃度 —— 非常活跃

- **几乎每天提交**：最近 30 天持续高频 commit（10-19/18/17/16/15... 各类 perf/feat/fix/refactor）。
- **正式 release**：v0.1.0（10-07）、v0.2.0（10-08）。
- **Issue 活跃**：最近 30 天新增 33 个 open issue。
- **内容广度**：支持统一 AR + diffusion，覆盖 Qwen3、MiniMax-H3、Hunyuan Video 等多模型，引入 DAPO、GSPO、DRPO、FlowGRPO 等算法。

## 外部 PR 合入可能性 —— 高（核心关切，已用数据核验）

用户关切：社区提的 PR 他们是否接受？数据结论：**接受，外部 PR 合入可能性高。**

1. **合并率高**：最近 100 个已关闭 PR 中 **94 个被合并（94%）**，仅 6 个关闭未合并。
2. **大量外部作者被采纳**（非腾讯核心 `CjhHa1`）：
   - `ruiling-smartbear` = **CMU（卡内基梅隆大学）** 的 Ruilin Gao
   - `chi030303` = **NTU（南洋理工大学）** 的 Tianqi He
   - `Ideny42`、`MinhaoLi0318`、`LingYi-Liang`、`Dmaner`、`inaniloquentee` 等个人开发者
3. **不是"只有自己开发"**：核心维护者 `CjhHa1` 贡献最多（90），但 contributors 前 15 混着大量不同账号，社区 PR 高频被合。
4. **贡献门槛友好**：被合 PR 多为 `fix(...)`/`docs(...)`/`chore(...)`/`perf(...)`——**修复具体 bug、文档、小优化易被接受**，符合"修复具体 bug 比大幅优化更可能被合入"的规律。

## 定位判断（给后续使用）

- **硬件**：仅 NVIDIA GPU（Linux/CUDA 13），不涉及昇腾——和 910C/昇腾关注不重叠。
- **价值**：作为**统一 AR + diffusion 多模态 RL** 框架，范围比 verl 广；外部 PR 合入友好度高。
- **建议定位**：**技术掘金高优关注项**，偏「学习 + 潜在贡献」；因其硬件范围（NVIDIA only），不与 DeepInfer（Ascend 支持）在硬件上冲突，但与 verl（同为 RL 训练框架）可能有研究视角重叠。

## 未验证项 / 下一步

- 以上基于 GitHub API + README/INSTALL 实时核验；**未 clone 源码精读**。
- 下一步：若深入，可 clone `Tencent-Hunyuan/UniRL` 做架构源码阅读，重点看其「统一 AR+diffusion」的训练栈如何实现（对比 verl），以及是否有值得贡献的修复候选。
