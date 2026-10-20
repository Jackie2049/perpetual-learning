# AReaL 贡献轨道

> 上游：[areal-project/AReaL](https://github.com/areal-project/AReaL)（蚂蚁系，2026-10 核验仓库名；2026-10-20 快照 5.8k stars、135 open issues）
> 定位：用户方向偏好的第二条 RL 训练框架轨道（verl 已为主贡献轨道）。2026-10-20 活跃度调研中 AReaL 合并效率（91%）为国内六家最高。
> 状态与候选追踪见 [tracker.md](tracker.md)。源码/原理学习见 [notebook/frameworks/](../../notebook/frameworks/readme.md)。

**负责人（2026-10-20 用户明确，状态已同步）**：prompt_mean 候选曾由用户明确交 macbook-M5 Agent 统一负责；2026-10-20 用户恢复 VPN 并明确本会话继续后，已完成框架跑通 + 实测闭环（不构成 defect，见 [tracker.md](tracker.md) 与 [源码笔记 §10.10](../../notebook/frameworks/areal-architecture-source-reading.md)）。统一入口见 [交接说明](handoff-m5.md)。

## 服务器环境（GPU 4090-1，2026-10-20 实测可用）

**环境名/位置：AReaL 官方 uv venv —— `/home/zxw/areal/.venv`**（随上游 clone `/home/zxw/areal` 一起，非 conda 环境；AReaL 基线 `3b93679`）。

- Python 3.12.15；torch 2.9.1+cu129、transformers 5.3.0、sglang 0.5.10.post1、ray 2.55.1、megatron-core 0.19.0（venv 14G）。
- **启动 AReaL 必须 `PATH=/home/zxw/areal/.venv/bin:$PATH` 前置**（否则 sglang 子进程会用系统 python3，ImportError 失败）。离线跑用 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`；本地信任环境加 `AREAL_ALLOW_DEFAULT_ADMIN_KEY=1`；显存紧加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- **空闲 GPU**：0/1/7（2-6 常被其他用户占用 ~22-24G）。
- 复现入口：smoke 配置 `/home/zxw/areal_smoke_ppo.yaml`（Qwen2.5-0.5B + FSDP2 + sglang，2-3 卡）；实测脚本/日志见 [源码笔记 §10.10](../../notebook/frameworks/areal-architecture-source-reading.md)。
- **环境变更规则（用户 2026-10-20 明确）**：需要修改环境时，从该 venv **基础上新建环境**（`uv venv`/conda clone 或复制后调整版本），不直接改动 `/home/zxw/areal/.venv`，避免破坏已验证基线。

## 贡献规范（community/CONTRIBUTING.md）

- 需 **DCO 签名**：`git commit -s`（Developer Certificate of Origin）。
- 提交前安装 pre-commit（format/lint/commit message 检查），PR 标题 `type(scope): description`（如 `feat(engine)`、`fix(trainer)`）。
- 有 `good first issue`、`help wanted` 标签可作为入口信号（截至 2026-10-20 无 open 的 good-first-issue；help-wanted 仅 1 个大型 feature #1110）。
- 对外发布与提交按 [AGENTS.md](../../AGENTS.md#执行边界) 执行（需用户授权）。

## 当前结论（2026-10-20 扫描）

近期可行动 bug 均被 **reporter 本人同日 PR** 覆盖（#1786→#1787、#1730→#1732），与 verl 的「reporter 深挖≈reporter 已开 PR」规律一致。**当前无未认领、可行动的贡献空间**；`#1730` 与 verl `#7913` 同属 microbatch 不变性 bug 类，且同 reporter，我方 verl geo_mean 经验可复用作评论/review 依据。详见 [tracker.md](tracker.md)。
