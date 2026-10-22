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
- **CUDA 版本**：CUDA 13。安装文档面向 NVIDIA，不等于任意普通卡、显存或卡数都能运行每个配方；例如下述 HI3 双引擎路径另有每节点至少8设备的源码前置条件。具体型号兼容性与容量仍须实测。
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

## AR 与 diffusion 的源码统一边界

2026-10-22 首次源码阅读固定官方 main `6fd82cba13f52f394747832344039408699ea378`。结论分两层：**普通 AR 与 diffusion 共享类型和机制，但保留不同 trainer；HI3 联合模型路径则进一步共享训练骨干、backend 和联合更新。** “统一多模态 RL”不能简化成所有模型、采样器和损失都走同一个循环。

| 层次 | 源码入口 | 统一与保留的差异 |
|---|---|---|
| 独立 AR | `train_ar.py` → `ARTrainer` | rollout 后取最后的生成 Part 交给 stack；可按算法是否需要优势决定 reward/advantage |
| 独立 diffusion | `train_diffusion.py` → `DiffusionTrainer` | 可积累多个 rollout 的末端 Part 后交给一次 train_track；有独立 residency/采样窗口逻辑 |
| HI3 联合模型 | `train_unified_model.py` → `UnifiedModelTrainer` | 传完整 Sample 给 `UnifiedModelTrainStack`，保留 AR 与图像的共同 prompt 树 |
| 共用机制 | `Sample` / `Part`、`StageAlgorithm`、Hydra `_target_`、placement/backend | 统一数据与接线接口，不代表相同 logprob、loss、设备生命周期或数值行为 |

入口分别见固定版 [AR](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train_ar.py)、[diffusion](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train_diffusion.py) 与 [unified](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train_unified_model.py)。这些是不同入口构造不同 trainer，不是一个 mode 开关自动抹平领域差异。

### HI3 的样本树与信用分配

[`UnifiedModelTrainer._build_request_sample`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/unified_model.py#L205)从 P 条输入依次 fork N 个 AR 分支、每个再 fork M 个图像分支。driver 要求 `[input, ar_shell, image_shell]`，预期行数分别 P、P×N、P×N×M；两引擎路径先生成 AR recaption，再把每条 recaption 重复 M 次送入 DiT，最后填回原 shell。driver 保留原 shell 的 lineage，但接线与行数检查本身不认证引擎返回顺序和身份正确。

`train_step` 在 reward 后调用 `Sample.propagate_rewards(op="mean")`：它仅向尚无 reward 的父 Part 回传子节点均值，不覆盖已评分的父节点。随后 AR 按原 prompt 的兄弟分组算优势；M>1 时图像按 AR 父分支算自己的优势。**M=1 走显式共享优势路径**，检查两路 batch size 相等后将 AR advantages 赋给图像 Part，而不是让图像每个单成员组独立居中为0。[类型实现](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/types/sample.py)保留 parent/child ID、条件、segment、rewards 与 advantages；分组要求等分支、同父兄弟连续。

这里存在一处需要分清的证据：同版 [`hi3_vllmomni.yaml`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/examples/unified_model/hi3_vllmomni.yaml#L205)的旧注释仍称 N×M 是平坦 AR/图像对、真正两层需后续实现；但当前 driver 已显式构造两层及复用 recaption。**配方注释与 driver 接线描述不一致**；本轮只确认 driver 的构造、传参和计数检查，不据此宣布真实 engine 已正确保留全部身份或生成语义，也不自动修文档/提 PR。

### 两路损失怎样进入同一次更新

trainer 将同一 bundle 注入 pipeline 和 backend，再把同一 pipeline 注入两路算法。HI3 的 [`pipeline._assemble`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/models/hunyuan_image3/pipeline.py#L101)把同一 bundle 给 AR 与 diffusion stage，两者的 `trainable_module()` 都返回 `model.transformer.model`。配方选择 AR 的 GRPO 与图像的 FlowGRPO，配置单份 LoRA；不是用两个同配置但独立的模型分别训练后再合并。

[`UnifiedModelTrainStack._train_one_step`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/stack/unified.py#L145)顺序为 `zero_grad → AR backward → image backward → optimizer_step`。两路仍分别计算 loss scale/微批损失；只在至少一路报告 backward 时尝试一次 backend step。`train_track` 按 `num_updates_per_batch` 重复这个联合更新（默认1），所以“一次更新共同累积”不等于“任意配置下每 rollout 永远只有一步”，也不保证两路梯度同权或训练收益。

### 本轮检查与真机边界

只读 clone 后读取了仓库 AGENTS、入口、trainer 接线、联合 stack、Sample/Part 与 HI3 stage 共享对象；未安装 UniRL、下载权重或导入训练模块。Python 3.12.13 运行原静态依赖方向检查 `python -B lint/check_core_dependencies.py` 得到 `ok`；四个核心文件 AST/内存 compile 通过。该 guard 检查指定层的静态 imports、跳过 TYPE_CHECKING，不认证动态导入、服务或训练执行。

四文件 SHA256 可用标准库复查，不需要任何设备：

```bash
python -B - <<'PY'
import ast, hashlib
from pathlib import Path
for name in ('unirl/train_unified_model.py', 'unirl/trainer/unified_model.py',
             'unirl/train/stack/unified.py', 'unirl/types/sample.py'):
    data = Path(name).read_bytes()
    compile(ast.parse(data, filename=name), name, 'exec')
    print(name, hashlib.sha256(data).hexdigest())
PY
```

```text
train_unified_model.py   4b0c65de767bc8f55dbe55ae13f4004d9753d1111df4d07e08f6a005bf7cc19a
trainer/unified_model.py f9b47b971380deb854491b17ba3fa2cb6e4c80dcbf75e0d56541c2fa3cb3ab1c
train/stack/unified.py   d2da53dea5143e98f0228465cbde5cc90f014a3c6bf3b123de658354dfe95028
types/sample.py         04c7a8caa840953e16fea401676ad30f4f1080fdf494fcf1daf095a15d78a460
```

HI3 双引擎构造器明确要求每节点至少8设备，batch_size 为 num_devices 的正倍数；这不是已完成的资源测量，也不能外推为所有 UniRL 路径都至少8卡。此配方的真实两层 rollout、共享参数更新与同步均未执行，资源条件已记入 [M5 接续](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不据安装说明给单4090或昇腾承诺支持。

## 稀疏 SDE 与轨迹状态位置

**选3个 SDE 转移，不是只做3步去噪，也不是只保存3份 latent。** 2026-10-22 沿用固定 `6fd82cba13f52f394747832344039408699ea378`，直接正常导入原 [`types/sampling.py`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/types/sampling.py)、[`sde/index_schedule.py`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/sde/index_schedule.py)及 require 模块。无 AST 摘取、包壳或替身；这些 leaf 模块只需标准库和 NumPy，本轮没有导入 torch/Ray/Hydra/OmegaConf。

与 HI3 配方相同，设总步数16、`timestep_fraction=[0,0.5]`、`num_sde_steps=3`。AllSDEScheduler 的候选池是半开区间 `[0,8)`，每个 rollout id 经独立 NumPy generator 无放回选3个。64个 id 的原函数输出均满足范围/数量/去重，逆序重查完全一致，NumPy 全局 RNG 状态未变；不要求不同 id 必选不同集合。当前 NumPy 2.0.2 下 rollout0 选 `[4,5,7]`。

`DiffusionSamplingParams.get_sde_indices` 的优先级是**显式列表（包括空列表）→ scheduler → 全部 num_inference_steps**。因此省略列表和 scheduler 时得到0至15；显式 `[]` 即使配了 scheduler 也返回空，`[1,2]` 则返回该列表。AllSDEScheduler 本身不设 num_sde_steps 时返回过滤池的全部8步，设0返回空；原构造器拒绝负数或超过池容量的请求。这是 helper 的解析契约，不是说直接传原始 None 给 stage 也会全选。

标准 [`DiffusionTrainer._build_request_sample`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/diffusion.py#L687)先解析列表，再以 `dataclasses.replace(..., sde_indices=..., scheduler=None)` 固定到本 rollout；unified trainer 同样如此。HI3 与 SD3 的 `diffuse` 则读这个已解析列表，在 `for i in range(T)` 中逐步设 `step_eta = params.eta if i in sde_set else 0.0`。所以源码仍有全部 T 次去噪迭代，非选中步走 eta0；这不等于声称恰好 T 次模型前向或量化了运行时间。

原 `compute_trajectory_positions` 给每个选中转移 i 保留相邻位置 i 和 i+1，再排序去重。HI3/SD3 采样循环还显式 `needed.add(T)`，保证最终状态可用于后续解码。下表端点列为原 helper 实测，最后一列仅按这条源码规则推导，未生成任何 latent：

| SDE 索引（T=16） | helper 端点 | 加终态后的预期位置数 |
|---|---|---|
| `[]` | `[]` | 1，仅位置16 |
| `[0,1,2]` | `[0,1,2,3]` | 5，包括16 |
| `[0,2,4]` | `[0,1,2,3,4,5]` | 7，包括16 |
| `[14,15]` | `[14,15,16]` | 3，终态不重复 |
| 全部0至15 | 全部0至16 | 17 |

穷举前半段8步的全部56种三索引组合，原 helper 端点均与独立集合表达式 `S ∪ {i+1}` 一致；加终态后5/6/7个位置分别有6/30/20组。原因是相邻转移共享中间状态，不能把存储位置数固定写成 SDE 数或2倍 SDE 数。[LatentSegment](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/types/segments/latent.py)也分开保存完整 sigma 表、状态位置 indices 与 sde_indices，三者不是同一轴。

空选择不是“自动继续原 FlowGRPO 训练”：[`FlowGRPO.compute_loss_and_backward`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/algorithms/flowgrpo.py#L116)对空 sde_indices 直接报告 has_backward=False；有选择时仍要求实际 transition logprob／replay 契约匹配。这里只核源码，没有执行 algorithm 或以索引个数保证 logprob 张量可用。

核心复查命令（在固定 UniRL 源码根目录，Python3.12.13 / NumPy2.0.2，无安装）：

```bash
python -B - <<'PY'
from collections import Counter
from itertools import combinations
import sys
import numpy as np
from unirl.types.sampling import DiffusionSamplingParams, compute_trajectory_positions
from unirl.sde.index_schedule import AllSDEScheduler
sched = AllSDEScheduler(16, [0,0.5], 3)
params = DiffusionSamplingParams(num_inference_steps=16, scheduler=sched)
before = np.random.get_state()
chosen = {i: params.get_sde_indices(i) for i in range(64)}
assert all(len(s)==3 and s==sorted(set(s)) and set(s)<=set(range(8)) for s in chosen.values())
assert all(params.get_sde_indices(i)==chosen[i] for i in reversed(range(64)))
after = np.random.get_state()
assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
assert DiffusionSamplingParams(num_inference_steps=16).get_sde_indices(0)==list(range(16))
assert DiffusionSamplingParams(sde_indices=[], scheduler=sched).get_sde_indices(0)==[]
assert DiffusionSamplingParams(sde_indices=[1,2], scheduler=sched).get_sde_indices(0)==[1,2]
hist = Counter()
for group in combinations(range(8),3):
    ends = compute_trajectory_positions(set(group),16)
    assert ends==sorted(set(group)|{i+1 for i in group})
    hist[len(set(ends)|{16})] += 1  # consumer 的 +T 规则，非执行采样
assert hist=={5:6,6:30,7:20}
assert not any(n in sys.modules for n in ('torch','ray','hydra','omegaconf'))
print(chosen[0], dict(sorted(hist.items())))
PY
```

三原文件 SHA256 分别为 require `d30f89e77ff72e9dddaaf8b8f0ab403790b3e2984fbd6afea28f6bcc60177238`、sampling `95325713e5f9d426ef56a2b84c0d7a2f701f0cd7c3a0f7f76b728d420a4f9ddc`、index_schedule `4a313b44df9275b8e470a1dd1226d3b835e5170eb4a707ab162c269fe307f3c4`。上面的核心复查及额外首尾/空/全步端点、0/全池和非法计数控制均通过；这是自编原模块检查，不是上游原测试。未执行 stage、engine、latent 构造、replay/backward 或性能/显存实验；位置数量不能外推节省比例、跨引擎数值一致或质量。需要真机时沿用 [M5 原条目](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不新增必跑矩阵。

## 下一步

继续以 UniRL 为高优源码/潜在贡献轨道，优先围绕真实配置与类型契约选题；不是已找到 bug 或获得对外发布许可。真机验证由 M5 独立选择获准且资源匹配的配方，条件不符时保留未测，不为复现默认 HI3 直接启动大模型或升级环境。
