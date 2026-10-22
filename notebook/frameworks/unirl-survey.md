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

## SFT 预取与数据消费游标

**预取下一批不推进消费游标，但消费当前批也不等于 optimizer 已成功更新。** 2026-10-22 固定 `6fd82cba13f52f394747832344039408699ea378`，用真实 JSONL 读取原 [`SupervisedDataSource`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/data/sft.py#L216)验证了前半句；后半句来自 trainer 调用顺序，不是模型故障实测。

`get_samples` 立即推进 epoch/position；`peek_next_batch` 暂时取出下一批与取完后的状态，再恢复公开游标，将结果留在 pending；`commit_peeked_batch` 才安装 pending 状态。state_dict 不保存 pending 内容，因此在预取后、commit 前记录状态，恢复时会重新读到该批，而不是跳过去。

5行训练数据 r0～r4、batch_size3、shuffle=False 的实测：

| 操作 | 取得的样本 | 公开 epoch / position |
|---|---|---|
| 直读第一批 | r0,r1,r2 | 0 / 3 |
| peek 下一批 | r3,r4,r0 | 仍0 / 3 |
| commit pending | 同一批 r3,r4,r0 | 1 / 1 |
| 从 peek 后保存的状态另建 DataSource 恢复再读 | r3,r4,r0 | 1 / 1 |

原实现对 pending 时再次 get、再次 peek，以及无 pending 的 commit 都抛 RuntimeError。shuffle=False/True 各10批与不预取的直读序列、游标逐批一致，跨越多个epoch，全局 Python random 状态不变；这是原数据源本地随机顺序的检查，不是分布式采样验收。

恢复还检查训练 manifest 的文件字节 SHA256 与 seed/shuffle。相同5行数量但内容变更、seed变更或shuffle变更，都被原 load_state_dict 拒绝；不是只核数据集长度。缺 fingerprint 的 legacy 状态则发警告后允许恢复，不能把兼容接受当身份已验证。另用3行验证集得到2+1两批，未改变训练游标或既有 pending。

[`SFTTrainer.train`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/trainer/sft.py#L201)先 get／commit 当前批，再 peek 下一批，然后才执行 train_step；循环没有按 result.optimizer_updates 回退数据游标。prefetch 的目的是准备下一批并保持保存游标指向已消费数据，不是 optimizer 提交协议。异常中断、非有限梯度跳步或“模型和游标共同恢复到同一步”仍需完整路径证据；本轮不将单个数据源控制说成 exactly-once 训练。

[复查源码与只读夹具](../../experiments/unirl-sft-cursor/readme.md)保存于本项目，不修改 UniRL：

```bash
python -B experiments/unirl-sft-cursor/verify.py /path/to/UniRL
```

Python3.12.13 / torch2.9.1，真实原 dataset/normalizer/Batch/DataSource。仅跳过 tensor 包初始化器中与本题无关的 Ray transport 汇总导入，未替换这些类或摘取方法。全部观察通过，不是上游原测试；无 trainer/worker预取、模型、optimizer、磁盘 checkpoint 或设备执行。本题不新增M5真机事项。

完整源文件 SHA256：data/sft.py `45dc13244d15a0168919e48e33f126b6c9f259f02587007f4adcbc2c24ed782c`；trainer/sft.py `de66b86676f81422a832ce560942fd22e239b2052a00059b685631661ea3d2c9`。后者仅读取，没有执行；脚本也核对这两个hash，以免不同源码混用结果。

## Packed 文本的样本轴与 token 轴

**packed 文本按 token 存储，重排却按样本选择；零 token 的样本仍是一行。** 固定 `6fd82cba13f52f394747832344039408699ea378`，2026-10-22 在真实 CPU 上运行原 [`TextSegment`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/types/segments/text.py)和 [`Batch`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/distributed/tensor/batch.py)，没有重写方法或用容器替身替代。

`TextSegment` 的 tokens/log_probs/loss_mask 是 PACKED 字段，平坦轴长度为各样本 token 数之和；status 是 CONCAT 字段，按样本行排列。`pack` 将每个 packed 字段的逐样本 tensor 列表拼成一维 tensor，同时建立 `[N+1]` 的 cu_seqlens，lengths 由相邻 offset 差得到。正确构造不只需要一个平坦 tokens 数组，还要保存样本边界。

三行长度 `[2,0,3]` 的实际结果：

| 操作 | 样本数 | tokens | cu_seqlens |
|---|---|---|---|
| pack | 3 | 100,101,300,301,302 | 0,2,2,5 |
| select `[2,0,2]` | 3 | 300,301,302,100,101,300,301,302 | 0,3,5,8 |
| slice `[1:3]` | 2 | 300,301,302 | 0,0,3 |

`select` 可重排、重复或选择空集；它按原 offsets 取每行 token 段，再重建 offsets，而非对平坦 token 直接应用样本索引。`slice` 同时截 token 范围并将 offset 起点归零。concat 时后续 offsets 要加上之前的 token 总数，不是样本数；相邻切片拼回后，tokens、log_probs、mask、status 和边界均应恢复。重复选择也真实重复 payload，不是共享前缀压缩。

[复查脚本](../../experiments/scripts/verify_unirl_packed_text.py)用4种三行布局 `[2,0,3]`、`[0,0,0]`、`[1,2,1]`、`[0,1,0]`，逐字段核数值、dtype、长度和 offsets：36次选择（含permutation/重复/空集/CPU索引tensor）、40次合法连续切片、16个concat切点与4次chunk拼回全部通过。另在不提供 status 时，3个零长度 tensor 仍得到 batch_size3、cu `[0,0,0,0]`；零 token 行不能据 token 总数丢掉。

入口保护也有范围：pack 拒绝直接传已打包 Tensor，并要求不同 PACKED 字段的逐行长度一致；3行chunk(2)拒绝不能均分；故意绕过pack、又没有offset的普通构造对象不能执行packed slice。4项拒绝控制符合预期；不是说pack校验了所有字段语义，也不将违反构造前提的输入称作正常训练缺陷。

[`ARSupervisedTrackBuilder.build`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/sft/track_builder.py#L316)确实用 TextSegment.pack 构建 tokens/mask，token-budget planner 最终调用 part.select(perm)。这两处仅源码接线核对；本轮没有执行 Part、planner、tokenizer、attention mask、TensorRef transport、模型前后向或训练，因此不从容器一致性外推注意力隔离、prefix共享、梯度等价或性能。

```bash
python -B experiments/scripts/verify_unirl_packed_text.py /path/to/UniRL
```

Python3.12.13 / torch2.9.1；只用一个真实__path__包壳跳过 tensor 包初始化器的 Ray transport 汇总导入，其余原类按源码加载。无Ray/Hydra/OmegaConf/worker-local transport导入、设备发现或环境安装。这是自编原容器检查，不是上游原测试；本题不新增M5硬件事项。

脚本核对完整文件SHA256：batch.py `a33122410e68c2295631bbc3e70ce4b69ba41081da7ed49abd06774640676e3c`；segments/base.py `f7b9ce0a6ae77bcdbde81a93b43096bdfe975368234753ef74ab6191d794ff32`；segments/text.py `9f58114b40f3949eea70fba0692fb96b746e3e47af0c94c7db99e314cf237b34`。

## 按域奖励的零分与失败标志

**0 分、评分失败与不适用的分量是三种不同状态。** 2026-10-22 固定 `6fd82cba13f52f394747832344039408699ea378`，在真实 CPU 上直接调用原 [`PerDomainRewardScorer`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/reward/local/per_domain.py)，配两个原 MCExactMatch 配置：graded_format_reward=True 与 require_answer_tag=True。它们用正则计算，不加载模型权重；successes 表示评分是否成功，不是答案是否正确。

四行有效请求得到以下原始响应；graded/strict 是本次配置的域名称，不是训练阶段：

| 行 / 域 | 生成文本（标准答案） | 主奖励 | successes | graded 分量 | strict 分量 |
|---|---|---|---|---|---|
| 0 / graded | A（A） | 0.5 | True | 0.5 | NaN |
| 1 / strict | A（A） | 0 | True | NaN | 0 |
| 2 / strict | `<answer>B</answer>`（B） | 1 | True | NaN | 1 |
| 3 / graded | `<answer>D</answer>`（D） | 1 | True | 1 | NaN |

严格配置下没有 answer 标签，得 0 分仍是成功评分。另加 unknown 域或缺 domain 的行，主奖励保留初始化的 0，但 successes=False，errors 说明没有配置对应 scorer；所有域分量均为 NaN。不能只看主奖励把失败解释成正常低分。路由器按域选取原请求行，再把响应散回原位置；排列 `[3,0,2,1]`、重复选择 `[3,1,3,0]`、空批和整体缺 metadata 的控制均通过。

分量的 NaN 则表示该域未为这一行评分，不应随意补 0 后平均。直接运行原 [`_tensor_stats`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/utils/wandb_metrics.py)：它过滤非有限值，四行有效请求的 graded 均值为 0.75，strict 为 0.5；故意先补 0 的对照分别变为 0.375、0.25。全 NaN 输入返回空统计。源码中的 rollout component 日志调用此 helper，但本轮仅执行 helper，没有运行 WandB、Part 汇总或训练日志链路。

**这不是可直接接入文本训练的配方。** 当前 router 的 preferred_input_kind 固定为 image，内层 MC scorer 是 text。本轮直接构造合法的文本 RewardRequest，验证的是 backend 路由契约；标准 [`RewardService.score_and_attach`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/reward/service.py)在调用 backend 前先要求 frontier 含所需生成模态，因此不能由本轮推出它接受 text-only frontier。服务源码还在收到响应后对 False successes 抛错，不是静默把未映射域的占位 0 用于训练；这不等于宣称它拒绝任意 NaN。以上服务接线仅读源码，未导入或执行。

[复查脚本](../../experiments/scripts/verify_unirl_domain_rewards.py)保存在本项目，不修改 UniRL：

```bash
python -B experiments/scripts/verify_unirl_domain_rewards.py /path/to/UniRL /path/to/pillow-deps
```

Python3.12.13 / torch2.9.1 / Pillow12.0.0。本机原 CPU 环境缺 Pillow，使用 uv --target 仅安装到临时目录 `/tmp/perpetual-unirl-reward.D8w73Z/deps`；未修改原 venv。复现时需自行提供 torch/Pillow，已有 Pillow 可省略第二参数，脚本不会自动安装。Pillow 仅为原类型导入依赖，未构造图像或加载奖励模型。

两个真实 __path__ 包壳分别跳过 tensor 包的 Ray transport 汇总初始化和 local reward 包的重型模型 scorer 汇总初始化；其余 Batch/Texts/Request/Response/router/MC 类及统计 helper 均从原模块加载，无摘取方法或假 scorer。MC 的空 _load_model 方法会由原构造器调用，model 保持 None；base_device 参数也不代表执行过设备加载。无 Ray/Hydra/OmegaConf/Transformers、服务/HTTP、设备发现、trainer、反向或 GPU/NPU 执行。这是自编原模块控制，不是上游原测试；不据此定性训练缺陷或新增 M5 必跑事项。

脚本锁定七个完整源文件 SHA256（包括仅读的 service.py）。关键 hash：per_domain.py `6aca28149c4dcc48b3f8e29e74ef7195daa997aa833daac778618d4f3b2b73f8`；mc_exact_match.py `452413e2c7db3cf722fd3a887f660a61e5a13513f015a539f103b0a32d1cf526`；wandb_metrics.py `e92b9c40ebab4b78759ad15e36db4a833f853f6b122db4e395577966cc150648`。本题收束，不堆 NaN/路由变体，也不将组件统计等同于服务验收。

## Token 预算与 update 成员

**token_budget 约束拼批代价，不是显存保证，也不是逐样本截断上限。** 固定 `6fd82cba13f52f394747832344039408699ea378`，2026-10-22 用原 [`TokenBudgetPlanner`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/stack/planner/packed.py)、Part、TextSegment 与 TextTokenCondition 做单进程 CPU 检查。规划器先按 num_updates 等分原行，再在每份内按总长度降序 first-fit 拼批，最后用 part.select 重排。原行不能被搬到另一份 update 来改善装箱；总行数不能整除 update 数时直接拒绝。

两种代价模型不能混用。sum 是各样本 `(prompt + response)` 之和；dense 是 `(max(prompt) + max(response)) × 样本数`。两行长度分别为 `(8,1)`、`(1,8)` 时，sum 为18，dense 为32，不是“每行总长最大值9再乘2”。预算18下，前者可同批，后者不可。这里是规划器的代价估计，不是 attention kernel 的实际分配量。

8行的 `(prompt,response)` 为 `[(8,1),(1,8),(2,2),(20,1),(1,8),(8,1),(2,2),(1,20)]`，num_updates=2、预算18。原条件 attention_mask 给出 prompt 长度，原 packed segment 给出 response 长度；实际输出如下：

| 代价模型 | 第一次 update 的原行分组 | 第二次 update 的原行分组 | 各微批代价 |
|---|---|---|---|
| sum | `[3] [0,1] [2]` | `[7] [4,5] [6]` | 21,18,4 / 21,18,4 |
| dense | `[3] [0] [1] [2]` | `[7] [4] [5] [6]` | 21,9,9,4 / 21,9,9,4 |

21-token 单样本仍被放进独立微批。first_fit_decreasing 只在尝试加入已有微批时检查预算，找不到位置就新建单样本批，不切割或拒绝超预算样本。因此“配置预算18”不能推出每批代价≤18，更不能推出不会 OOM。两种模型下原 sample_ids、文本、metadata、prompt 条件与 response tokens 均按相同排列对齐，每行只出现一次，原 Part 不被重排修改。

有有效逐样本长度时，将 micro_batch_size 从1改999不改变本例 token 规划。若 segment 不提供长度，planner 则警告并回退原 CountPlanner：本例8行、两次update、micro_batch_size3得到 `[0:3,3:4]` 与 `[4:7,7:8]`，此时不再按 token 预算拼批。读取配置名还不够，须确认实际输入提供了哪种长度。

[`_sync_micro_count`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/stack/planner/packed.py#L93)在多进程组内用 MAX 对齐微批数量，必要时调用 balance_into_k；本次真实调用的是未初始化 process group 的原 no-op 分支，没有替换同步函数或启动通信。此处数量对齐不证明跨 rank 预算或梯度等价。[`TrainStack`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/stack/single.py#L322)将外层 plan 分别交给 _run_update；实际 optimizer_step 还取决于 backward 等条件，计划两份不等于实测完成两次更新。这段训练接线仅核源码。

[复查脚本](../../experiments/scripts/verify_unirl_token_budget.py)：

```bash
python -B experiments/scripts/verify_unirl_token_budget.py /path/to/UniRL /path/to/pillow-deps
```

Python3.12.13 / torch2.9.1 / Pillow12.0.0，复用上一节隔离 Pillow 依赖，无新安装。脚本使用真实路径包壳，仅跳过 tensor 包的 Ray transport 汇总初始化和 stack 包的训练模块汇总初始化；原 planner 包、Part 与条件/segment 均正常加载，无假类、方法摘取或规划器改写。两种分批、成员/字段对齐、数量回退及三项配置拒绝控制通过；这是自编原模块检查，不是上游原测试。没有模型、训练、通信 collective、设备发现或吞吐/显存测量，不新增 M5 必跑事项。

脚本锁定七个完整源文件 hash，包括仅读的训练接线。packed.py SHA256 `b5f2789f768cbdb7d766bebb5c1d9feb316dbe32db52637e3683e8e6dc8fe70f`，types.py `d9278a269104724a63abae6eceedde694ad92b1016c415f1b88ed087dcb3e8be`；其他见脚本。超预算独占批符合原实现说明，不据此提出缺陷或自动改规划策略。

## EMA 衰减方向与更新时机

**EMA 的 decay 乘旧影子，更新频率与传入的计数同样影响结果。** 2026-10-22 固定 `6fd82cba13f52f394747832344039408699ea378`，原 [`EMA`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/ema.py)执行 `shadow = decay × shadow + (1-decay) × live`，decay≤0 时直接复制 live。不能把小 decay 解释成“只混入少量新参数”；例如 EmaLoraConfig 默认0.001，对应新参数权重0.999，但何时混入还取决于 timing。

原 make_decay_fn 对 full EMA 使用 `min((1+t)/(10+t), target_decay)`，不是从第一步就固定用 target。target=0.5 时，t=0/1/10/100 分别为0.1、2/11、0.5、0.5。LoRA 配置另有 constant、linear 和 warmup：本次 uprate=0.1、uphold=0.5、flat_steps=2 的 warmup 在 t=0/1/2 都为0，t=3为0.1，t=7为0.5；零衰减阶段会复制当前参数，不是暂停 EMA。

CPU 对照使用原 inject_mirror 与一个仅含两元素可训练参数的 nn.Module，手动赋值模拟参数变化，没有执行 optimizer。初始 live 从 `[2,4]` 改为 `[3,5]` 后才执行原 apply_deferred_ops，影子复制到 `[3,5]`，说明初始复制是延后回调，不是在注册 shadow 时立即完成。原 mirror 只为 requires_grad 参数创建影子；本例冻结参数与 buffer 不在配对中，不能将它理解成完整 module 状态副本。

两份 mirror 从相同影子 `[3,5]` 开始，live 先后手动改为 `[13,25]`、`[23,45]`；均使用 full target=0.5：

| timing | 实际调用 | 最终 shadow |
|---|---|---|
| optimizer_step | step(0)，step(1) | `[21,41]` |
| rollout_end | 两次 step 无效，之后 on_rollout_end(2) | `[18,35]` |

第一路 decay 依次0.1、2/11，第二路只混合一次且用t=2的0.25；差异既来自混合次数，也来自计数。反向时机的调用不会更新，另用原 constant decay=0 验证 shadow 直接变为 live `[23,45]`。这些是 EMA 本体的实测，不是两次真实 optimizer 更新或完整 rollout。

[`BaseTrainBackend`](https://github.com/Tencent-Hunyuan/UniRL/blob/6fd82cba13f52f394747832344039408699ea378/unirl/train/backend/base_backend.py)源码中，grad_norm 有限时先 optimizer.step / scheduler.step，再以尚未递增的 _optimizer_step_count 调 EMA.step，随后计数加1；on_rollout_end 则传当前已提交计数。grad_norm 非有限时提前返回，不执行该次 optimizer/EMA.step 或计数递增。因此训练步时机与 rollout-end 时机不是同一个 t 定义；这里仅核对后端调用顺序，没有导入后端或制造梯度故障。

原 full mirror 的 use_shadow 用数据存储交换而非重新绑定 Parameter 对象，finally 再交换恢复。单层 CPU 上下文内主动抛异常后，live/影子数值、data_ptr 和 Parameter 身份均恢复，requires_grad 未改变、梯度仍为空。这不认证嵌套/并发使用、PEFT 双适配器切换或远端导出。

[复查脚本](../../experiments/scripts/verify_unirl_ema.py)：

```bash
python -B experiments/scripts/verify_unirl_ema.py /path/to/UniRL
```

Python3.12.13 / torch2.9.1，全部原模块正常导入，无包壳、方法摘取或假 Shadow；真实 inject_mirror 返回原 Shadow，原 local_view 直接处理普通 CPU Tensor。四类衰减、延后复制、两种时机、零衰减及异常退出恢复控制通过。这是自编原模块检查，不是上游原测试；无新依赖安装、预训练权重、forward/backward、optimizer、LoRA 注入、DTensor/FSDP、设备发现或 GPU/NPU 运行，不新增 M5 必跑事项。

脚本核对五个完整源文件 SHA256，包括仅读的 backend。ema.py `b4b057cd565780b467c56d0ca8032a48c0e5b5366702fbd11cf2f06c9654701b`，configs.py `18dcc241700eeaf83f6e1aa9cc3a73519350c60c83d1c1631f7a8c574e29e03e`；其余见脚本。此处解释实现契约，不推荐衰减值，也不由局部 CPU 结果判断训练质量。

## 下一步

继续以 UniRL 为高优源码/潜在贡献轨道，优先围绕真实配置与类型契约选题；不是已找到 bug 或获得对外发布许可。真机验证由 M5 独立选择获准且资源匹配的配方，条件不符时保留未测，不为复现默认 HI3 直接启动大模型或升级环境。
