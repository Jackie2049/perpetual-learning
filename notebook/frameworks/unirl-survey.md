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

## 奖励能力校验与模型加载

**配置能声明一种评分能力，不等于模型已能加载；把加载延后，也不等于整个构造过程没有副作用。** 2026-10-22只读[#621](https://github.com/Tencent-Hunyuan/UniRL/pull/621)，读取时open、未合并，固定head `27cb49307fc89dd60331f86d23cadb2e8cc4b0cb`、base及官方compare祖先`6fd82cba13f52f394747832344039408699ea378`（ahead1/behind0），7文件40行新增。它将本地reward的模型物化从叶子构造器移到服务层校验之后，不修改奖励公式或配置项。

### 递归构造为何让校验太晚

[Worker._resolve_init_kwargs](https://github.com/Tencent-Hunyuan/UniRL/blob/27cb49307fc89dd60331f86d23cadb2e8cc4b0cb/unirl/distributed/group/worker.py)先递归构造字典的children，最后调用该层_target_。T2AV是文本到音视频；其组合scorer需要至少一个正权重分量连接prompt与video。旧版ImageBind(mode=audio_video)+CLAP会先在两个叶子的构造器调用_load_model，再由T2AVCompositeScorer拒绝缺少prompt-video能力的组合。缺依赖时，甚至会先遇到模型导入错误，而到不了组合配置错误。这是错误发现顺序，不是原校验条件不存在。

| 阶段 | base中的顺序 | 候选中的顺序 |
| --- | --- | --- |
| 构造本地叶子 | 保存配置并调用_load_model | 保存配置，model/processor为None、_loaded=False |
| 构造组合与服务 | 此时叶子模型已尝试加载，再校验 | 先校验组合能力，再校验服务输入种类及截断配置 |
| 物化模型 | 已在叶子构造器发生 | RewardService末尾调用backend.load，组合逐个转发给叶子 |

变化见[LocalRewardBackend](https://github.com/Tencent-Hunyuan/UniRL/blob/27cb49307fc89dd60331f86d23cadb2e8cc4b0cb/unirl/reward/local/base.py)与[RewardService](https://github.com/Tencent-Hunyuan/UniRL/blob/27cb49307fc89dd60331f86d23cadb2e8cc4b0cb/unirl/reward/service.py)。这不是首次compute_rewards时自动懒加载：直接构造local scorer的脚本须显式load；服务构造成功返回前仍执行load。T2AV、PerDomain、Video三个wrapper新增转发，基类load本身是no-op，自定义组合不能靠继承自动遍历孩子。

### 能力声明的适用范围

[ImageBind.covers_prompt_video](https://github.com/Tencent-Hunyuan/UniRL/blob/27cb49307fc89dd60331f86d23cadb2e8cc4b0cb/unirl/reward/local/imagebind.py)按mode及text_video权重判断，[VideoAlign](https://github.com/Tencent-Hunyuan/UniRL/blob/27cb49307fc89dd60331f86d23cadb2e8cc4b0cb/unirl/reward/local/videoalign.py)按ta_coef>0判断，均不需要先有模型对象。但RewardService只对实现PromptVideoReward协议的backend检查该声明；它不是任意奖励模型语义、checkpoint、依赖或评分质量的通用验证器。preferred_input_kind属于image/video/text也只验证种类名，实际frontier是否包含该模态仍在score_and_attach检查。

服务层require_prompt_video=False只跳过服务自己的检查，不撤销[T2AV构造器](https://github.com/Tencent-Hunyuan/UniRL/blob/27cb49307fc89dd60331f86d23cadb2e8cc4b0cb/unirl/reward/local/t2av_composite.py)的正权重prompt-video要求。组合load遍历所有配置的孩子，既不按当前domain按需加载，也不跳过零权重项；延迟到统一时点不等于按实际请求节省模型内存。

### 成功标志不是资源事务

LocalRewardBackend在_load_model正常返回后才置_loaded=True；后续顺序load可直接返回。这个标志不能由model is not None替代：无权重的MCExactMatch也有空_load_model。onload只把已有model移回device，不承担首次加载；offload也不清除_loaded。源码没有并发load锁，也没有失败自动回滚：某孩子已加载、后一个抛错时，组合循环不撤销前一个；失败叶子的False只说明未成功返回，不证明中途分配已释放。它提供重试入口，不是整图加载的原子性保证。

本PR不改变[ManagedScorerProcessBackend](https://github.com/Tencent-Hunyuan/UniRL/blob/27cb49307fc89dd60331f86d23cadb2e8cc4b0cb/unirl/reward/managed_process.py)：其构造器仍_start_child，创建监听socket/日志并Popen，等待子进程health达到预期状态，然后才返回；load继承no-op。纯RemoteRewardBackend构造HTTP session、模型在外部服务；不能从“本进程没有model”推出“配置拒绝前没有启动子进程或远端工作”。本地叶子构造的resolve_device也仍可能查询CUDA可用性；这里延后的是_load_model，不是所有设备相关操作。这些文件与base的Git blob相同，是范围边界而非本PR新增行为。

### 验证范围与复查

本机Python3.9.6只对6个修改的完整Python模块在base/head各做内存compile，共12次通过；另核worker、managed_process、remote、ImageBind、VideoAlign、device六文件Git blob前后相同。未导入奖励包、执行resolver/构造器/load、启动子进程/HTTP或探测设备，也未制造新模型替身。最小语法复查（对象须已在本地）：

```bash
GIT_NO_LAZY_FETCH=1 python3 -B - /path/to/UniRL-git <<'PY'
import subprocess, sys
for rev in ['6fd82cba13f52f394747832344039408699ea378',
            '27cb49307fc89dd60331f86d23cadb2e8cc4b0cb']:
    for path in ['base.py', 'local/base.py', 'local/per_domain.py',
                 'local/t2av_composite.py', 'local/video.py', 'service.py']:
        src = subprocess.check_output(['git', '-C', sys.argv[1], 'show', f'{rev}:unirl/reward/{path}'])
        compile(src, path, 'exec')
PY
```

head SHA256：local/base.py `b977c2075fa417c17a2f09e1dfa3c159ad1e2382977c112adf8f36955706d221`；service.py `4284399e73f5e01c6e8be65770b10acdfe6bbde3f2d80651b0b5a63762469cd2`。作者报告两种resolver的CPU harness基线24/42、候选42/42，所有_load_model都替换为fake，harness未提交；这是构造顺序/转发的作者证据，不是实际checkpoint加载或本机验证。有效ltx2配方在两版都通过，作者也明确它只是回归检查，不是修复区分项。

本题以初始化生命周期学习收束；真实依赖/权重及失败后的资源状态留[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，只在其未来独立选择匹配reward任务时考虑，不为本题新增加载或训练任务。

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

## Smoke 结果格式与训练验收

UniRL [已有 PR #576](https://github.com/Tencent-Hunyuan/UniRL/pull/576)把 benchmark 结果统一为 schema v1，但它的 `smoke-synthetic` **只测 Python 空循环，不运行训练**。固定 head `a6ed0ec8899f353e33ed1d9ad364bdb560a0f179`、base `6fd82cba13f52f394747832344039408699ea378`，四文件新增 240 行；生产训练、奖励、评测代码没有改动。候选中的 `BenchmarkResult` 实际调用方只有这个 smoke CLI，README 对 rollout/reward/train producers 的描述是接入方式，不是三路已接通的证据。

### 样本计数与可选字段

[`benchmark_result.py`](https://github.com/Tencent-Hunyuan/UniRL/blob/a6ed0ec8899f353e33ed1d9ad364bdb560a0f179/unirl/utils/benchmark_result.py)接收调用方声明的环境、负载和计时，再由 `samples / wall_clock_s` 派生吞吐；它不探测或认证这些声明。`started_at` 要求带时区，输出转 UTC；JSON 顶层键顺序固定，phases 名称排序。

| 字段或操作 | 固定候选的行为 | 解释 |
|---|---|---|
| `workload.samples` | 总测量样本数，不是每迭代数量 | CLI 的 samples=3、measured=4 得 12；warmup=2 不计入 |
| `wall_clock_s` | 必须正且有限；吞吐不能单独填写 | 合成 12 样本 / 3 秒得到 4 samples/s，只验证公式 |
| `torch=None` | 保留为 JSON null | 不把版本未知补成空字符串或某个版本 |
| memory/phases 未提供 | 对应键省略 | 不等于实测为 0；显式 `peak_memory_bytes=0` 保留 |
| `phases_s` | 复制后用只读 mapping 保存 | 后续修改传入字典不改变记录；不自动测量阶段 |
| 同一路径再次写入 | `open("x")` 报 FileExistsError | 旧文件保持原字节，不证明断电原子性/持久性 |

schema 允许 samples=0，此时吞吐为 0；CLI 则额外要求每迭代样本数为正。结果没有 success/status 字段，数值校验也不检查模型质量、收敛、checkpoint 或 GPU 正确性，不能把“JSON 可写”汇总成“训练通过”。`outputs/benchmark/` 与评测的 `benchmarks_results/` 分离；schema_version 是格式版本，不是模型或代码身份哈希。

### CPU 核对与最小复查

2026-10-22 使用 Python 3.12.13，将六个所需原文件从固定 Git 对象原样归档到独立临时目录，正常导入，无 AST 摘取、包壳或假类。`-S -B` 隐去 site-packages 并禁写 bytecode；本机另一个环境装有 torch，这里 `torch=null` 只代表该进程查不到其发行元数据，**不是整台机器没有 torch**。没有安装依赖、导入 torch/Ray 或发现设备。

原 CLI 以 samples=3、warmup=2、measured=4 执行，输出 `smoke-synthetic`、total samples=12、正墙钟时间，吞吐精确等于 12/该时间；这里只报告合同检查，不引用空循环速度作为性能。再次运行同 ID 返回 FileExistsError、文件原字节不变；measured=0 返回 argparse 退出码 2，旧文件仍不变。

直接调用原 schema 另核对 UTC、键序、可选缺省/显式 0、phase 输入快照、0 样本记录，以及六项拒绝：bool 样本数、0 measured、0/NaN 墙钟、Inf phase、负显存。均符合上述规则。这是自编检查；作者在 PR 正文报告的 19 项临时 harness 未提交，也没有被当作本轮运行过的原测试。

最小 CLI 复查（使用支持该候选的 Python 3.12；需要已有 Git 克隆）：

```bash
git -C /path/to/UniRL fetch origin pull/576/head
schema_scratch=$(mktemp -d /tmp/unirl-schema.XXXXXX)
git -C /path/to/UniRL archive a6ed0ec8899f353e33ed1d9ad364bdb560a0f179 \
  unirl/__init__.py unirl/utils/__init__.py unirl/utils/run_id.py \
  unirl/utils/benchmark_result.py benchmarks/__init__.py \
  benchmarks/speed_benchmarks/benchmark_smoke.py | tar -x -C "$schema_scratch"
cd "$schema_scratch"
UNIRL_RUN_ID=smoke-contract python3.12 -S -B -m benchmarks.speed_benchmarks.benchmark_smoke \
  --samples 3 --warmup-iterations 2 --measured-iterations 4 --out results
python3.12 -S -B - <<'PY'
import json
from pathlib import Path
data = json.loads(Path('results/smoke-contract.json').read_bytes())
assert data['benchmark'] == 'smoke-synthetic'
assert data['workload'] == {'samples': 12, 'warmup_iterations': 2, 'measured_iterations': 4}
assert data['environment']['torch'] is None
metrics = data['metrics']
assert metrics['wall_clock_s'] > 0
assert metrics['samples_per_second'] == 12 / metrics['wall_clock_s']
assert set(metrics) == {'wall_clock_s', 'samples_per_second'}
print('schema/CPU-loop check only, not training or UniRL performance')
PY
```

SHA256：schema `760bcc3bb262907eebc1600045b5f83b770c717027d97ea43f4f88eff608a17c`；CLI `cc42e43ca953c871042daf829ca3ce5c500067e40def2c388a71fc7d2a048aac`；复用 run_id helper `e8038f48a11f2ac48e1958acc4b07fb8a4be98a2a5eb523a7aa7ffd9de148a82`。没有向 UniRL 提交 harness、采用候选或启动训练，此项不新增 M5 硬件任务。真实性能比较仍需对应模型/算法、负载、硬件、奖励与测量窗口；格式统一不能替代这些条件。

## 损失单位与微批次权重

**平均 loss 必须同时说明单位与窗口。** [已有 PR #622](https://github.com/Tencent-Hunyuan/UniRL/pull/622)将单位声明移到 `StageAlgorithm.loss_units(part)`，训练栈不再从 algorithm 的属性猜测单位；固定 head `9dded4e2fd92732dd9de40252a4ae5b342b164e0`、base `6fd82cba13f52f394747832344039408699ea378`，四文件 53 增/34 删。这是 #580 第一阶段接口迁移：微批缩放仍为 `w × G / D`，不是引入新的 token 目标，也没有将 `LossNorm` 传入每个算法的 backward 接口。

### None 与零计数

基类在 token-mean 时要求 TextSegment，按 packed cu_seqlens 对 loss_mask 做累计和差分，得到每行有效 token 数；没有 mask 时回退 lengths。其他模式返回 None。训练栈随后分两路：

| `loss_units` 结果 | 微批权重与归约 | 约束 |
|---|---|---|
| None | 微批行数 / 本 update 行数；本 helper 不做单位 all-reduce | 表示 sample share，不是无监督或跳过训练 |
| 计数 tensor | w 是该微批有效单位数，D 是全局单位总数，G 是梯度平均 world；权重 w×G/D | 要求 SP size=1；不能把 G 随意替成总卡数 |
| tensor 全为0 | 仍进入计数路径；全局 D≤0 时返回零权重与该 D | 不能将“本 rank 无 token”改成 None，否则 collective 分支可能不一致 |

`LossNorm(denominator, grad_world)` 在 helper 内保存 D/G，不执行归约或认证单位。backend 的 `all_reduce_loss_sums` 与 `gradient_average_world_size` 使用同一 FSDP mesh；G 用于抵消梯度平均，因此分布式下单个 rank 的 scale 不必加总为1。自定义 `loss_units` 必须在所有 DP rank 一致返回 None 或都返回计数，这是文档前提，不是本轮证明的自动一致性检查。

原 single/unified stack 没有在该 PR 中改动：TrainStack 对全局零 token update 抛错；unified 单路为零时保留零 scale 及警告，由另一轨道继续。这是源码接线，不是本机执行过的 optimizer 行为。[算法基类](https://github.com/Tencent-Hunyuan/UniRL/blob/9dded4e2fd92732dd9de40252a4ae5b342b164e0/unirl/algorithms/base.py)、[缩放 helper](https://github.com/Tencent-Hunyuan/UniRL/blob/9dded4e2fd92732dd9de40252a4ae5b342b164e0/unirl/train/stack/loss_scale.py)、[训练约定](https://github.com/Tencent-Hunyuan/UniRL/blob/9dded4e2fd92732dd9de40252a4ae5b342b164e0/unirl/train/readme.md)。

### 本机聚合函数对照

2026-10-22 用原 `aggregate_token_losses` 和真实 CPU autograd 核对两种局部归约。合成 packed 行长 `[2,3]`，loss 为 `[2,999 | 4,6,8]`，999 所在槽始终被掩码；horizon=1。两个版本的原聚合函数 AST 相同，本轮没有执行新 `loss_units` 或 `resolve_loss_scales`。

| 每行有效 token 数 | 归约目标 | 原函数整批 loss | 两个单行微批的手动权重 |
|---|---|---|---|
| 1、3 | token-mean | 5 | 1/4、3/4 |
| 1、3 | seq-mean-token-mean | 4 | 1/2、1/2 |
| 0、3 | token-mean | 6 | 0、1 |
| 0、3 | seq-mean-token-mean | 3 | 1/2、1/2 |

四组重组的前向值及对 loss 输入的梯度均与整批相同；第一组 token 梯度为 `[1/4,0,1/4,1/4,1/4]`，seq 梯度为 `[1/2,0,1/6,1/6,1/6]`。完全掩码的序列在 seq-mean 下贡献0但仍占一行，不能从分母删除。把 token-mean 的微批误乘等行数份额，会得到另一个目标，不能用“都是 mean”解释为等价。

窗口也有独立语义：single stack 的 accumulation window 逐 Part 求缩放，再乘1/M，因此平均各 Part 的 token mean，而不是把全部 Part 拼成一个 token 分母。例如有效计数1、3的两 Part 均值为2、6时，Part 平均为4，全 token 平均为5；这是现有文档规定的窗口目标，不据差值定性缺陷。本轮没有运行 accumulation stack。

复查（Python3.12.13 / torch2.9.1，显式 CPU）：

```bash
git -C /path/to/UniRL fetch origin pull/622/head
python -B experiments/scripts/verify_unirl_loss_reduction.py /path/to/UniRL
```

[脚本](../../experiments/scripts/verify_unirl_loss_reduction.py)完整执行原 loss_agg 词汇模块，AST 仅选择完整原聚合函数、延后注解，无包壳/假 backend。微批权重是显式单进程算术，不是假装运行生产缩放器。打印完整源码 SHA256：head base.py `f586474992dee187755ae47f431a90dc4857180fa1b1aa01dd8b27e2f59a4f1a`，loss_scale.py `d627ea261f0867d46335a1264622911913ff1d684450b7333aaf0dc67209f13c`，loss_agg.py `8049957fa4ecbf2086afab6e066fa1beeb9d35bdabebfd3d8890f7ee078ecd3f`。

本机环境缺 Ray/Hydra/OmegaConf/Pillow，未补装或导入训练栈，未复跑作者报告的双 rank Gloo/FSDP2 gate、80,640 组差分或 AROPD。这里确认局部目标差异和固定源码路径，不认证候选所有输入 bit-identical、分布式归约、模型梯度或训练收益；不自动采用或新增 M5 必跑事项。

## AROPD 的采样梯度与 KL 目标

[已有 PR #534](https://github.com/Tencent-Hunyuan/UniRL/pull/534)新增 AR 端 on-policy distillation：teacher 评价 student 自己生成的 token，再将 teacher/student 的 log-prob 差作为冻结权重。固定 head `1962740c4aabe104c2528d0a79ed853c52d05046`、base `0fe4f99dc2384a022fe0c8472aec889c216d3c07`；本节区分源码机制和固定前缀下的数学，不把采样 surrogate 的值直接当作 KL。

### Teacher 与 student 的接线

[`AROPD`](https://github.com/Tencent-Hunyuan/UniRL/blob/1962740c4aabe104c2528d0a79ed853c52d05046/unirl/algorithms/aropd.py)要求注入 backend，并在同一模型上找到命名的 teacher adapter。配方是同 backbone/tokenizer 的可训练 default LoRA 加 frozen teacher LoRA，不是任意远程大模型 teacher 服务。`prepare_segment` 在 no_grad 与 adapter_active 上下文内 replay，将结果 detach 到 CPU 的 packed `teacher_log_probs`；anchor contract 按训练微批几何准备并恢复行序。teacher forward、adapter 冻结及实际输出身份这里只读源码，未运行 PEFT/模型。

student 的旧 log-prob 来自 `segment.log_probs`，新 log-prob 来自当前 replay；raw gap 是 `sg(teacher_logp-old_logp)`，只有 new_logp 接收梯度。代码声明无需奖励优势、单 teacher、同步路径与不支持 multi-update；配方 num_updates_per_batch=1，teacher replay 温度取 sampling.temperature。一次 update 的声明本身不证明实际 rollout/replay 概率逐位相等，代码另记录 drift 指标。

默认 `normalize_gap=true` 还对当前微批 gap 做中心化/除标准差（元素数>1才做），指标保存 raw gap。本节的恒等式仅讨论 **raw gap**，不能据此认证默认 z-score 的无偏性；普通 baseline 的零期望论证要求其相对于当前动作独立，不能不加条件套到动作相关权重或样本自身统计量。

### 固定前缀的梯度恒等式

令 student 分布 p=softmax(z)，teacher 分布 q 冻结，所有动作概率为正、旧新 student 相同，不改变温度或采样支持；这里把前缀当固定输入。reverse KL 为 `K=Σ p·(log p−log q)`，其梯度为 `E_p[(log p−log q)∇log p]`。对应最小化 surrogate 是 `L=−sg(log q−log p_old)·log p_new`；两者**期望梯度**相同，不要求标量值相同。

与它不同，冻结抽样权重后直接微分 `log p−log q` 只剩 score，其期望 `Σp∇log p=0`；有限 batch 仍可能有非零采样噪声。这不表示 KD 普遍无效：完整词汇分布 KL 的梯度正是这里的比较基准。另一个不同目标是 teacher cross-entropy `C=−E_p log q`；`K=C−H(p)`，少了 student entropy 项便不是同一个梯度。固定候选 README 将最大化 E_p log q 与 raw gap 式连写、把 student log-prob 称为 control variate，这个等同关系不能直接沿用。

这也要区分原论文的范围：[GKD §3 式2、4](https://arxiv.org/html/2306.13649v3#S3)在 student 生成的前缀上比较 teacher/student 的 token 分布，并明确不反传外层序列抽样过程；它允许不同 divergence。on-policy 描述数据来自谁，不自动证明某个 sampled-token、归一化或完整轨迹目标都彼此等价。

### 本机精确枚举

2026-10-22 用自编 CPU torch FP64 参考，完整枚举三个动作，不实际抽样。p=[0.2,0.3,0.5]、q=[0.7,0.2,0.1]，对 student logits 的梯度如下：

| 计算图 | 梯度 |
|---|---|
| 完整 reverse KL | `[−0.385714,−0.081102,0.466816]` |
| raw score surrogate 的精确期望 | 同上 |
| 直接 gap，冻结抽样权重 | `[0,0,0]`（浮点误差内） |
| teacher cross-entropy，保留 p 的梯度 | `[−0.269757,−0.028806,0.298563]` |

KL=0.675806，而 surrogate 值约−0.300991：负 distill surrogate 不是负 KL 的证据。再令 q=p，reverse KL/raw surrogate 的梯度均为0，但 teacher cross-entropy 梯度为 `[0.115957,0.052296,−0.168253]`，仍会改变已匹配的分布。两例均核对 `∇C+∇(Σp logp)=∇K`，且原始 KL 的中心有限差分与 autograd 一致。

[参考脚本](../../experiments/scripts/verify_aropd_score_gradient.py)：

```bash
python -B experiments/scripts/verify_aropd_score_gradient.py
```

Python3.12.13 / torch2.9.1，显式 CPU，无包壳、模型或依赖安装。**这是自编参考数学，不是执行原 AROPD，也不是作者测试**；没有证明多步轨迹梯度、默认 gap normalization、off-policy correction、PEFT/FSDP 或训练收益。源码/配方 SHA256 分别为 `c96283715000b9979f84b0d1712c542fb7140575e4997a493ac3a3110e221795` / `4664747d7123e874bfb13d61ee5526880344becc54f5427dbcbe2e25f27305d6`，推导 README 为 `6214d59d75759899dfea09bd2df500c24d10d18223a24e9daaef70d7f7db35e9`。

作者报告的6×H20训练早于该 head 的 anchor/单位接线 rebase，正文明确未在 rebase 后复跑 GPU；其 CPU harness 也有模型 forward 替身，不当作本机结果。此处只保存理论限定与学习证据，不改上游、发布审查结论/候选、启动 teacher 实验或新增 M5 必跑事项。

## 关键帧条件的数据身份与功能就绪

**rollout和replay读到同一份条件数据，是概率比较的前提之一；消费接口存在却不等于关键帧生成功能已能启用。** 2026-10-22只读已有[#584](https://github.com/Tencent-Hunyuan/UniRL/pull/584)，读取时open，固定head `55a3cad3e934cb0e637d8f3cf1c37d2a91886891`、目标base `accba01082115534e8b60d849e7276a636eba269`、merge-base `c7c6b8c0e0c60db7205d9b81fdafca778c747ece`。6文件125增22删，全部在MiniMax-H3模型包，无提交测试文件。FL2VA指首帧、末帧或首末帧条件的视频/音频生成；本候选仅实现路线图的消费端step1a。

### 条件行不等于原始图像或目标轨迹

[`MiniMaxH3KeyframeCondition`](https://github.com/Tencent-Hunyuan/UniRL/blob/55a3cad3e934cb0e637d8f3cf1c37d2a91886891/unirl/models/minimax_h3/conditions.py)按样本保存3个对齐列表：已经加噪的fp32 rows、anchors和未padding的text_token_tags。rows不是待读取的图片路径，也不由replay重新编码/抽噪。模型latent通道24、patch为(1,2,2)，故每行宽96；rows_per_frame由目标画布latent高宽除patch尺寸后相乘。

| 样本/批形态 | keyframes表示 | 必须保留的边界 |
|---|---|---|
| 纯文本生成批 | 全批keyframes=None | 使用原t2va布局 |
| 带关键帧的混合批中的文本样本 | anchors=()、fp32 rows形状[0,96]、全text tags | 是一个空条目，不是省略该样本 |
| 首帧或末帧条件 | anchors=(first,)或(last,)，rows为[R,96] | 标签长度对应去padding后的文本 |
| 首末帧条件 | anchors=(first,last)，rows为[2R,96] | 顺序和每样本对应关系不能只靠形状推定 |

R表示rows_per_frame。[build_layout](https://github.com/Tencent-Hunyuan/UniRL/blob/55a3cad3e934cb0e637d8f3cf1c37d2a91886891/unirl/models/minimax_h3/packing.py)核对batch-1、合法anchor元组、long型0/1 tags及长度、fp32 rows形状；空anchors必须全text tag。条件容器另核3列表等长，以及keyframes样本数等于text样本数。它们是结构检查，不认证某行确属对应rollout样本、图像内容或随机噪声。

通用[Batch拼接](https://github.com/Tencent-Hunyuan/UniRL/blob/55a3cad3e934cb0e637d8f3cf1c37d2a91886891/unirl/distributed/tensor/batch.py)对字典key取并集后递归处理：其他项已有值时，非零行项缺少key会报错；零行项可以缺省。这项保护来自已合入的通用规则，不是本PR重新实现。否则将3份关键帧条目与4份文本行对齐，后续样本可能整体错位。混合批要为每个真实样本提供条目，不能用“这个样本没有关键帧”作为删掉一项的理由。

`trim_text_padding`改用dataclasses.replace，仅替换text字段、保留keyframes等槽，避免重建只含text的容器时丢条件。tags本来就应未padding，不能跟着已padding文本长度生成。保留字段并不让可变tensor/list成为不可变快照；运输/切片/训练对齐还需保证实际payload没有被换样本或转低精度，结构guard不是字节身份校验。

### 条件参与前向但不变成待评分目标

三个入口generate、replay和predict_noise_at_step都经相同的[`predict_noise`](https://github.com/Tencent-Hunyuan/UniRL/blob/55a3cad3e934cb0e637d8f3cf1c37d2a91886891/unirl/models/minimax_h3/diffusion.py)：将携带rows stack后搬到目标device，放在video_sample前，再调用transformer；返回时裁掉前面的condition video velocity，只将target video/audio送回下游。NFT的clean latents则由segment末态video/audio打包，不把条件行附到目标上。本轮未执行NFT loss，不能由forward形状正确推断训练正确。

联合音视频replay还要求segment.aux_latents，不能只凭视频轨迹和关键帧恢复整个前向。在[batched replay](https://github.com/Tencent-Hunyuan/UniRL/blob/55a3cad3e934cb0e637d8f3cf1c37d2a91886891/unirl/models/types/batched_replay.py)中，多step tile通过 `type(conditions).concat([conditions] * S)`复制条件，再送predict_av_noise；因此条件数据和目标状态的sample/step排列必须一起核对，不能只对最终log-prob均值。

直观地，比较同一转移的条件概率时，分子与分母需要对应同一已实现的条件c；若训练时重抽条件噪声得到c'，它就不再只是“在相同输入上换一套策略参数”。这是条件身份的解释，不是本轮对某个SDE核或NFT目标的数学/数值验收。README所述diffusers/vLLM-Omni噪声与第二帧resize差异，是作者的实现对照；本机未审这两条外部生成路径。

### 仍有两个启用前提

固定head的[pipeline.generate](https://github.com/Tencent-Hunyuan/UniRL/blob/55a3cad3e934cb0e637d8f3cf1c37d2a91886891/unirl/models/minimax_h3/pipeline.py)仍仅构造 `MiniMaxH3Conditions(text=...)`，与目标base逐字节相同；它没有本PR所需的keyframe编码、conditioner prompt和数据producer。类型导出/下游接线不能算实际用户请求已到达关键帧路径。

`row_timestep_plan`的base/head AST也相同：标量分支将condition_video_timestep设为target video_t=1−sigma；向量分支先将所有行填video_t，再只改audio target列。两条分支都未给条件帧独立的近clean timestep。[候选README](https://github.com/Tencent-Hunyuan/UniRL/blob/55a3cad3e934cb0e637d8f3cf1c37d2a91886891/unirl/models/minimax_h3/README.md)明确尚需step2的max(1−sigma,0.999)条件时间，之前禁止recipe启用FL2VA。即使合成条件在rollout/replay之间保持一致，也只说明这条实现的一致性，不说明已等价发布模型的条件语义。

作者报告外部一次性harness的合成条件parity、tiny随机transformer及CPU/MPS往返；harness未提交，本轮无法复跑。作者也明确没有实际producer、NFT loss或GPU/FSDP验证。这些报告不当本项目结果，也不以“205通过”或t2va兼容声称真实关键帧训练已就绪。

M1只做源码学习和静态检查：Python3.12.13对5个改动Python文件及pipeline/config/Batch/batched_replay共9个完整模块内存compile；核row_timestep_plan AST未变、pipeline字节未变且仅一个text条件构造点，均通过。没有导入torch/UniRL、实例化条件/变换器、复跑parity或loss、使用MPS/GPU/NPU，未增加容器/模型替身或永久一次性工具。

复查静态入口（只compile不执行，须本地已有Git对象）：

```bash
python3 -B - /path/to/UniRL-git <<'PY'
import ast, subprocess, sys
head, base = '55a3cad3e934cb0e637d8f3cf1c37d2a91886891', 'accba01082115534e8b60d849e7276a636eba269'
def src(rev, name):
    return subprocess.check_output(['git', '-C', sys.argv[1], 'show',
                                   f'{rev}:unirl/models/minimax_h3/{name}.py'])
for name in ['conditions', 'packing', 'diffusion', 'pipeline']:
    compile(src(head, name), name, 'exec')
def plan(rev):
    return next(n for n in ast.parse(src(rev, 'packing')).body
                if isinstance(n, ast.FunctionDef) and n.name == 'row_timestep_plan')
assert ast.dump(plan(head)) == ast.dump(plan(base))
assert src(head, 'pipeline') == src(base, 'pipeline')
PY
```

SHA256：conditions.py `59d217094b12f23bd09dd434ea3009ad4f2db16004447d886c766624f1eb1206`；packing.py `ae1693dc89e6523ac5a19cfb458058fab3a58bada75228722e7feefe05f8cc19`；diffusion.py `c9bc12117fa8babc041ee451576cdc47baffb795bb40ec3fd9465969891a26c9`；pipeline.py `7164e4d552338ed344e8a0244426758cb905b5c9c3f229239ea08323b2fd0efa`。真机缺口仅补[原M5 UniRL入口](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)：先等producer/条件时间语义就绪并独立选型，当前不启用、不另开硬件任务，也不修改已有作者的方案。

## 设备资源与通信后端的分层边界

**资源预约、进程可见性、实际设备与通信组不是同一个承诺。** 2026-10-22只读[#623](https://github.com/Tencent-Hunyuan/UniRL/pull/623)：当轮API状态为closed、merged=false，关闭于09:39:42Z；不是已合入的支持声明。固定head `c1477a030cda826b6befd2e6e787ef64f67e41dd`，实际父提交/官方compare merge-base `c6d4b11b24743fe1e933b61192f9d49558f7171e`，仅3文件46增7删、无新增测试。API目标base为 `6fd82cba13f52f394747832344039408699ea378`，两侧ahead1/behind59；不能把目标分支至head的两点diff当作本PR改动。本地浅克隆起初找不到merge-base，已用官方compare/commit parent核定并获取父对象，不据失败猜版本。

### 四层分别核对

| 层次 | 固定head中的行为 | 不能由此推出 |
| --- | --- | --- |
| Ray资源预约 | placement group仍按每设备`GPU:1`与CPU bundle、STRICT_PACK预约 | actor的`num_gpus=0`不等于无GPU资源需求或CPU集群可跑 |
| Actor可见性 | `_spawn_worker`无平台分支地将num_gpus改0；从driver的CUDA_VISIBLE_DEVICES列表按全局device_id选一个值注入 | bundle_index不是物理卡身份；全局索引超列表长度不注入，不能认证多节点映射 |
| Torch设备选择 | cuda可用时，单个纯数字ID且小于device_count则pin该ID，否则pin0；设置`cuda:<pin>` | 回退0不是正确分卡的验证；仍用cuda命名空间，不代表原生NPU支持 |
| Tensor传输组 | 先建显式ProcessGroupNCCL/eager connect；任意Exception转`init_process_group(backend="eccl")`并取WORLD | 不等于自动选择任意硬件后端，也不保证默认组生命周期无冲突 |

具体见[device_pool.py](https://github.com/Tencent-Hunyuan/UniRL/blob/c1477a030cda826b6befd2e6e787ef64f67e41dd/unirl/distributed/group/device_pool.py)、[worker.py](https://github.com/Tencent-Hunyuan/UniRL/blob/c1477a030cda826b6befd2e6e787ef64f67e41dd/unirl/distributed/group/worker.py)与[store.py](https://github.com/Tencent-Hunyuan/UniRL/blob/c1477a030cda826b6befd2e6e787ef64f67e41dd/unirl/distributed/tensor/backend/colocate_store/store.py)。[Ray资源文档](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)区分逻辑资源与物理使用，[placement group文档](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)说明bundle预约，[加速器文档](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)说明资源请求及环境变量分配。它们解释API职责，不是对作者GCU栈或本PR的认证。

设备编号的源码推演例：若单值CVD="2"被运行时重映射为仅1张可见卡，判断得到pin0；若运行时忽略该mask且报告8张，则得到pin2。空/多值/非数字（如UUID）或越界也走0。这只是条件分支推导，M1没有调用is_available/device_count/set_device，没有证明具体插件遵守哪种语义。原注释称NVIDIA等价，但改动没有NVIDIA豁免分支，回归仍需独立证据。

### 回退还改变了组的所有权

`Worker.setup_global_pg → transport.setup_transfer → TensorStore.setup_global_pg`为这条传输组接线。显式NCCL对象失败后改初始化进程默认组，不只是换一个局部变量的后端。[PyTorch初始化文档](https://docs.pytorch.org/docs/2.14/distributed.html#torch.distributed.init_process_group)明确该API初始化默认进程组；[第三方后端文档](https://docs.pytorch.org/docs/2.14/distributed.html#third-party-backends)要求插件注册并由实现决定能力。文档版本用于解释语义，非作者环境版本。需要实际栈确认eccl已注册、默认组是否已有持有者、rank/world与销毁职责；宽泛异常也可能来自非厂商原因，源码没有在except中显式清理已部分创建的NCCL对象。这些是未核条件，不是已复现的故障或修复建议。

send/recv按`hasattr(group, "send"/"recv")`选择成员方法，否则调用`dist.send/recv(..., group=group)`；不是按后端名选择。取到WORLD也可能仍走成员方法，方法存在不认证P2P可用。原put仍按tensor.is_cuda决定本地设备存储或ray.put，不能单凭新增eccl外推原生NPU高速传输。

作者正文报告8张燧原GCU L600上GRPO 10/10 rollout、曲线形态及NVIDIA等价检查，未附可复跑命令/配置/原始日志；本机没有复验，不能转称4090、昇腾或跨节点结果。PR关闭原因未知，不据merge_commit_sha字段推断合并，也不自动采用、修改或对外评论。

### 本机范围与复查

Python3.12.13仅标准库，对父/head各3个完整模块内存compile，共6次通过。没有导入UniRL/Ray/torch、执行构造器、模拟设备/通信成功或运行原测试；这是语法与源码证据，非设备、训练或NVIDIA回归验收。最小静态复查（Git对象须已在本地，禁止隐式拉取）：

```bash
GIT_NO_LAZY_FETCH=1 python3 -B - /path/to/UniRL-git <<'PY'
import subprocess, sys
for rev in ['c6d4b11b24743fe1e933b61192f9d49558f7171e',
            'c1477a030cda826b6befd2e6e787ef64f67e41dd']:
    for path in ['unirl/distributed/group/device_pool.py',
                 'unirl/distributed/group/worker.py',
                 'unirl/distributed/tensor/backend/colocate_store/store.py']:
        src = subprocess.check_output(['git', '-C', sys.argv[1], 'show', f'{rev}:{path}'])
        compile(src, f'{rev}:{path}', 'exec')
PY
```

head SHA256：device_pool.py `eeb37485a54a1c8b4d9aaa07f3fad7db20c72916433fb2189a61d331651dc543`；worker.py `2f2c2577b8649ee9836006503d748826c42505067432995b470d0b5c94eeb189`；store.py `c19fef9e947de74a2d8ff61508d25221cbad6995e8f806f82cf090f60b1ebcc2`。真机缺口记入[原M5 UniRL参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不另开硬件队列。本题以分层契约学习收束，不继续造设备/通信替身。

## TempFlow 分支共享与逐行时间索引

**共享确定性前缀的输入，不等于复用前缀计算；合并存储槽，也不等于合并实际时间。** 2026-10-22只读[#552](https://github.com/Tencent-Hunyuan/UniRL/pull/552)，当轮open、未合并，固定head `ef9cb362ee8f92f1dcae6ec474075ed922ef994d`、base/官方compare共同祖先 `6fd82cba13f52f394747832344039408699ea378`，ahead17/behind0，23文件1367增62删。差异包含#546/#589的Flash-GRPO及replay前置工作，不全算作TempFlow独有实现。以下解释此候选的接线，不认证与论文/官方实现等价。

### 共用初始噪声但仍重复完整生成

[trainer._build_branch_request_samples](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/unirl/trainer/diffusion.py)为每个选中的步j建立一次包含全部prompt的请求，固定`sde_indices=[j]`。生成sample_id带步组序号和分支序号以保持唯一；init_noise_group_id则只带prompt父ID和seed序号，使同seed的6分支、不同步组使用同一初始噪声键。它是当前输入批次内的共享，不据此承诺跨rollout、批形状或引擎逐位轨迹一致。

这些显式键通过[NoiseRecipe.from_sample](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/unirl/types/noise_recipe.py)优先于init_same_noise的默认分组；[SD3Pipeline](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/unirl/models/sd3/pipeline.py)再生CPU fp32 x_T并送diffuse。因此不能因继承配方init_same_noise=false就推断此分支路径没有共享x_T；缺driver latent shape时trainer反而拒绝branched布局。

[SD3 diffuse](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/unirl/models/sd3/diffusion.py)仍遍历全部T步，仅j的eta非零，其余为ODE步。分支前相同输入沿相同确定性路径是设计目标，但每组重新调用generate；源码没有在这里将前缀算一次后缓存给所有步组。减少训练的随机转移数量，不等于只做一次模型前向或缩短为一步图像生成。

固定[配方](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/examples/diffusion/sd3/sd3_tempflowgrpo.yaml)继承10步SD3.5-M生成，24样本/prompt=4种子×6分支，候选分支步为0..8，每rollout轮转选3步。按源码计数，步组依次为[0,1,2]、[3,4,5]、[6,7,8]，每prompt产生72行；全9步组则216行。batch_size=16时，每个步组384行、三组共1152行；这是逻辑行数，不是并发峰值、显存、调用耗时或训练收益。池排除最后去噪步，候选还检查其存储形状与早期步组不兼容。

### 先组内算优势再保存每行的真实步号

trainer逐步组完成reward/advantage后才合并生成frontier，不复制共享的root前缀。配方advantage_group=prompt、adv_use_global_std=false，所以每个(step,prompt)的24行一起归一化，不跨步组比较奖励；可选seed分组则以连续6行为一组。共享初始seed和优势分组是不同选择，不能把默认配方称为“每seed单独归一化”。这里只核原Part.compute_advantages及trainer接线，未运行组内奖励计算。

合并前的`_stamp_sde_index_per_sample`把每行潜变量固定为以下存储槽，同时单独保存真实j：

| 合并后的字段 | 含义 | 消费方式 |
| --- | --- | --- |
| latents[:,0/1/2] | 本行分支前、本行分支后、终态 | 固定槽位置，不是所有行共有的全局去噪时间 |
| sde_index_per_sample | 每行原来的j | 按j/j+1取完整sigma表中的两项 |
| indices=0,1,2与sde_indices=[0] | 合并布局的槽/单转移元数据 | 不能由这个0推出所有行都在去噪第0步 |

[PerSampleStepReplayMixin](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/unirl/models/types/per_sample_replay.py)据每行j取sigma/sigma_next，用槽0/1重算该转移，返回log_probs[N,1]；不会用共享latents_at(j)解释合并后的槽。它还在replay端init_schedule，不能假设该rank已跑过生成。step_index=0这个调用参数也不代替逐行sigma。

[PerSampleStepGRPO](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/unirl/algorithms/per_sample_grpo.py)取每行旧logp首槽，clip优势后算clipped ratio loss；[TempFlowGRPO.row_weights](https://github.com/Tencent-Hunyuan/UniRL/blob/ef9cb362ee8f92f1dcae6ec474075ed922ef994d/unirl/algorithms/tempflowgrpo.py)再乘weight_scale×FlowSDEStrategy.transition_std（配方weight_scale=2.25），后者含std_dev_t×sqrt(−dt)，不是只用扩散系数或一个统一步权重。构造要求beta=0、eta>0及支持逐行索引的FlowSDE stage；async trainer明确拒绝此布局。配方的old_logp_source继承sd3_trainside的replay，并非类默认rollout；本轮没有执行anchor准备、loss或backward。

### 本机仅验证原噪声模块

Python3.12.13 / torch2.9.1，将固定Git blob的完整`unirl/sde/noise.py`原样载入独立命名空间，无方法摘取、包壳或模型替身。该文件与base的blob相同，是复用的基础模块，不是本PR修复验证。向regen_initial_noise显式输入2个prompt×4个seed×6个分支的48个测试键、latent_shape=(2,3)、CPU fp32/base_seed42：三次再生相同，同seed六行相同，测试中的不同seed/prompt不同；逆序输入对应逆序输出，改base_seed会改变输出，CPU全局RNG状态未变。这些键是控制输入，不是实际执行trainer生成的Sample，也不证明去噪前缀/branch logp或loss parity。

最小复查（对象须已在本地，模块第三方依赖仅torch；本例只调用CPU噪声函数）：

```bash
GIT_NO_LAZY_FETCH=1 python -B - /path/to/UniRL-git <<'PY'
import subprocess, sys, torch
head = 'ef9cb362ee8f92f1dcae6ec474075ed922ef994d'
src = subprocess.check_output(['git', '-C', sys.argv[1], 'show', f'{head}:unirl/sde/noise.py'])
ns = {'__name__': 'tempflow_original_noise'}
exec(compile(src, 'unirl/sde/noise.py', 'exec'), ns)
keys = ['p::seed0'] * 6 + ['p::seed1'] * 6
x = ns['regen_initial_noise'](keys, 42, (2, 3), torch.device('cpu'))
assert torch.equal(x[:6], x[0].expand(6, 2, 3))
assert not torch.equal(x[0], x[6])
PY
```

noise.py、noise_recipe、trainer.diffusion、TempFlow/PerSample GRPO、PerSampleReplay、SD3 diffusion/pipeline及Sample共9完整模块内存compile通过，其余模块不执行。noise.py SHA256 `694a81dcf17cf2ff1c5de13399dcc5d721bc02c100394620acc845a9b7b1830c`；trainer.diffusion.py `55dc251e5f707af2501c16e76d892efd65794f60efb0207855c397a12e2d5458`；per_sample_replay.py `bbf5b6358c65c3aa047312ce8a5c9fbab9945d90240286a52286ad1f50a4c6f3`。

PR未提交所述外部harness；作者40项CPU/早期8×L40S smoke与5a1eff0的fp32 parity报告均非本机结果，且正文明确最新API/replay修改未重跑GPU。其原生批形状latent阈值未过、统一microbatch=4后才逐位匹配的观测，也不能外推默认bf16或训练等价。真机缺口补[原M5 UniRL参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不自动选用、部署配方或新增训练队列；本题以分支与时间轴契约学习收束。

## 异步快照与指标消费

**先分清状态、事件和采样时刻，再解释异步曲线。** 2026-10-22只读[#620](https://github.com/Tencent-Hunyuan/UniRL/pull/620)，读取时open、未合并，固定head `f404c9024c00bbf8a5d05315153444cee894865a`。API目标base为`6fd82cba13f52f394747832344039408699ea378`，官方compare共同祖先为`0397ddb6a6fc0c9ccdc3508c0bee0e687000bca7`，ahead4/behind19；按共同祖先核对才是7文件228增9删，不把落后主线的历史差异算作本PR修改。新增接口不代表此版已部署，也不是训练通过证据。

### 队列深度不是一种单位

[RolloutPool.snapshot](https://github.com/Tencent-Hunyuan/UniRL/blob/f404c9024c00bbf8a5d05315153444cee894865a/unirl/rollout/manager/dispatch.py)在condition锁内复制本地值，不probe handle、不resolve或发RPC；[manager.snapshot](https://github.com/Tencent-Hunyuan/UniRL/blob/f404c9024c00bbf8a5d05315153444cee894865a/unirl/rollout/manager/rollout.py)随后读取driver持有的ready/pending状态。应在driver操作之间读取；只有pool部分加锁，不是集群全局原子快照。closed/failed会拒绝读取，后台错误可能因此较早暴露，不是“监控永不影响报错时机”。

| 字段 | 本地计数点 | 不能当成 |
| --- | --- | --- |
| queued_tasks | 尚在派发队列 | 已在设备运行的工作 |
| launch_reserved_tasks | 已出队、尚未登记返回handle；调用可能已经发出 | 尚未发出任何远端调用 |
| running_tasks | 已登记、尚未被后台观察为ready的handle | 精确的GPU执行中数量 |
| completed_unrouted_tasks | 后台已观察ready，driver尚未取走 | 已resolve、组装、过滤且可训练的样本 |
| ready_groups / pending_roots | 已组装的逻辑组 / 尚未齐全的terminal sibling根 | 同一种任务数、chunk数或token数 |

native AR/DiT每个任务提交一个prompt树，所以trainer日志用prompts命名；agentic任务则是轨迹，不能把通用manager所有计数一律解释成唯一prompt数。slot_occupancy=(running+reserved)/sum(capacities)，是本地逻辑slot占用比，不是GPU利用率；capacity也不是trainer的max_inflight准入预算。

### lag 与横轴各有自己的时刻

[CompleteGroups.oldest_output_version](https://github.com/Tencent-Hunyuan/UniRL/blob/f404c9024c00bbf8a5d05315153444cee894865a/unirl/rollout/manager/buffers.py)只取ready存储中generated Part的最小版本；不覆盖queued/running/completed-unrouted。trainer以`train_version - oldest`生成max_ready_lag_updates，单位为optimizer update，不是秒、rollout轮数或所有在途工作最大lag。没有generated Part则为None；[同版logger](https://github.com/Tencent-Hunyuan/UniRL/blob/f404c9024c00bbf8a5d05315153444cee894865a/unirl/utils/wandb_logger.py)跳过None，缺点不能补成0或沿用上个值。

[共享async循环](https://github.com/Tencent-Hunyuan/UniRL/blob/f404c9024c00bbf8a5d05315153444cee894865a/unirl/trainer/async_rollout.py)在本批训练、增加optimizer版本及reset transport后取pre_sync，再按需quiesce/finish/publish/resubmit，随后取step_end；即使本步不需sync，两个名字仍分别是这两个采样点。pre_sync才记录reserved/running/unrouted/occupancy，step_end不记录这些字段，不是把它们写成零。两份状态一起通过log_rollout(step=rollout_id+1)发送到rollout/async/*，绑定rollout/step；不是两次设备采样或optimizer-step横轴。记录发生在eval/save之前，因此不能由该点认证后续eval/checkpoint成功。

collect/refill的三个headroom分别来自容量、freshness、hard boundary，扣除outstanding后换算为prompt，实际新准入取三者最小值、整批提交。日志保存的是**这次提交前**计算的余量，而非提交后的剩余容量；任一项为0足以禁止新准入，不能只凭slot空闲判断调度停滞。新表达式与原min预算再扣outstanding在正batch_size和batch对齐前提下代数等价，源码变化不是放宽准入策略的证据。

### 增量只有一个消费者

snapshot保留累计事件；drain_metrics计算“当前累计−共享baseline”后推进baseline，初值为0。额外观察者应读snapshot：两个drain调用者不各有独立窗口。连续无事件时delta为0，不能据此判断生命周期总数为0；计数随新manager重置，checkpoint不恢复。

launch计handle登记，complete计观察ready，resolve仍可能失败；dropped_groups只计成功complete-buffer过滤移除的逻辑组，既不专指staleness，也不包括所有incomplete-root丢弃。carried计成功quiesce返回任务，finished计成功finish的输入，均是事件而非去重样本。baseline在成功读取时已更新，先于logger调用；logger还会吞下记录异常并warning，所以不是W&B持久化确认后才消费的可靠事件队列。此为源码契约，不是本机注入日志失败的复现，也未据此提出修复。

### 证据范围

本机Python3.9.6仅对manager __init__/buffers/dispatch/rollout、trainer.async_rollout及utils.wandb_logger共6完整模块内存compile通过；没有导入或执行它们，没有构造假handle/训练包壳、启动后台pool、Ray、模型、设备或W&B。最小复查（Git对象须已在本地）：

```bash
GIT_NO_LAZY_FETCH=1 python3 -B - /path/to/UniRL-git <<'PY'
import subprocess, sys
head = 'f404c9024c00bbf8a5d05315153444cee894865a'
paths = ['rollout/manager/__init__.py', 'rollout/manager/buffers.py',
         'rollout/manager/dispatch.py', 'rollout/manager/rollout.py',
         'trainer/async_rollout.py', 'utils/wandb_logger.py']
for path in paths:
    src = subprocess.check_output(['git', '-C', sys.argv[1], 'show', f'{head}:unirl/{path}'])
    compile(src, path, 'exec')
PY
```

head SHA256：dispatch.py `2515cab5013f1645f7e441b28833627a766ef0b61f83a6fcc6bb8cbc9112e707`；rollout.py `25e9701e0525ed4c2cdb7b1cd0d9e6ce0c074a7a8730abbc7f1ca9ce94739db2`；async_rollout.py `b1769c6a4fb3333d472e55672dc60c5a12e12c56cbf1456e2996f799e78e5a04`。作者正文的CPU mock检查及5次100-rollout/400-update、500条W&B记录均为作者报告，PR未提交该harness，本机未复跑；报告中没有staleness drop，不用平滑曲线认证全部过滤路径。

本题以监控契约学习收束，不新建CPU替身或GPU实验队列。若M5日后独立选择匹配的async真机任务，只在[原UniRL参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)保留观测入口；缺环境或不属原任务则维持未测，不为指标说明部署训练。

## Omni 原生 IPC 的副本与导出边界

**训练rank的集体参与、向哪个rollout副本发权重、权重内容是否正确，是三个不同问题。** 2026-10-22只读[#617](https://github.com/Tencent-Hunyuan/UniRL/pull/617)，读取时open、未合并，固定head `19f263968deecdecd26cebe8966f8ba012a5575d`；API目标base为`3ecbbc32b2fc373ba30b9842032552e041e84b12`，官方compare祖先`6624b28aab3299d116ae1e3d90aaa14926676911`（ahead4/behind36）。按祖先核16文件454增92删，研究范围是Qwen-Image全参数训练与vLLM-Omni原生IPC接线，不把落后主线的差异当本PR改动。

### 每卡副本不是共享 TP 引擎

[IPCWeightSync](https://github.com/Tencent-Hunyuan/UniRL/blob/19f263968deecdecd26cebe8966f8ba012a5575d/unirl/distributed/weight_sync/full/ipc.py)以rollout的rank_local_weight_transfer选择发布者；[Omni engine](https://github.com/Tencent-Hunyuan/UniRL/blob/19f263968deecdecd26cebe8966f8ba012a5575d/unirl/rollout/engine/vllm_omni/engine.py)声明该标志。两种拓扑不能混读：

| 接收布局 | 训练端参与 | 发布者与接收者 |
| --- | --- | --- |
| 本候选Omni：每训练卡一个独立TP1副本 | 所有训练rank物化参数、交换IPC handle并做错误共识 | 每rank向自己卡上的副本发布；本地fanout须为1 |
| 原共享vLLM TP引擎 | 仍需所有训练rank参与上述工作 | rank0负责发布，接收fanout与训练world匹配 |

[VLLMNativeIPCTrainerEngine](https://github.com/Tencent-Hunyuan/UniRL/blob/19f263968deecdecd26cebe8966f8ba012a5575d/unirl/distributed/weight_sync/transfer/vllm_native_engine.py)的is_sender控制init/start/finish客户端动作，不把其他rank排除在_send_planned_packed之外。因此“只让rank0调用整个sync”会违反这条物化/collective接线；不能由只有一个发布者推导只有一个参与者。

原生路径先核同节点、每actor一个CUDA可见设备、物理UUID与接收worker按TP rank排序的UUID对应。[Omni backend](https://github.com/Tencent-Hunyuan/UniRL/blob/19f263968deecdecd26cebe8966f8ba012a5575d/unirl/rollout/engine/vllm_omni/backends/native.py)还要求恰好一个启用WTE的diffusion stage。supports_native_weight_transfer只报告发现WTE配置，不认证资源、UUID或reload实现就绪。这里是同节点CUDA IPC，不是跨节点或NPU传输支持。

### 元数据规则与真实数据物化分开

[peft_merge.py](https://github.com/Tencent-Hunyuan/UniRL/blob/19f263968deecdecd26cebe8966f8ba012a5575d/unirl/utils/peft_merge.py)抽出_base_state_dict与_iter_rollout_tensor_views，让raw_state_dict_metadata和实际raw/merged导出使用相同的embedding过滤与packed-expert命名/切片规则。metadata用全局shape创建meta tensor；wire dtype只改浮点字段，整数buffer保留原dtype。tie_word_embeddings=True按配置跳过精确名称lm_head.weight，不在此验证真实模型绑定关系。

共享模块也覆盖Qwen MoE packed专家布局：gate_up为[E,2I,H]，每expert前I行作gate、后I行作up，down为[E,H,I]。这是共享导出模块的规则，不表示Qwen-Image是MoE。metadata和CPU视图不需CUDA；实际_iter_rollout_tensors仍调用_to_full_tensor，其中有.cuda及DTensor redistribute，不能用meta检查冒充数据导出parity。

本机以完整原模块执行两个窄接口，见[固定Git对象复查脚本](../../experiments/scripts/verify_unirl_export_metadata.py)。手写真实nn.Module状态布局含embedding/head及浮点/整数buffer，不是Qwen模型：tied开关×wire dtype(None/fp32/bf16)共6组，名称顺序、shape、meta设备与dtype全部通过，整数count始终int64。另对实际CPU arange张量[E=2,I=3,H=4]核gate/up的4片与down的2片，值/名称/形状及共享storage通过；奇数gate宽、与I不符的gate宽/down宽三种输入均拒绝。没有替换原函数或模拟通信，也没有运行raw_state_dict/merged_state_dict/CUDA导出。

```bash
python -B experiments/scripts/verify_unirl_export_metadata.py /path/to/UniRL-git
```

Python3.12.13/torch2.9.1（git `5811a8d7da873dd699ff6687092c225caffcf1bb`）；原模块SHA256 `e03bf04d38966e548bf0a0f742260dc805cf29ec96270e5ffe3eb8526abc8798`，脚本校验此hash并禁止Git隐式拉取。8个修改的完整Python模块另经Python3.9.6内存compile通过；只有上述元数据/CPU视图接口实际执行，没有DTensor进程组、PEFT、Ray、vLLM、模型forward/reload或设备发现。

### 发布完成不是内容一致或事务提交

[structural_manifest](https://github.com/Tencent-Hunyuan/UniRL/blob/19f263968deecdecd26cebe8966f8ba012a5575d/unirl/distributed/weight_sync/transfer/vllm_native_protocol.py)哈希有序的name/shape/dtype/numel/nbytes，不读张量payload；相同fingerprint不是参数值逐字节相等。发送端按schema验证实际流，各训练rank比对结构manifest；Omni检查RPC报告的失败，再做rank共识。上游layerwise reload和CUDA同步由配套Omni实现负责，本轮没有复审或运行该外部实现，不能把本仓库返回检查当内容回读。

IPC内部_next_model_version从1起、成功发布后递增，是发布序号；[DiffusionTrainer](https://github.com/Tencent-Hunyuan/UniRL/blob/19f263968deecdecd26cebe8966f8ba012a5575d/unirl/trainer/diffusion.py)另外读取optimizer step count，sync/push正常返回后才调用rollout.set_version用于生成标记。每轮多次optimizer update时，两者不必相等，不能把header序号直接当optimizer版本。

失败可能已经部分覆写接收权重；失败共识后的poison/shutdown是停止使用并重建运行时，不是回滚到旧权重。Omni保留level-1 sleep，以免仅更新transformer却丢掉未发布的冻结text encoder/VAE；这也要求主机RAM留备份余量。原生preflight检查最大单tensor不超buffer，并用3×buffer+按名字估计的最大layer+headroom核空闲显存；不是全模型训练峰值或任意显卡容量保证。boot还默认设置VLLM_ALLOW_INSECURE_SERIALIZATION=1以传递pickled IPC handle，这条控制链应处于可信环境，不能当成不可信HTTP输入的安全认证。

### 配方与作者实测范围

新增YAML的1x1覆盖LoRA为null、cpu_offload=false、256MiB IPC buffer；1x2再指定两FSDP rank、colocate及train_resident。只读defaults/override，不宣称本机已完成Hydra组合或单卡运行。原生Omni路径限一个diffusion WTE stage和每卡TP1副本，不能外推AR、多WTE stage、量化或rollout TP>1。

作者报告UniRL此head搭配Omni `8607b97ea5`（[配套PR #7686](https://github.com/vllm-project/vllm-omni/pull/7686)）、vLLM0.31/torch2.13，在两张约140GiB的L20X上跑tiny/full×RPC/HTTP四个短实验：三optimizer step、每副本两次publication、版本[0,1,2]；tiny审77个source tensor、full只审3个选定tensor。HTTP只覆盖生产WTE控制请求，生成仍native，不是通用HTTP生成backend或服务CLI验收。上述报告不是本机结果，也不认证收敛；1x1配方未独立GPU验收，GPU发布失败注入未跑。作者CPU/Gloo/PEFT harness未入仓库，不能记作本机或上游已提交单测。

真实接线/权重及失败关闭缺口记[原M5 UniRL参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不部署配套分支、下载模型或新排训练。本题以副本/元数据边界学习收束，不继续造IPC替身。

## 原始 block 权重训练与冻结外围

**冻结参数不等于切断该模块的输入梯度，也不等于整个模型已经能训练。** [#593](https://github.com/Tencent-Hunyuan/UniRL/pull/593)（2026-10-23读取open draft）为MiniMax-H3增加原始`transformer_blocks`权重训练。固定head `da8d285ac529c2f06819bf3a098e3cc657ae6618`、base及实际祖先 `2d9e57fdee65ea69254fdb958d3fca9ce4872661`，ahead3/behind0；四文件仅新增42行，无已提交测试。它不是全transformer微调，也不是LoRA方案的新名字。

### 参数选择与原分片布局

[bundle](https://github.com/Tencent-Hunyuan/UniRL/blob/da8d285ac529c2f06819bf3a098e3cc657ae6618/unirl/models/minimax_h3/bundle.py)在meta/eager两条transformer构造分支之后、backend包装之前执行：

```python
if config.full_finetune_blocks:
    transformer.requires_grad_(False)
    transformer.transformer_blocks.requires_grad_(True)
```

因此text refiner、输入/输出投影和timestep模块保持冻结，只有block子树的原参数可训练；text encoder及两个VAE沿用原冻结路径。默认false只是不执行新增选择，不代表自动构造LoRA。[新配方](https://github.com/Tencent-Hunyuan/UniRL/blob/da8d285ac529c2f06819bf3a098e3cc657ae6618/examples/diffusion/minimax_h3/minimax_h3_t2va_trainside_blocks.yaml)同时设true和`backend.lora_cfg: null`：后者单独取消backend的adapter注入，不能只把bundle开关当成对所有LoRA/EMA配置的互斥保护。

继承配方只wrap `MiniMaxH3TransformerBlock`，保留`root_wrap:false`、`cast_forward_inputs:false`、BF16计算与FP32 trainable masters。外围投影/timestep原有混合精度不由root统一转换。[原fsdp_wrap](https://github.com/Tencent-Hunyuan/UniRL/blob/da8d285ac529c2f06819bf3a098e3cc657ae6618/unirl/train/backend/fsdp/wrap.py)在默认进程组已初始化且world_size>1时，拒绝仍可训练却不属于DTensor分片组的参数；冻结外围解决这项布局前提，但不证明collective、权重装载或完整反向正确。`trainable_module()`仍返回整个transformer；原优化器另以`requires_grad`过滤参数，不按返回模块的名字推定所有权重可训练。

### 冻结输出头仍要传递梯度

固定[vendor前向](https://github.com/Tencent-Hunyuan/UniRL/blob/da8d285ac529c2f06819bf3a098e3cc657ae6618/unirl/models/minimax_h3/vendor/transformer_minimax_h3.py)在block之后仍执行`norm_out`、视频/音频输出投影和行选择，没有为冻结头插入`no_grad`。冻结叶子参数只是不为它积累参数梯度；只要输入需要梯度，这些运算仍能把链式法则传回block。直接把整个冻结输出头放进`no_grad`会切断这条链，不能作为等价省显存办法。`train()/eval()`控制的模块行为又是另一件事，新增两条选择语句没有切换模式。[PyTorch 2.9 autograd说明](https://docs.pytorch.org/docs/2.9/notes/autograd.html#locally-disabling-gradient-computation)

这还关联首个block的checkpoint：冻结输入投影且原输入不求导时，输入activation可以没有`requires_grad`，但block参数仍需训练。原FSDP checkpoint helper用`use_reentrant=False`，不要求输入中先有需梯度tensor；重入版本有这项要求。[PyTorch checkpoint说明](https://docs.pytorch.org/docs/2.9/checkpoint.html) 这是原helper已有行为，不是本PR新增修复或本项目提出的替代方案。

### 本机 CPU 控制与作者真机边界

[复查脚本](../../experiments/scripts/verify_unirl_block_freeze.py)固定Git对象，无隐式fetch；Python3.12.13 / torch2.9.1，显式CPU FP64。只将原选择条件/两语句应用到声明的三Linear数学夹具，输入2、外围权重3/7、block权重5，不是tiny H3。完整原`OptimizerConfig`与optimizer模块正常执行，预加载叶子模块绕过UniRL父包；checkpoint仅取原两个helper的AST，未执行FSDP wrapper。

| 控制 | 实际结果 |
|---|---|
| 开关关闭与开启 | 关闭保留三份可训练参数；开启仅保留block，Parameter身份与train模式不变 |
| 冻结外围的正常前向/反向 | 输出210，block梯度42，与全可训练参考一致；外围grad为None |
| 原optimizer factory及一次AdamW | lr=.01、weight_decay=.1，只有block进入参数组/状态，5更新为4.985000000002381，外围仍3/7 |
| 输出头改用no_grad的负控制 | 输出无反向图，backward按预期拒绝，block没有梯度 |
| 原非重入helper与重入负控制 | 前者保留梯度42；后者在本例无需梯度输入时告警并失去反向图 |

```bash
python -B experiments/scripts/verify_unirl_block_freeze.py /path/to/UniRL
```

两次执行结果一致。修改的两个完整Python模块另仅内存compile；optim、FSDP wrap和vendor transformer的base/head Git blob逐项相同。冻结不保证外围激活/反向计算免费，也不证明实际H3参数规模、混合精度、meta物化、LoRA默认输出、FSDP/TrainStack、replay、恢复或学习质量。这里没有运行原H3或作者外部harness、安装新依赖、设备探测/模型加载。

作者报告tiny H3参数选择、CPU/Gloo FSDP/TrainStack更新与恢复成功，但harness和原始日志在仓库外，本项目未复跑。[PR正文及新增README](https://github.com/Tencent-Hunyuan/UniRL/pull/593)明确：原checkpoint `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`在8×H100 80GB、256×384/124帧、10 CPSSDE步、FP32 masters及resident VAE条件下能load/generate，首次backward仍OOM；开启expandable segments的后一次也未到Adam，报告75.94GiB allocated与1.43GiB reserved-but-unallocated。16卡更新验证待完成，不能把LoRA配方历史收益或加载成功挪为原始block权重训练成功，更不能外推4090可跑。真机前提补[原M5入口](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不自动启配方、扩大卡数或演化offload方案。

源SHA256：bundle `9d7681dda134e5a38c1bd0b20c07330193ace5d93104250a9d542e92c847e0cc`；config `f84ce6b98089e73f49016b22a977f5b002192e1220f32df6c9d644c7a3b80864`；optimizer `a0f0fcf664da5049c1f3652d2e5f3fe1e469e721ea91c0b613294374f4a5da05`；wrap `0a0a6ec9f587b2ce7fe26f8bc759ca3a701ff1b9b67e769e19b476120ca5a915`。

## 全词表条件熵与有效 token 汇总

**一次采样的负log-prob不是该位置的全词表熵，微批均值的均值也未必是全局token均值。** [#563](https://github.com/Tencent-Hunyuan/UniRL/pull/563)（2026-10-23读取open）新增可选AR熵监控。固定head `00e5597f6ff6de639d7997a2ea21266d688e9f72`、base/实际祖先 `94e8f26b8768a3695b5985c2fa072e07d9dbfa0d`，ahead3/behind0；13文件270增43删，无提交测试。本节是已有方案的源码学习，不是本项目新增修复或对熵坍塌故障的定性。

### 监控哪个分布

[Qwen3 replay](https://github.com/Tencent-Hunyuan/UniRL/blob/00e5597f6ff6de639d7997a2ea21266d688e9f72/unirl/models/qwen3/ar.py)在原LM-head分块内对输出先`.float()`再除以温度，求`p = softmax(logits/T)`，以`sum(entr(p))`计算完整词表条件熵，单位nats。`entr(0)=0`可处理softmax下溢的零概率，不需用`0*log(0)`；这不修复非有限logits。[PyTorch entr定义](https://docs.pytorch.org/docs/2.9/special.html#torch.special.entr)

正温度沿用sampling_temperature，否则回退T=1；没有应用top-k/top-p截断。因此greedy rollout时该指标可以非零，它不是确定性采样器的熵。对二元`p=(.9,.1)`，条件熵约0.325083，而采到第一项的surprisal仅0.105361；只有在同一分布下取适当期望时，二者才有熵的关系，不能把单次选择或另一rollout分布的样本均值直接当完整词表求和。

`_categorical_entropy`使用`no_grad`，结果detach后作为`ReplayResult.entropy`返回；原log-prob仍走训练图，`ReplayResult.logits`不因此被填充。GRPO/GSPO/DPPO/DRPO/CPPO/PPO的`monitor_entropy`默认false，开启才请求额外字段，不添加entropy loss。仅输出每token标量不等于没有词表临时量：每块仍多做全词表softmax/entr，开销与词表、块形状及后端有关，不能外推作者短序列时延为普遍零成本。外层replay及其chunk checkpoint循环本机未执行。

### 分母是有效位置的观察次数

[原metric helper](https://github.com/Tencent-Hunyuan/UniRL/blob/00e5597f6ff6de639d7997a2ea21266d688e9f72/unirl/algorithms/base.py)接收packed response entropy，对`loss_mask`转bool筛位置；None表示全部位置，包含实际存在的EOS。mask不是软权重：非零浮点值也转True。prompt/padding由replay布局排除，tool等位置由原mask控制，不能把helper视为独立核验数据身份的入口。

每个微批保留`policy_entropy_sum`及`policy_entropy_count`，mean为sum/count；空响应或全False mask的sum/count均0，mean用0占位，不表示观察到确定性策略。masked_fill去掉非活跃NaN/Inf，活跃NaN仍传播，不是健康认证或异常修复。[原聚合器](https://github.com/Tencent-Hunyuan/UniRL/blob/00e5597f6ff6de639d7997a2ea21266d688e9f72/unirl/utils/metrics.py)仅对这两个键求和并重算mean，其他普通数值键仍按原均值规则处理。

本机合成微批共用4词表：一枚位置在前两项均匀，熵ln2；另三枚位置全词表均匀，熵ln4。对同样的两份字典，旧通用聚合器等权mean为1.039720774；新聚合器sum=4.852030277、count=4，token mean为1.213007569，分层聚合一致。这是给旧/新聚合函数同样新增指标字典的对照，不代表base原本已有此熵监控入口或完整训练bug已复现。

`TrainStack`在微批结束后以两次stack/tolist搬回sum/count，再交原backend的loss-reduction mesh求和；当rank_info存在时，仅TP rank0、SP rank0且pipeline最后阶段贡献，其他副本置零。CPU字典检查不认证真实mesh/副本去重。跨optimizer updates汇总时，同一数据位置在不同策略版本下会再次计数：结果是各次条件熵观察的加权平均，不是唯一语料token计数，也不是平均策略分布的熵。

### 本机局部复查与真机范围

[复查脚本](../../experiments/scripts/verify_unirl_policy_entropy.py)固定本地Git对象、禁止隐式fetch；Python3.12.13 / torch2.9.1，全部显式CPU。AST原样选择两个LM-head chunk函数、原温度赋值、entropy和metric helper；完整原ReplayResult与base/head metrics模块执行。LM-head采用声明的Linear数学夹具，不伪装Qwen backbone，不运行含CUDA探测的外层forward或补模型替身。

- FP32/BF16 × T=-2/0/.5/1/2 × flat/padded两种chunk形状，共20组：监控开关前后log-prob、输入及Linear参数局部梯度逐位相同，entropy为detach FP32；另对实际head输出的FP64全词表参考，以atol/rtol=3e-7核对。
- mask转bool、detach、空/全masked归零、活跃NaN传播、类型/缺字段/两类形状错误的4项拒绝均符合原helper；未更改断言或soft-fail。
- 1/3长度微批的加权与分层聚合结果如上；11个改动Python完整模块仅内存compile，不把这项检查当成六种完整算法或原作者测试通过。

```bash
python -B experiments/scripts/verify_unirl_policy_entropy.py /path/to/UniRL
```

两次执行一致。没有Qwen模型、packed/padded外层replay、checkpoint循环、loss/optimizer、FSDP/TrainStack/Ray、TP/SP/DP collective或GPU/NPU运行，也未安装依赖。作者另报tiny L4、Qwen3-0.6B单/双A100与Ray/FSDP一次更新证据，原harness在仓库外；这些不是本项目复跑，也不覆盖ARTrainer的完整rollout/reward/sync循环或VeOmni。后续真机核对仅补[原M5参考](../../experiments/handoff-m5.md#unirl-联合路径的资源与身份边界)，不部署候选或把它当#8250根因诊断。

SHA256：Qwen3 ar `49557b0cf37b0823940307531f99efa2505242411492b2fe506f5f14387e5972`；metric helper所在base `da4a741a0315c0ad8f530c7952593c8bade2bdaffffcb9e68730af527d306b30`；新metrics `2897870ce27577033b2d7963df9dbc71178fef3a523b53924d786172ae585626`，旧metrics `f19ac2211065ea4927927648a3ec108ecb5b1b7bd7e8f1d3b73dd3ef198beab9`。

## 下一步

继续以 UniRL 为高优源码/潜在贡献轨道，优先围绕真实配置与类型契约选题；不是已找到 bug 或获得对外发布许可。真机验证由 M5 独立选择获准且资源匹配的配方，条件不符时保留未测，不为复现默认 HI3 直接启动大模型或升级环境。
