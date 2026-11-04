# verl #6067 重构后的配置消费点体检：一条回归变一类，但真失效只有三条

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（RL 训练框架）

**【贡献点】** 把上周发现的 `tiled_mlp` 单条回归，扩成对 #6067「deprecate workers, migrate to engines」全部 worker 配置读取点的体检：**新发现 `actor.freeze_vision_tower` 同样失效**（3 年历史特性的接线被同一次重构丢掉），并把 30 条文本候选逐条判成真失效/改名/搬家，写死判据与负向自检。[#6067](https://github.com/verl-project/verl/pull/6067)

**【类型】** 问题诊断（B 类：主动发掘）

**【价值评估】** 7 / 10 — 新找到一条独立失效（VL 模型放开视觉塔训练会静默不生效，且用户以为已省显存而实际没省），但证据仍是**静态代码取证**，未在 GPU 上跑训练验证；且这一批只覆盖 FSDP/Megatron worker 删除前的配置读取点，不覆盖行为回归（如 MTP 在 FSDP 上从"报错"变"静默清零"）。

**【进展】** 2026-11-04 M1 本地完成，**未触上游、未提 issue**。结论与复现脚本见 [实验目录](../../../../experiments/verl-6067-config-consumer-scan/readme.md)（脚本不复制一份在本目录，避免两处分叉）。

**【大白话讲明白】**

verl 今年四月做了一次大重构，把旧的"worker"文件整份删掉、换成"engine"。删文件时，有些控制"要不要开某个省显存/省算力开关"的代码跟着一起没了，但开关本身还留在配置文件里——用户照着文档写了 `freeze_vision_tower=True`，程序不报错、照常训练，只是这个开关**什么也不做**：视觉塔还在训练、显存一点没省。

上周已经确认 `tiled_mlp` 是这么丢的。这次把当时删掉的那几个文件里**所有**读配置的地方列出来（102 处），一条条问"新代码里还有没有人读它"。答案分四种：真失效（3 条）、改了名字照样生效（绝大多数）、搬到了别处（1 条）、以及一处旧行为被悄悄换了语义（MTP 在 FSDP 上原来会报错，现在静默清零）。真失效的三条里有两条是这次新发现的。

**【What–Why–How】**

- **What：** #6067（`044bbba2`，2026-04-20，BREAKING）删除 `verl/workers/fsdp_workers.py`、`megatron_workers.py` 后，旧文件里有 102 个 `config.<role>.<key>` 读取点；其中哪些在 main 的引擎侧没有对应消费者。
- **Why：** 配置键留在 yaml/dataclass 里意味着**用户可以设**，删除读取点意味着**设了也没用**。这种行为对用户完全静默（无报错、无警告），与"配置项写了就生效"的直觉相反；同类问题已单独出现过一次（`use_prefix_grouper`，TM-005），说明是重构方法的系统性问题而不是个例。
- **How：** ① `scan_consumers.py` 从 `044bbba2^` 的两个 worker 文件抽出全部 `config.*` 读取点，在当前树的 `verl/workers/engine/**`、`verl/trainer/**` 里查叶子名是否被读；② `verify_findings.py` 对 30 条文本候选逐条写死判定（旧读取点位置、当前消费证据、期望结论），跑一遍必须 0 不一致；③ 负向自检 `--selftest` 注入一个读取点，判据必须翻转。**边界**：纯静态；不覆盖行为回归；只覆盖被删的两个 worker 文件的读取点，不覆盖 #6067 同时改的 71 个文件里的语义变化（MTP 那条是逐个人工读出来的）。

<!-- contribution-summary:end -->

## 一、结论

**104 个配置读取点**（`fsdp_workers.py` 54 个路径 + `megatron_workers.py` 50 个，去重后）里，文本级扫描报出 30 条"字段还在、引擎侧查不到读取"，逐条人工核对后：

| 判定 | 条数 | 说明 |
|---|---|---|
| **真失效**（用户能设、没人读） | **3** | `tiled_mlp`、`freeze_vision_tower`、`use_prefix_grouper` |
| 改名/重建后照常生效 | 26 | 主要是 `actor.fsdp_config` → `actor.engine`（dataclass 整体换了位置）、`log_prob_*` → `ppo_*`（engine_workers 改名）、`rollout.free_cache_engine` 等由 rollout 层自己读 |
| 搬到了别处 | 1 | `model.mtp.enable=False` 时的 `num_nextn_predict_layers` 清零，从 worker 搬进 `HFModelConfig.__post_init__` |

真失效的三条里，`tiled_mlp` 与 `use_prefix_grouper` 上周已单独取证；**`freeze_vision_tower` 是本轮新发现**。

## 二、三条真失效

### 2.1 `actor.freeze_vision_tower`（本轮新发现）

**是什么。** 多模态（VL）模型训练时，视觉编码器（vision tower）通常不需要训练——它只负责把图片编码成 token。`freeze_vision_tower=True` 让 vision tower 的参数 `requires_grad=False`，好处是省掉这部分激活/梯度显存，配合 `use_orig_params` 还能让 FSDP 少分一片。

旧实现（`044bbba2^:verl/workers/fsdp_workers.py`）：

```python
# :545-553，actor 侧
if self.config.actor.get("freeze_vision_tower", False):
    vision_tower = get_vl_model_vision_tower(actor_module)
    if vision_tower is not None:
        vision_tower.requires_grad_(False)
        self.use_orig_params = True
```

`get_vl_model_vision_tower()` 定义在同一文件（`:133-145`：取 `model.visual`，兼容 transformers ≥4.52 与更早版本），**随文件一起删除，main 里没有任何替代**：`git grep -n vision_tower HEAD -- "*.py"` 全仓只剩 `verl/workers/config/actor.py:161` 的字段声明一行，没有任何 Freeze 逻辑。

**为什么算"用户能设但没人读"。** main 里：

- `verl/workers/config/actor.py:161` 仍声明 `freeze_vision_tower: bool = False`；
- `verl/trainer/config/actor/actor.yaml:51-52` 仍是
  ```yaml
  # Whether to freeze vision model, if set true, it will be freeze vision model
  freeze_vision_tower: false
  ```
- `docs/ascend_tutorial/dev_guide/model_dev/parameter_and_metrics.md:45` 还把它列在参数表里，标着"是否冻结视觉塔（多模态模型）"；
- **`verl/workers/**` 与 `verl/trainer/**` 里没有任何 .py 读它**（`verify_findings.py` 的 `lost` 判据，排除 config 声明本身）。

⇒ 用户照 yaml 注释和 Ascend 文档设 `freeze_vision_tower=true`，训练照跑、不报错、视觉塔照常训练。**这不是"没实现"，而是一个 2025-09 合入（PR [#3178](https://github.com/verl-project/verl/pull/3178)，issue #2526）的真实特性被一年后的重构删掉了接线。**

### 2.2 `tiled_mlp`（上周已取证，本轮复核结论不变）

见 [tiled-mlp-unwired](../verl/evidence/tiled-mlp-unwired/readme.md)。本轮补充一条口径：`HFModelConfig`（`:141-142`）与 `FSDPCriticModelCfg`（`verl/workers/config/critic.py:282`）**各自**还声明着 `tiled_mlp`，所以 actor 与 critic 两条路径都"能设、不读"。

### 2.3 `use_prefix_grouper`（TM-005，本轮只作同源复核）

`verl/trainer/ppo/ray_trainer.py` 仍在读它（决定组级均衡与批内重排），而 FSDP 引擎 `_build_module` 不把它传给 `apply_monkey_patch`。**这条不是"完全没人读"，而是"读了但读的人只付代价、不拿收益"**——比另外两条更隐蔽，所以单独登记在 TM-005。

## 三、一条不是失效、但语义变了：MTP 在 FSDP 上不再报错

旧 `fsdp_workers.py:429-432`：

```python
if self.config.model.get("mtp", {}).get("enable", False):
    raise NotImplementedError("Right now,  MTP is not supported in FSDP")
else:
    if hasattr(actor_model_config, "num_nextn_predict_layers"):
        actor_model_config.num_nextn_predict_layers = 0
```

新代码里这段清零搬进了 `verl/workers/config/model.py:235-243`：

```python
if not self.mtp.enable:
    if hasattr(self.hf_config, "num_nextn_predict_layers"):
        self.hf_config.num_nextn_predict_layers = 0
    ...
```

**搬对了**（清零照旧发生，而且覆盖 qwen3.5 的 `mtp_num_hidden_layers`），但**那句 `raise NotImplementedError` 没了**：在 FSDP 引擎上开 `model.mtp.enable=true`，不会再报"FSDP 不支持 MTP"，而是……走 FSDP 引擎、MTP 相关字段被下游忽略。

这一条**不列入失效**（因为不影响任何默认配置，且是否有意为之需要维护者判断），但值得作为"重构顺手改变了行为契约"的例子记下来：同一类改动里，"删代码"和"搬代码"要分开看，扫文本的脚本看不见这种差别。

## 四、判据与自检（本轮自己踩的坑）

**① 第一版把配置声明当成了读取点。** `lost` 判据要求"引擎侧没读"，但 `verl/workers/config/*.py` 里的 `tiled_mlp: dict = ...` 本身就是一次文本出现。第一版把 `verl/workers/config/` 也算进搜索范围，于是 3 条真失效全部自证"有人读"，报"对不上"。修法：`lost` 的读取搜索**排除 `verl/workers/config/` 前缀**，另用一份独立的 `config_declarations` 查询确认"字段仍在"。

**② 自检第一次没通过，暴露的是正则问题。** 自检往假仓库里写 `apply(use_tiled_mlp=True)`，判据却报"没有读取"——因为 `\btiled_mlp\b` 在 `use_tiled_mlp` 里匹配不上（`_` 是单词字符）。同一缺陷会让真实扫描漏掉 `tiled_mlp_config`、`_tiled_mlp` 这类派生名。改成子串匹配，宁可多报（误报由逐条人工复核兜底），自检随即通过。

⇒ **"注入一个读取点、判据必须翻转"这条自检，失败时暴露的是工具而不是被测对象**；如果不写它，第一节那张"26 条是改名、3 条真失效"的表就没有分辨力证明。

## 五、影响面与已知未做

- **`freeze_vision_tower` 影响谁：** 在 FSDP/VeOmni 引擎上训练 VL 模型、且显式开了这个开关的用户。默认 `false`，所以不受影响的占多数；A800/H20 上开它主要是为省显存，失效后**显存占用回升**，但不会报错——用户可能表现为 OOM 或吞吐下降而归因错误。
- **未做：**
  - 没有在 GPU 上跑「开/关该开关」的显存对照（需要 VL 模型 + 训练环境，属 M5 真机范围）；
  - 没有扫 #6067 同时修改的另外 71 个文件里的语义变化，只逐个人工核了 MTP 一条；
  - Megatron worker 的 50 个读取点里有 26 条判为"改名/生效"，只抽查了 `mtp_loss_scaling_factor`、`log_prob_*`、`fsdp_size` 三条的证据链。
- **对外动作：** 未建 issue、未评论、未开 PR。是否把 `freeze_vision_tower` 并入 TM-006 的对外材料（或单独提一条），待用户决定。

## 六、复现

```bash
# 1) 文本级候选（30 条）
python3 scan_consumers.py --repo <verl 仓库> --json scan-result.json

# 2) 逐条判定（必须 0 不一致）
python3 verify_findings.py --repo <verl 仓库> --json findings.json

# 3) 判据负向自检（必须打印"自检通过"）
python3 verify_findings.py --selftest
```

本轮实测（verl main `1a9e711e`，父提交 `044bbba2^`）：`scan_consumers.py` 报 102 个读取路径 / 30 条候选；`verify_findings.py` 7 条判定 **0 不一致 exit 0**；`--selftest` 通过。
