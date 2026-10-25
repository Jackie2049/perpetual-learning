# verl Checkpoint Management 源码级深度阅读

> 2026-06-15 | 源码: verl/utils/checkpoint/ + verl/checkpoint_engine/ + verl/model_merger/ + verl/trainer/ppo/ray_trainer.py
> 核心: 2层架构(CheckpointManager→磁盘持久化+CheckpointEngineManager→实时权重同步); FSDP每rank独立shard+HF export(rank0 gather); Megatron dist_ckpt+async save; CheckpointEngineManager 5后端(naive/NCCL/NIXL/Kimi/Mooncake/HCCL); FSDPModelMerger离线合并→HF; GRPO只存actor+dataloader→省critic

2026-10-22 补充：[保留目录与完整恢复的边界](#保留目录与完整恢复的边界)固定于 verl `1876b06d`。`max_ckpt_to_keep=1` 保存前保留旧目录，成功登记后才删旧；目录保留、保存完成通知与可恢复性是不同保证。下方早期架构段落的版本和行号不自动更新为本次验证范围。

2026-10-24补充：[异步保存能力与恢复指针](#异步保存能力与恢复指针)核对#7955及旧DCP候选#4311；原查找函数的一例测试与五组真实文件系统控制说明“找到目录”并不验收恢复内容，异步写入能力也不自动提供tracker发布。

2026-10-25补充：[Megatron 调度进度与超参数恢复](#megatron-调度进度与超参数恢复)用官方调度器和真实CPU optimizer区分进度、超参数来源与参数组覆盖；六组控制不等于完整分布式checkpoint恢复。

## V1 到期前保存的触发与计时范围

[verl #7757](https://github.com/verl-project/verl/issues/7757)指出默认V1 trainer没有接入ESI到期判定，已有报告者YeonwooSung的[#7760](https://github.com/verl-project/verl/pull/7760)。2026-10-25核open、非draft、未合并，普通／行内评论均0；编号及esi_redundant_time查重已有该覆盖，不另实现。固定head `16cd240d14cf0928961492fbb017b8a4356905bd`、实际共同祖先 `a9f2985159536a607211dcac730d3f5d55028950`，两提交、两文件211增7删；直接父提交`6043685b`已经有首版修复，API目标base`3f9d4c0d`也不能替代本次比较祖先。

候选将原频率／末步保存条件收进`_maybe_save_checkpoint`，再加入共享`should_save_ckpt_esi`。`save_freq > 0`仍是写盘前提；默认配置为−1，设了冗余时间也不自动开启保存。`__init__`和每次`fit()`起点都把max_steps_duration设为0。fit中的保存判断位于step计时块内，结束计时后才记录历史最大值，再进入validation，因此当前判断使用此前步骤的历史，历史值含保存耗时但不含验证。共享helper的保存耗时估计仍默认60秒，不等于实测写盘时间，也不是必能赶在回收前完成的保证。

### 两个提供方不是同一到期条件

候选没有修改[共享helper](https://github.com/verl-project/verl/blob/16cd240d14cf0928961492fbb017b8a4356905bd/verl/utils/checkpoint/checkpoint_manager.py)，其完整函数AST在共同祖先、head及固定main `3c6d9b43`相同。本机Python3.12.13用原函数和真实系统时钟，临时环境变量退出后恢复，得到以下有限控制：

| 到期输入 | max_steps_duration | 原helper结果 |
| --- | --- | --- |
| Volcengine MLP距现在30秒 | 0 | False |
| SageMaker AWS距现在30秒 | 0 | True |
| AWS已过期30秒 | 0 | True |
| AWS非法整数字符串，且候选save_freq=−1 | 0 | 解析ValueError，未调用保存 |

MLP分支要求剩余时间>0、历史最大耗时>0且剩余时间≤阈值；非法float字符串返回False。AWS分支只比较时间差小于阈值，没有前两项门槛，且整数解析异常没有同样捕获。这是已存在helper的不同契约，不是#7760新加的回归；这里不擅自统一云厂商语义。候选在检查save_freq前调用helper，所以“禁用写盘”不等于“跳过到期参数解析”。PR所说的“首次未记录耗时就不受ESI保护”应限于MLP分支；常规末步／周期保存仍可触发。

### 原测试通过与实际接线的区别

[原12个测试体](https://github.com/verl-project/verl/blob/16cd240d14cf0928961492fbb017b8a4356905bd/tests/trainer/ppo/v1/test_v1_esi_save_on_cpu.py)隔离执行全部通过。原测试自己用SimpleNamespace、真实OmegaConf及MagicMock保存方法；本机绑定未改的两个原方法，接原helper及原CPU fallback计时器／真实codetiming，没有完整PPOTrainer或写盘。原fixture隔离MLP／AWS环境，但设置到期变量的用例均走MLP；这12项不认证AWS行为、文件完成或恢复。

源码核对确认head的fit确有保存调用及计时后的记录调用，祖先与固定main则没有这两个调用。**这是调用位置的静态证据，不是执行fit。** 当前测试文件的说明提到`test_fit_records_the_step_duration`，实际12个定义中没有它；stub也自行提供max_steps_duration，从未执行真实`__init__`。因此这12项不能替代fit调用／初始化守卫，不能据PR正文的mutation报告认定当前文件已具备这些回归保护。

**供用户审视的评审要点**：请已有作者收窄首步／计时描述的提供方范围，补回声明的fit和初始化接线守卫，并将AWS继承行为与本次V1接入分开说明。是否改共享helper需单独决策，不把历史差异塞进竞争修复。反馈只留个人项目，任何上游评论、PR修改或既有PR-head推送仍逐项待用户明确同意。

本机未原生导入trainer／TransferQueue、运行全目录测试、真实云实例或模型，也没有checkpoint文件和恢复验收。两版trainer／helper及head原测试／CPU timer共六份完整快照compile通过；命令、版本和校验值见[当日日志](../../diary/2026-10-25.md)。本轮只解决触发契约与证据范围，不新排M5算力任务；保存完成和恢复指针仍见[原专题](#异步保存能力与恢复指针)。

## Megatron 调度进度与超参数恢复

恢复训练时，继续到哪一步与使用哪套调度超参数是两个问题。[#7953](https://github.com/verl-project/verl/issues/7953)已有报告者的[#7954](https://github.com/verl-project/verl/pull/7954)，2026-10-25读取open、非draft；固定head `53e632859898a248f89c8dd7e9abbf56e2248944`，base及实际祖先 `fbb4b3a8bf636f290c9c59fc346f756849e9c241`，5文件111增16删。不另开竞争修复。

候选在Megatron普通dist checkpoint与Megatron-FSDP两条加载路径中，只要payload包含scheduler且对象存在，就调用 `load_state_dict`；开启 `use_checkpoint_opt_param_scheduler` 时仍要求payload必须有该字段。两处仍嵌在 `should_load_optimizer` 内，不是改成从extra恢复；FSDP的存储位置不能直接套给Megatron。[构造接线](https://github.com/verl-project/verl/blob/53e632859898a248f89c8dd7e9abbf56e2248944/verl/utils/megatron/optimizer.py)把该flag与其逻辑反值分别传给官方scheduler的use-checkpoint和override选项，正常推进方法则调用 `step(1)`。

### 选择超参数不应丢掉计数

官方MCore0.19.2固定提交 `4b4acac9a1d28ea6829c8d4f566d75698a21249d` 的[完整调度器](https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/megatron/core/optimizer_param_scheduler.py)由 `_check_and_set` 选择其管理的超参数，而 `num_steps` 在外部经 `step(increment=...)` 恢复。这里不能把flag理解为“False就不加载”。

本机采用warmup 2、总长10的线性调度，min/init LR为0、constant WD为0.1；数值1和2只为易核算的控制值，不是训练超参数建议。旧scheduler的max LR=1，在计数4保存；新对象max LR=2，从计数0开始。真实CPU SGD参数组、原scheduler和 `torch.save/load(weights_only=True)` 得到：

| 恢复方式 | 恢复后计数 | 恢复后全局max LR | 参数组当前LR | 再推进1后的LR |
|---|---|---|---|---|
| use-checkpoint=True | 4 | 1 | 0.75 | 0.625 |
| use-checkpoint=False、override=True | 4 | 2 | 1.5 | 1.25 |
| 不调用load | 0 | 2 | 0 | 1.0，此时计数仅为1 |

前两行的下一步分别与对应超参数下的连续调度一致。另将两flag均设False、当前与保存max LR不同，会在超参数检查时抛AssertionError；这不是verl默认配对，因为verl会同时传入互补的override标志。

### 新对象与参数组也是恢复条件

该版本的加载通过增加计数，而不是无条件赋值。本例先恢复到4并推进到5，再对同对象加载保存计数4，结果为9、LR为0.25；不能把它当幂等的“重新设回第4步”接口。冷恢复应核新对象初值，不能只检查checkpoint里存在num_steps。

全局超参数也不是实际LR的唯一来源。原 `get_lr` 优先读取optimizer参数组的 `max_lr/min_lr`，再回退scheduler的全局字段。有限反例中，旧SGD组保存max_lr=1，新对象配置2，按候选顺序先恢复optimizer组、再恢复scheduler并保留当前全局超参数后，全局max_lr仍为2，但参数组max_lr被恢复为1，计数4时实际LR为0.75。**这是普通SGD附加字段的真实状态控制，不是已证明Megatron分布式optimizer存在同一分组回归**；它说明完整验收需同时观察参数组与全局值。[PyTorch加载文档](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.load_state_dict.html)也提示初始化和状态加载次序会影响恢复后的学习率，不能跨scheduler实现照搬次序。

### 验证范围与接续

Python3.12.13／torch2.14.1，六组有限控制通过。官方scheduler、rank日志和Slurm环境helper三个原模块完整执行；仅用包壳和utils重导出连接原日志函数，不是原生导入整个MCore包。没有假scheduler或optimizer；真实torch临时状态文件已自动清理，但没有执行SGD更新、模型、Megatron optimizer、manager load或分布式保存恢复。五份候选相关Python另做AST/compile，连同三个官方模块共八份语法检查；一次性harness留工作目录，复现输入与hash见[日志](../../diary/2026-10-25.md)。

作者原manager测试mock底层checkpoint加载和scheduler对象，仅检查调用payload；本轮未复跑其43/57测试报告，也未执行Megatron-FSDP第二路径。真实完整后端的进度、参数组、下一次更新、optimizer/RNG与数据位置一致性仅补入[M5原checkpoint参考](../../experiments/handoff-m5.md#verl-训练-checkpoint-发布参考)，不新增训练队列或自动应用候选；上游写入及已建PR的fork head推送仍逐项待用户明确同意。

## 1. Checkpoint 2层架构

2026-10-23 补充：[文件系统权重同步候选 #8355](../verl-v1-checkpoint-engines-deep-reading.md#文件系统权重同步的发布与清理)说明两层应按用途而非“是否写文件”区分。临时 safetensors 在 rollout 加载后删除，不含完整恢复状态；原 helper/receiver 的本地 I/O 控制不认证共享挂载或 server 安装。下方后端数量和早期代码仍按原日期理解。

```
★ ★ verl checkpoint = 2层系统!

Layer 1: CheckpointManager → 磁盘持久化
  → BaseCheckpointManager → abstract
  → FSDPCheckpointManager → FSDP1/FSDP2 per-rank shard + HF export
  → MegatronCheckpointManager → dist_ckpt + mbridge HF export

Layer 2: CheckpointEngineManager → 实时权重同步
  → 训练→rollout GPU 权重传输
  → 5后端: naive/CUDA IPC/NCCL/NIXL/Mooncake/HCCL
  → 不是磁盘checkpoint → 是live weight sync!

★ 关键: CheckpointManager = 保存到磁盘 → 恢复训练 → 持久化
  CheckpointEngineManager = 训练→rollout权重同步 → 非训练恢复协议；传输可经内存/网络或临时文件
```

## 2. FSDPCheckpointManager — Per-Rank Sharded Save/Load

```
fsdp_checkpoint_manager.py (57-395):

★ ★ Save (220-395): 每rank独立保存

每rank保存3个文件:
  model_world_size_{W}_rank_{R}.pt → sharded model state dict
  optim_world_size_{W}_rank_{R}.pt → sharded optimizer state dict
  extra_state_world_size_{W}_rank_{R}.pt → lr_scheduler + RNG(cpu/numpy/random/cuda)

FSDP1 context:
  with ShardedStateDictConfig(offload_to_cpu=True):
    → FlatParameter shards → torch.save per rank

FSDP2 context:
  with nullcontext(): → DTensor local shards → torch.save per rank
  → FSDP2 state_dict()直接返回DTensor local shard → 不需要ShardedStateDictConfig!

★ ★ HF Export (287-395): rank 0 only + optional

if "hf_model" in checkpoint_save_contents:
  → rank 0: get_fsdp_full_state_dict(model, offload_to_cpu=True, rank0_only=True)
  → FSDP1: FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
  → FSDP2: get_model_state_dict(model, full_state_dict=True, ...)
  → init_empty_weights() → AutoModelForCausalLM.from_config() → 空骨架
  → save_model.save_pretrained(hf_local_path, state_dict=state_dict) → safetensors!

rank 0 always saves:
  huggingface/ → model_config + tokenizer/processor + generation_config

★ LoRA: lora_train_meta.json → {r, lora_alpha, task_type} → 下游恢复PEFT config

★ Load (138-218): 每rank独立加载
  torch.load(model shard) → model.load_state_dict()
  torch.load(optim shard) → optimizer.load_state_dict()
  torch.load(extra shard) → lr_scheduler + RNG恢复
  ShardedStateDictConfig(offload_to_cpu=True) → CPU加载→避免GPU OOM!
```

### LoRA only 的键完整性与加载副作用

LoRA-only恢复可以缺少冻结基座权重，不能因此忽略缺失的adapter张量。[verl #8350](https://github.com/verl-project/verl/pull/8350)补的是这层区别，而不是完整恢复事务。2026-10-23读取open、非draft、作者aoshen02；固定head `19b89e7100c06d7731247545274b117887ad8d38`，base及共同祖先 `a4b9354968705a441fb7da7367924b15b902e771`，ahead1/behind0，两文件29增1删。已有作者实现，不另开竞争修复，也不把候选当主干行为。

原[键判定](https://github.com/verl-project/verl/blob/a4b9354968705a441fb7da7367924b15b902e771/verl/utils/checkpoint/checkpoint_manager.py)要求字典非空、所有键均包含 `lora_` 或 `.adapter_`。进入该分支后，[FSDP manager](https://github.com/verl-project/verl/blob/19b89e7100c06d7731247545274b117887ad8d38/verl/utils/checkpoint/fsdp_checkpoint_manager.py)用strict=False加载模型：原来只拒绝unexpected_keys，候选再从missing_keys中过滤adapter键并拒绝。检查依赖load_state_dict返回的诊断；返回None时按空missing列表处理，不独立重算键清单。遗漏的base键仍允许，键名规则也不是任意PEFT格式或基座身份认证。

模型的load_state_dict先执行，结果检查在后，optimizer和extra/RNG的加载又在模型之后。因此拒绝缺键能阻止本次调用继续恢复后续状态，却不撤销已经写进模型的张量。这是恢复失败后的对象状态边界，不是另一个待补rollback缺陷；不能捕获异常后就假设对象仍为干净初值。

#### 真实 CPU 小模块的对照

从两版原load_checkpoint中取出完整的LoRA判定if/else分支，保留原分支体；原静态键判定和日志函数也原样执行。仅用属性容器提供model/rank/判定函数，model为真实torch.nn.Module，包含base、lora_A、lora_B三个1×1 Linear，无FakeFSDP/PEFT。每次初值base=99、A=B=7，输入adapter值A=0.5、B=0.3：

| 输入 | base版本 | PR head | 执行后的模型 |
|---|---|---|---|
| 完整adapter键，不含base | 接受 | 接受 | base仍99，A=0.5、B=0.3 |
| 缺lora_B | 接受 | missing adapter keys的ValueError | 两版均已把A写为0.5，B仍7 |
| 多出lora_C键 | unexpected keys的ValueError | 同左 | 两版均已写A=0.5、B=0.3 |
| 完整state，包含base=42 | 接受，走默认严格加载 | 同左 | base=42，A=0.5、B=0.3 |

两版各四组、合计八组CPU控制两次一致。第一行也说明：adapter键齐全不证明加载到正确基座；LoRA-only文件不提供这份base权重。lora_train_meta.json中的r/alpha/task_type并非基座内容hash，也不代替完整adapter键检查。

Python3.12.13/torch2.9.1，复用已有CPU环境，无新增依赖。没有运行完整manager、torch.load文件读取、FSDP/DTensor、优化器/RNG、跨rank通信或训练。作者[原17项测试](https://github.com/verl-project/verl/blob/19b89e7100c06d7731247545274b117887ad8d38/tests/checkpoint_engine/test_fsdp_lora_only_checkpoint_on_cpu.py)使用FakeFSDP/PEFT并修改missing_keys返回，本机只读源码并做完整模块compile，没有复跑该套件，不能把八组控制叫作原17项通过。另两版各两个完整变更Python文件AST/内存compile，共四份。

base/head manager SHA256分别为 `ff5aa26224081af3937155f19561629e8e1da69bf290ec2fc67953dd4704ff22` / `5c494485f1d9a91d7da1282a9c33a21e6b7cc8aedf39f8d0bbf9821bd3614714`，原head测试 `1966b702ed7658970a2a93cf8ace161c1c0e5ce724885549e0cc3935a444c40d`；完整patch `a7e68e89145ffb78128f42702caeb75c18c85210f2450231de1f0f5011780a2f`。实际分片/PEFT恢复条件仅补[原M5参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)，不自动应用候选、下载模型或启动恢复实验。

以下复查仅需Python3.12+/torch及含固定对象的verl Git仓库；不新建永久实验工具：

<!-- lora-key-control:start -->
```bash
python3.12 -B - /path/to/verl-git <<'PY'
import ast
import hashlib
import logging
import os
import subprocess
import sys
from types import SimpleNamespace

import torch

BASE = "a4b9354968705a441fb7da7367924b15b902e771"
HEAD = "19b89e7100c06d7731247545274b117887ad8d38"
ROOT = sys.argv[1]
PATH = "verl/utils/checkpoint/fsdp_checkpoint_manager.py"


def source(rev, path):
    data = subprocess.check_output(["git", "-C", ROOT, "show", rev + ":" + path],
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    print(rev[:8], path, hashlib.sha256(data).hexdigest())
    return ast.parse(data)


predicate_tree = source(BASE, "verl/utils/checkpoint/checkpoint_manager.py")
base_class = next(n for n in predicate_tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseCheckpointManager")
predicate = next(n for n in base_class.body if isinstance(n, ast.FunctionDef) and n.name == "is_lora_only_state_dict")
shell = ast.parse("class Predicate: pass")
shell.body[0].body = [predicate]
ns = {}
exec(compile(ast.fix_missing_locations(shell), "<original-predicate>", "exec"), ns)
log_tree = source(BASE, "verl/utils/logger/aggregate_logger.py")
log = next(n for n in log_tree.body if isinstance(n, ast.FunctionDef) and n.name == "log_with_rank")
ns["logging"] = logging
exec(compile(ast.Module(body=[log], type_ignores=[]), "<original-logger>", "exec"), ns)
logger = logging.getLogger("lora-key-control")
assert torch.get_default_device().type == "cpu"
print("Python", sys.version.split()[0], "torch", torch.__version__)

for rev in (BASE, HEAD):
    tree = source(rev, PATH)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FSDPCheckpointManager")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_checkpoint")
    branches = [n for n in ast.walk(method) if isinstance(n, ast.If) and isinstance(n.test, ast.Call)
                and isinstance(n.test.func, ast.Attribute) and n.test.func.attr == "is_lora_only_state_dict"]
    assert len(branches) == 1
    wrapper = ast.parse("def apply_state(self, model_state_dict, remote_model_path, logger): pass")
    wrapper.body[0].body = branches
    exec(compile(ast.fix_missing_locations(wrapper), "<original-load-branch>", "exec"), ns)
    for case in ("complete_adapter", "missing_adapter", "unexpected_adapter", "full_state"):
        model = torch.nn.Module()
        for name, value in (("base", 99.0), ("lora_A", 7.0), ("lora_B", 7.0)):
            layer = torch.nn.Linear(1, 1, bias=False, device="cpu")
            layer.weight.data.fill_(value)
            model.add_module(name, layer)
        state = {"lora_A.weight": torch.tensor([[0.5]]), "lora_B.weight": torch.tensor([[0.3]])}
        if case == "missing_adapter": state.pop("lora_B.weight")
        if case == "unexpected_adapter": state["lora_C.weight"] = torch.tensor([[0.9]])
        if case == "full_state": state["base.weight"] = torch.tensor([[42.0]])
        owner = SimpleNamespace(model=model, rank=0, is_lora_only_state_dict=ns["Predicate"].is_lora_only_state_dict)
        error = None
        try:
            ns["apply_state"](owner, state, "<in-memory-state>", logger)
        except ValueError as caught:
            error = str(caught)
        expected_error = case == "unexpected_adapter" or (case == "missing_adapter" and rev == HEAD)
        assert (error is not None) == expected_error
        if error: assert ("unexpected keys" if case == "unexpected_adapter" else "missing adapter keys") in error
        for name, expected in (("base", 42.0 if case == "full_state" else 99.0),
                               ("lora_A", 0.5), ("lora_B", 7.0 if case == "missing_adapter" else 0.3)):
            torch.testing.assert_close(getattr(model, name).weight, torch.tensor([[expected]]))
        print(rev[:8], case, "ValueError" if error else "accepted", "A=0.5; B=" + ("7" if case == "missing_adapter" else "0.3"))
print("8 original-branch CPU controls passed; no FSDP, checkpoint I/O, optimizer or training.")
PY
```
<!-- lora-key-control:end -->

## 3. MegatronCheckpointManager — Dist Checkpoint + Async Save

```
megatron_checkpoint_manager.py (115-1298):

★ ★ V2 Split Layout (3个sibling目录):

local_path/
  ckpt_contents.json → manifest(最后写入→signals completion)
  transformer_config.json → Megatron TransformerConfig
  model/
    huggingface/ → HF weights via mbridge(可选)
    dist_ckpt/ → Megatron sharded model weights
  optimizer/dist_ckpt/ → optimizer + lr_scheduler shards
  extra/dist_ckpt/ → rng_state shards

★ ★ Async Save:
  → Megatron.AsyncCallsQueue → async_save_request返回
  → finalize callback(manifest写入+HDFS上传+tracker更新)→attached to last request
  → 不阻塞训练! → 大模型checkpoint保存极慢 → async → 继续训练!

★ ★ Save (1169-1298):
  → FullyParallelSaveStrategyWrapper → 多rank并行写
  → _build_model_sharded_state_dict(metadata):
    → model.sharded_state_dict(metadata=model_metadata) → per VP rank
  → Megatron-FSDP: model.state_dict_for_save_checkpoint() → DTensor-aware
  → mbridge HF export: bridge.save_weights() / bridge.save_hf_weights()

★ Load (856-985):
  → dist_checkpointing.load() → FullyParallelLoadStrategyWrapper
  → Megatron-FSDP: load_fsdp_dtensor_checkpoint
  → Legacy v1 layout detection: _raise_for_old_layout() → reject!
```

## 4. CheckpointEngineManager — 实时权重同步

```
checkpoint_engine/base.py (345-515):

★ ★ 架构: Trainer → [NCCL/NIXL/RDMA] → Rollout → ServerAdapter

update_weights() (470-514) 核心流程:
  1. if backend=="naive"(colocated): trainer.update_weights() → 零拷贝generator!
  2. else:
     a. abort_replicas() → 取消所有in-flight rollout请求
     b. 创建临时RayWorkerGroup → 所有replica workers
     c. release_kv_cache_replicas() → 释放KV cache内存(weights留在原位)
     d. build_process_group() → prepare+build_topology+init_process_group
     e. trainer.update_weights() + rollout.update_weights() → NCCL/NIXL send/receive
     f. finalize() → 所有workers
     g. resume_kv_cache_replicas()
     h. resume_generation_replicas()

★ ★ 5+后端实现:
  → ColocatedCheckpointEngine("naive") → 同进程generator零拷贝
  → NCCLCheckpointEngine → NCCL collective send/receive+bucketing
  → NIXLCheckpointEngine → RDMA(SM90/NVLink必需)
  → KIMICheckpointEngine → Kimi专用
  → MooncakeCheckpointEngine → Mooncake
  → HCCLCheckpointEngine → Ascend HCCL

★ ★ Weight提供: engine.get_per_tensor_param() → yields (name, tensor)

FSDP engine实现(transformer_impl.py:794+):
  → LoRA: collect_lora_params() or collect_merged_lora_params() → merge/unmerge
  → FSDP2: model.named_parameters() → param.full_tensor().detach().cpu()
  → Bucketing: split_weight_chunks() / merge_weight_chunks() → 大tensor分bucket

★ ★ vs rLLM Tinker: rLLM用save_checkpoint→new SamplingClient → 磁盘中转!
  verl用CheckpointEngineManager → 实时通信 → 不写磁盘 → 更快!
  但: naive backend = 同进程generator → 等价于rLLM in-process → 极快!
```

## 5. Training Loop Checkpoint Integration

```
ray_trainer.py (974-1108):

★ ★ _save_checkpoint() (974-1041):
  1. actor_rollout_wg.save_checkpoint(actor_local_path, ...)
  2. critic_wg.save_checkpoint(critic_local_path, ...) → only if use_critic!
  3. torch.save(self.train_dataloader.state_dict(), data.pt)
  4. latest_checkpointed_iteration.txt → 同步路径直接open("w")/write；async只跳过trainer写入，需有实际backend finalize承接（见下节，不是所有backend都已支持）

★ ★ GRPO只保存:
  → actor checkpoint + dataloader state → 无critic
  → PPO额外保存critic；实际存储比例取决于模型、优化器及save_contents，不能固定算50%或3倍

★ ★ _load_checkpoint() (1043-1108):
  1. 读取latest_checkpointed_iteration.txt → step number
  2. global_steps = int(folder.split("global_step_")[-1])
  3. actor_rollout_wg.load_checkpoint(actor_path)
  4. critic_wg.load_checkpoint(critic_path) → only if use_critic
  5. torch.load(data.pt) → dataloader.load_state_dict()
  6. checkpoint_manager.update_weights(global_steps) → sync到rollout replicas

★ resume_mode 3种:
  → "disable" → 不恢复
  → "auto" → 找最新checkpoint
  → "resume_path" → 指定路径
```

### 异步保存能力与恢复指针

**保存出文件、异步请求完成、发布恢复入口和恢复出完整训练状态，是不同保证。** [#7952](https://github.com/verl-project/verl/issues/7952)报告非Megatron后端开启actor async_save后，trainer跳过恢复指针写入而后端没有承接；已有同报告者YeonwooSung的[#7955](https://github.com/verl-project/verl/pull/7955)，不另开修复。2026-10-24读取issue全文/唯一评论及两份候选正文，#7955与下述#4311均open/non-draft、未合并。

#### 当前保护针对配置能力

#7955固定head `b8c1aa6a076ba38e1710ce0e8e1a2adb7f554e9b`，API目标base `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`，实际祖先`23a341e361e18063b4ec53e1f7cc4eae68f102cf`（ahead1/behind6）。真正PR范围三文件84增1删，不把目标base当已包含的主线。[ActorConfig保护](https://github.com/verl-project/verl/blob/b8c1aa6a076ba38e1710ce0e8e1a2adb7f554e9b/verl/workers/config/actor.py)在post_init中读checkpoint.get(async_save)，只允许白名单megatron；它没有实现非Megatron异步保存，也不修改trainer发布协议。critic flag不在本项保护中，不能说所有保存配置均已校验。

固定主线`3c6d9b43b67c74ec343d77874cd8fa6e841523da`的[v0 trainer](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/ppo/ray_trainer.py)和[V1 trainer](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/trainer/ppo/v1/trainer_base.py)仍按actor async_save跳过自己写tracker；同步路径直接open写文本，不是临时文件+rename式原子发布。[Megatron manager](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/checkpoint/megatron_checkpoint_manager.py)确有async队列finalize和rank0 tracker写入，但这段actor侧逻辑的存在，不能单独认证critic、dataloader、远端副本及退出阶段都形成了同一个完整快照。本轮没有执行这些保存流程。

#### 旧DCP候选不能直接替代发布协议

[#4311](https://github.com/verl-project/verl/pull/4311)head `f0411c956ee5ae91070cc378115b434a357dcc9c`，API目标base同为3f9d4c0，官方compare祖先却是`f9d6198684f86112451d25c3227f185b1249973b`（ahead1/behind1148）。其八文件512增94删是旧分支改动，不是当前主线已完成的集成。不能按PR编号/正文“支持FSDP async”就从#7955白名单增加一个名字。

该head的[FSDP manager](https://github.com/verl-project/verl/blob/f0411c956ee5ae91070cc378115b434a357dcc9c/verl/utils/checkpoint/fsdp_checkpoint_manager.py)保存DCP future，开始下一次保存时可等待先前future；完成日志线程等待结果并记录成功/异常。这里未写latest_checkpointed_iteration，八文件改动也没有PPO trainer发布接线。**日志线程等待完成不是恢复指针发布者。** 这不是对整个旧PR的完整review或当前训练故障复现；只说明若将来整合到当前trainer，仍须明确谁在所有必要状态就绪后发布、如何传播失败以及退出如何drain，不能以DCP有async API代替这些契约。

#### 查找入口只检查指针和路径

主线[find_latest_ckpt_path](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/checkpoint/checkpoint_manager.py)读取tracker中的整数并检查拼出的路径是否存在，不扫描global_step目录选择最大值，也不验证模型/optimizer/extra/data.pt、manifest或checksum。M1用原两helper AST接真实os/torch，在真实临时目录做五组控制：path=None→None；目录在但指针缺失→None；指针3+空global_step_3→返回该路径；指针4+对应目录缺失→None；非整数指针→ValueError。最后一种不是静默“从头开始”，不能与缺指针情形混写。

另从#7955原测试类摘取`test_missing_tracker_makes_auto_resume_start_from_scratch`完整原方法，绑定原helper及真实unittest/临时目录，一项通过。它只证明发现入口，不是实际恢复模型或完整ActorConfig测试。Python3.12.13/torch2.14.1，无distributed初始化、trainer、Ray、FSDP/DCP或checkpoint数据保存/加载；四个相关原模块另仅AST/compile，作者14测试/5 subtests不是本机结果。版本与hash留[当日日志](../../diary/2026-10-24.md)，实际整合条件留[M5参考](../../experiments/handoff-m5.md#verl-训练-checkpoint-发布参考)，不新增实验队列、应用候选或对外发布。

### 对象存储下载与 checkpoint 提交

**下载入口能读对象存储，不等于训练状态已经完整写入并可恢复。** [S3 RFC #7641](https://github.com/verl-project/verl/issues/7641)提出原生FSDP2对象存储生命周期；2026-10-25读取全文及唯一评论，维护者[询问S3 FUSE挂载路线](https://github.com/verl-project/verl/issues/7641#issuecomment-5487704372)，尚不是对某个实现的采纳或验收。查重另有旧[#2334](https://github.com/verl-project/verl/pull/2334)（head23bd321f，17文件774增54删）与下载侧[#6850](https://github.com/verl-project/verl/pull/6850)，两PR均OPEN／非draft／未合并；本轮不另写后端或自动选型。

RFC的 `trainer.default_remote_dir` 与 `_COMMITTED` 是提案接口；#2334正文给出的是 `trainer.s3_checkpoint_dir`。不能把不同候选中的配置名拼成主线已可使用的统一API。

#### 当前路径与下载候选

固定main `3c6d9b43b67c74ec343d77874cd8fa6e841523da` 的[fs.py](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/fs.py)只将hdfs://归为non_local；默认use_shm=False时，`copy_to_local("s3://...")`原样返回URI，并没有发起S3下载。[FSDP manager](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/utils/checkpoint/fsdp_checkpoint_manager.py)的hdfs_path明确未使用；legacy PPO的_load_checkpoint遇default_hdfs_dir非空直接拒绝，SFT共享handler的发现入口也以本地路径为准。不能给hdfs参数换一个URI就宣布获得对象存储恢复；挂载路径则应另按实际文件系统语义判断。

#6850固定head `c87b60ceace43f33eee027c95cbab32d05f8bc74`，API目标base3f9d4c0d；官方compare实际祖先为 `2390a3f5cff9d80d4844f8b4616ef8c2616276de`，ahead2／behind344，三文件120增4删。head本身是merge，直接第一父提交已含功能，不能将空parent diff当作没有实现。候选保持is_non_local仅识别HDFS，另加is_fsspec_path，将下载交给fsspec.fs.get，复用MD5路径缓存／filelock／目录记录；没有上传器、分布式checkpoint marker或remote resume发现协议。

本机Python3.12.13、fsspec2026.7.0、filelock4.2.4、pytest9.1.1，执行完整原fs.py／hdfs_io.py模块，原测试仅替换框架import，选择该候选新增的五测试体，结果重复一致：

| 原五测试体 | 固定main／实际祖先 | 候选head |
| --- | --- | --- |
| URI分类 | 缺is_fsspec_path，AttributeError | 通过 |
| 单文件下载、glob拒绝、目录尾斜线 | 三项预期检查失败 | 三项通过 |
| 本地路径透传 | 通过 | 通过 |

使用真实fsspec MemoryFileSystem和临时本地缓存，不是S3服务；未跑同文件另外三项HDFS mock测试或完整原生pytest套件，也未安装s3fs／gcsfs、读取云凭据、访问桶／HDFS或执行checkpoint manager。URI分类通过不证明相应云后端已安装，memory下载通过不证明认证、重试、远端校验或持久性。主线另一个S3 URI透传控制只核函数返回值，没有云端请求。

#### 多对象完成条件需要应用协议

[AWS一致性说明](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html#ConsistencyModel)区分单key原子更新与跨key事务：成功PUT后的读取／列举可见，不代表多个rank的不同对象同时完成。因而“能LIST到global_step前缀”不是完整checkpoint条件；[条件写入](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html)可约束单对象覆盖，但不会自动建立整步事务。RFC提出先上传分片及common.pt，再发布actor/.metadata与最后的_COMMITTED，这是待实现／验收的协议，不是主线已有保证。

审阅时应追问marker引用的对象集合是否固定、如何绑定run／step／world_size及对象内容、是否容许随后覆写，以及多writer重试怎样避免同前缀混入另一快照。只有marker名或对象存在不能回答这些问题。multipart完成只组装**一个对象**，不是一次提交所有rank；[AWS文档](https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html)也说明最终ETag不必是内容MD5，校验不能无条件以ETag代替约定的checksum。这些是存储协议审阅条件，不是本机发现某个已实现S3后端的新故障。

RFC的初期范围为actor-only、相同FSDP world size；包含critic的训练仍须另说明critic与数据／scheduler／RNG的同一步恢复契约。作者1.37TiB／64rank案例中，save_state返回约5.1秒与durable save约39.4秒分别描述暂停和持久化延迟，不能把前者当全部远端完成。本轮没有重跑该案例，也没有核验#2334完整17文件或其mock上传测试，不将其正文中的auto-resume主张挪作RFC实现或本机结果。

FUSE可以提供另一种路径接口，但仍须指定实现／版本并验证写入、seek、rename／关闭／同步、并发可见和冷恢复语义；不因能挂载就视为POSIX等价，也不未经实测否决这条路线。后续选型应先取得维护者方向与用户明确的云资源授权，而不是拼接#2334／#6850／RFC或从本机更多缓存测试推导durability。[复查脚本](../../experiments/verl-object-store-read/verify.py)、[结果](../../experiments/verl-object-store-read/results.json)随仓库保存，命令／hash见[当日日志](../../diary/2026-10-25.md)；GPU／云端恢复条件只补[原M5参考](../../experiments/handoff-m5.md#verl-训练-checkpoint-发布参考)，不新建存储服务或实验队列。

### 保留目录与完整恢复的边界

`max_ckpt_to_keep` 控制已登记目录的淘汰，不是保存事务或磁盘容量上限。`BaseCheckpointManager` 将“保存前腾位置”和“成功后登记”拆开，避免上限为 1 时提前删掉唯一旧目录。[固定版本源码](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/utils/checkpoint/checkpoint_manager.py)

| 上限 | 保存前 `ensure_checkpoint_capacity` | 成功后 `register_checkpoint` |
|---|---|---|
| 1 | 不删除旧目录 | 加入新路径后删除超出的旧目录，只跟踪新目录 |
| 2 及以上 | 已登记数达到上限时，先裁到上限减 1 | 加入新路径，再裁掉超限部分 |
| 0 | 不删除 | 继续记录所有路径，不按数量删除 |

因此，上限为 1 的保存过程中可以同时存在旧目录与正在写入的新目录，需要相应磁盘余量。路径列表从当前 manager 的空列表开始，并非磁盘目录扫描；删除使用 `shutil.rmtree(..., ignore_errors=True)`，列表裁短也不认证磁盘删除已经成功。

`register_checkpoint` 的前提是调用方已完成保存；它不检查模型分片、优化器、manifest 或恢复结果。不能绕过前提，手动登记一个空路径，再据此判定标准保存流程丢失数据。实际接线须与保留 helper 分开看：

- [FSDP manager](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/utils/checkpoint/fsdp_checkpoint_manager.py)：rank 0 在保存前调用容量清理，在保存流程尾部登记；这些 helper 不是 trainer 的 actor、critic、dataloader 整轮联合提交。
- [Megatron manager](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/utils/checkpoint/megatron_checkpoint_manager.py)：同步路径先清理，async 路径跳过保存前清理。`_dispatch_finalize` 将回调挂到最后一个 async request，并将所有请求入队；`_finalize_save` 按 manifest、可选 HDFS copy、async rank 0 tracker、登记保留策略的顺序执行。这是源码中的完成通知安排，不是本机对 Megatron 队列、跨 rank 故障或存储持久化的验收。
- [Trainer](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/trainer/ppo/ray_trainer.py)：写完 `data.pt` 后，同步路径以 `open(..., "w")` 和 `write` 更新 tracker；actor async 时跳过该更新，由 worker finalize 写入。尽管注释有 `for atomic usage`，这两处写 tracker 都不是临时文件加原子替换；不能据此称整个 checkpoint 保存具备崩溃原子性，也未由此复现恢复故障。

**CPU 复查**：[原 5 项清理测试](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/tests/utils/ckpt/test_checkpoint_cleanup_on_cpu.py)全部通过，覆盖上限 1 的保存前保留、保存后删除、上限 2 预清理、0 不限量、连续三轮上限 1。测试创建的内容只有 `checkpoint.txt`，不含模型分片。原 fixture 自带空 model/optimizer 及 rank/world-size monkeypatch；没有初始化进程组。

复现使用[实验脚本](../../experiments/scripts/verify_verl_checkpoint_retention.py)，Python 3.12.13、torch 2.9.1、NumPy 2.0.2、pytest 8.4.2：

```bash
python experiments/scripts/verify_verl_checkpoint_retention.py /path/to/verl
```

脚本从固定 Git 对象选择完整原 `BaseCheckpointManager` 类、延后注解求值并提供真实标准库/torch/NumPy，完整加载原测试模块，由 pytest 执行 5 项原测试及 fixture。包根、框架导入、conftest/外部插件和缓存被隔离；收集限定到脚本目录。临时目录与删除审计限制在本次新建 scratch，退出时清理。未调用 RNG/设备方法或 FSDP/Megatron save/load，不属于完整包、真实 checkpoint 恢复或断电/进程崩溃测试。

源文件 SHA256：manager `be4fa702f0fab63dfb1d482716d31bc13a3b8f5129adbbd61872a169b07cbbc6`；原测试 `30f2df35a31c6eeb21db737557336c9178bb4b69a5427a44621121eaa4a18a18`。

### SFT epoch 边界与加载器状态

取到最后一批数据，不等于迭代器已经观察到`StopIteration`。[verl #7402](https://github.com/verl-project/verl/pull/7402)处理的正是这个差别：SFT从下一epoch开始训练，却恢复了尚停在上一epoch末尾的加载器状态，第一次遍历可能没有任何batch。已有作者ai-yang负责该修复，本项目只核验数据流，不另做竞争实现。

2026-10-23 API核对为open、未合并；固定head `b7d1f36e446da4caeb85a7f9d770daa416a9e76f`，目标base `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`，实际祖先 `09ac37258ea66b0cb69b2738eec3074ea4e7261c`（ahead4/behind149）。相对祖先5文件322增5删，不把目标分支的149个提交算进修复范围。

#### 最后一批与结束确认

本机Python3.12.13、torch2.9.1、torchdata0.11.0的真实CPU对照：8条数据、batch2、单副本显式rank的DistributedSampler、shuffle/seed0、两层drop_last、0worker且不pin。取完4批后立即保存，与再请求一次、确认结束后保存，恢复到新加载器并设置epoch1，结果不同：

| 保存时机 | 恢复后第一次遍历 |
|---|---|
| 最后一批已返回，尚未观察到StopIteration | 0批 |
| 已观察到StopIteration | 4批，顺序`[5,4] [2,6] [7,3] [1,0]` |

这两个快照在该版torchdata中的私有字段`_iterator_finished`分别是False/True。它解释本次现象，不应作为跨版本的公共API约定；不能笼统说“任何耗尽快照都会丢掉下一epoch”。

#### 边界修复需要旧长度

[原CheckpointHandler](https://github.com/verl-project/verl/blob/b7d1f36e446da4caeb85a7f9d770daa416a9e76f/verl/utils/checkpoint/checkpoint_handler.py)将保存时的`len(train_dataloader)`写入原`data_<dp_rank>.pt`字典的`_verl_steps_per_epoch`，加载时先取出，才把剩余状态交给torchdata。只有保存/当前长度相等且为正，恢复step也为正且整除该长度时，才跳过加载器状态、使用新迭代器。中途位置正常恢复；长度变化或缺元数据时警告，并按原生语义恢复，不猜测新边界。

[复查脚本](../../experiments/scripts/verify_verl_sft_loader_resume.py)提取祖先/head未改写的原加载方法及原日志helper，使用真实加载器、采样器与torch.save/load临时文件；只用属性容器提供self的loader/rank字段，不构造引擎。以下8组自编控制两次执行一致；前六组保存时均尚未观察到StopIteration，保存epoch长度为4。

| 控制 | step / 当前epoch批数 | 祖先→head返回批数 | 边界 |
|---|---|---|---|
| 新元数据、相同长度的边界 | 4 / 4 | 0→4 | 修复目标 |
| 新元数据、相同长度的中途 | 2 / 4 | 2→2 | 保留游标 |
| 新元数据、epoch缩短（batch4） | 2 / 2 | 1→1 | 不用新长度猜边界 |
| 新元数据、epoch变长（batch1） | 4 / 8 | 0→0 | 回退不保证补全下一轮 |
| legacy无元数据、中途 | 2 / 4 | 2→2 | 兼容旧位置 |
| legacy无元数据、边界 | 4 / 4 | 0→0 | 旧空迭代行为未修复 |
| 新元数据、已观察到结束 | 4 / 4 | 4→4 | 原生行为本已开始下一轮 |
| 新元数据、长度相同但数据ID全部加100 | 4 / 4 | 0→4 | 不检测数据身份变化 |

基线对照也读取带新增键的字典：该版torchdata容忍额外键；另有不带该键的legacy对照。这不是基线save格式已包含元数据。改变batch大小的结果只是该固定原生加载器的回退现象，不认证更换数据或拓扑后的精确恢复。相同batch数更不代表同一份数据。

#### 完成保护与验证范围

两份SFT `fit()`源码在`resume_global_step >= total_training_steps`时立即返回，早于tracker、sampler设置和batch遍历；但构造trainer及初次加载引擎checkpoint在fit之前，不能据此宣称完成任务不会加载模型。当前修复没有新增sidecar、配置校验异常或checkpoint发布顺序保证。

本机执行的是2个原生迭代状态控制及8组原方法数据流控制，并非原生verl导入、完整handler构造、Ray/SPMD调度或fit/模型/optimizer恢复。作者在同head报告的24项测试（12个checkpoint例、12个fit例，torch2.11.0+cpu/Ray2.56.1，含引擎/同步替身）与本机结果分开；本机未收集执行那24项。作者也明确没有重跑分布式/GPU、HDFS和异步finalize，旧实现的GPU报告不移作当前证明。本题是CPU加载器学习，不为未执行的完整训练另排M5任务，也不改上游实现。

复现：`python -B experiments/scripts/verify_verl_sft_loader_resume.py /path/to/verl-git --deps /path/to/torchdata-target`，需要上述固定Git对象、torch及torchdata0.11.0。本次仅将torchdata用`uv pip install --target ... --no-deps torchdata==0.11.0`装入独立work目录，未修改复用的torch环境。原handler SHA256：祖先`3b547c2b97e01b7587866053b4aa624d81b8cdd9d75fff20a97420690206a896`、head`676cf43c504922e610e88d236f0afcc5b4a449b63de1893616dee96a384960e8`；原日志helper模块`64d8556b0d6acc667a5be2dd43908111c9e58aae58d779ee9b02f0f28e2b42f7`。脚本打印hash并禁止隐式Git fetch。

### RL 数据子集与恢复身份

**加载器恢复了相同索引，不保证这些索引仍指向相同行。** [#7816](https://github.com/verl-project/verl/issues/7816)讨论重启时重新随机抽取 max_samples 子集；[#7824](https://github.com/verl-project/verl/issues/7824)讨论验证子集使用哪个 shuffle 标志。它们与上节 SFT 的迭代结束状态不同：即使恢复位置正确，重建数据的身份仍可能改变。2026-10-25核对已有两位作者的方案，不另开竞争实现。

#### 固定种子与验证集选择分开

[#7819](https://github.com/verl-project/verl/pull/7819)仍 OPEN、非draft、未合并，固定 head `0554e35fcf642a16b53877d330a7e7d21286752e`。API目标base为 `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`，实际共同祖先为 `a9f2985159536a607211dcac730d3f5d55028950`；相对祖先4文件182增6删。生产变化为未配置 data.seed 时用42抽子集，显式seed及shuffle=False的前N行分支不变。固定主线 `3c6d9b43b67c74ec343d77874cd8fa6e841523da` 的完整 rl_dataset.py 与该祖先字节相同，尚无这个fallback。

另一个 [#8141](https://github.com/verl-project/verl/pull/8141)仍 OPEN、非draft、未合并，head `9a7bd53b4c56d68f23ed0b647a899cf98e7880ef`、base及祖先 `75879f7f475fd6b64c779f7d9212e45503f58b8f`，2文件125增1删。它只在 `create_rl_dataset(..., is_train=False)` 时复制配置，将 validation_shuffle 映射到 dataset.shuffle；未配置该键则退回 data.shuffle。这个head仍使用旧的无seed抽样逻辑，因此“validation_shuffle=True且seed未设置”不因该PR自动稳定。

两个修复不能互相替代。固定种子解决固定输入下的子集重建；选择 shuffle 标志决定随机抽取还是按文件拼接顺序取前N行。后者改变评测数据组成，不能只因配置名更一致就称统计更具代表性；头部截取也不保证每个数据来源都有样本。

#### 原测试体与真实数据读取

复用 Python3.12.13／torch2.14.1、torchdata0.11.0+cpu、datasets5.1.0、PyArrow26.0.0、pandas3.0.6、NumPy2.5.4、OmegaConf2.3.1。完整原 RLHFDataset 类、get_dataset_class、工厂及采样器经 AST 隔离，fs／hdfs_io载入完整原模块；本地 Parquet／JSONL、HF Dataset 和 StatefulDataLoader 均真实运行。关闭长度过滤、tokenizer=None，原测试自带 row-ID view 避开tokenization；没有伪造采样器、文件加载或状态恢复。

直接调用两份原测试的函数体与其原fixture／helper，只移除 verl 导入声明并绑定上述原定义，不改断言；不是原生 pytest 收集或完整包导入。两次执行得到相同通过／失败集合：

| 原测试体 | 祖先 | 对应候选 |
| --- | --- | --- |
| #7819显式seed、前N行、未设seed重建、恢复后续行 | 2过、后两项AssertionError | 4过 |
| #7819默认常量固定为42 | 无该符号，不运行 | 通过 |
| #8141五项工厂测试 | 4过、validation_shuffle=False项AssertionError | 5过 |

#7819的64行取8行，fallback子集为 `[62,63,53,38,60,26,5,44]`；显式seed1234为 `[6,16,10,57,55,22,58,56]`。恢复例先消费两批共4行，state_dict之后的剩余行与重建／load_state_dict后的行一致。再保持文件路径和行数、将全部 row_id 加1000：加载器接受状态并读到原剩余行对应的加1000值。后者仅说明固定种子不是文件内容校验，不是该PR承诺支持数据变更却失败。[NumPy随机流兼容政策](https://numpy.org/doc/stable/reference/random/compatibility.html)不提供任意跨版本／构建／环境的Generator一致性保证；文件顺序／内容、过滤条件或tokenizer变化也须分别固定，不能以相同长度推出身份一致。

#### 工厂测试通过不代表所有入口接线完整

在 #8141 固定head上，原五项测试都显式传入 is_train；它们未覆盖遗漏该参数的调用者。将四个生产文件中**原验证集 create_rl_dataset 调用表达式**单独接到该head的原工厂和真实JSONL读取，配置 shuffle=True、validation_shuffle=False、seed1234、64行取8行：

| 原调用入口 | is_train 参数 | 返回 row_id |
| --- | --- | --- |
| [V1 trainer_base](https://github.com/verl-project/verl/blob/9a7bd53b4c56d68f23ed0b647a899cf98e7880ef/verl/trainer/ppo/v1/trainer_base.py#L718) | False | 0至7 |
| [旧同步 ray_trainer](https://github.com/verl-project/verl/blob/9a7bd53b4c56d68f23ed0b647a899cf98e7880ef/verl/trainer/ppo/ray_trainer.py#L387) | 省略，默认True | 6,16,10,57,55,22,58,56 |
| [one_step_off_policy](https://github.com/verl-project/verl/blob/9a7bd53b4c56d68f23ed0b647a899cf98e7880ef/verl/experimental/one_step_off_policy/main_ppo.py#L84) | 省略，默认True | 同上 |
| [fully_async_rollouter](https://github.com/verl-project/verl/blob/9a7bd53b4c56d68f23ed0b647a899cf98e7880ef/verl/experimental/fully_async_policy/fully_async_rollouter.py#L393) | 省略，默认True | 同上 |

这里 self／config 仅提供原表达式需要的字段，没有构造或执行trainer、Ray或rollout。因此证据是：**该head的三处调用表达式未进入新增验证分支**，不是三条真实训练失败记录。#7819正文仍提及调用点已显式传参，但其当前四文件diff没有这类修改；不能依赖这段描述假定叠加两PR就会接好。是否补齐入口和是否采用头部选择语义，应由现有作者／维护者与用户审视，不在本项目另写第三个修复。

供用户审阅的反馈要点：#8141的直接工厂测试通过，但三个旧入口漏传 is_train=False；建议现有作者核对这些调用者，并增加调用层覆盖。2026-10-25读取PR行审评论为0、普通评论仅CLA机器人；本项目未发评论、改PR或推其head分支。

七份固定源码快照另通过完整 AST／内存 compile；输入、环境、hash与一次性复现命令见[当日日志](../../diary/2026-10-25.md)。未运行完整trainer checkpoint／模型恢复、prompt过滤／tokenization、原生pytest、并行worker或设备，不需要为这项CPU学习另排M5任务，不把已有作者成果计为我方修复。

## 6. FSDP2 / DTensor Checkpoint

```
fsdp_utils.py (422-471):

★ ★ Version Detection:
  fsdp_version(model):
    isinstance(model, FSDP) → 1 (FSDP1)
    isinstance(model, FSDPModule) → 2 (FSDP2)
    else → 0

★ ★ FSDP1: ShardedStateDictConfig → FlatParameter shards
★ ★ FSDP2: nullcontext() → DTensor local shards → state_dict()直接返回!

get_fsdp_full_state_dict() (438-471):
  → FSDP1: FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
  → FSDP2: get_model_state_dict(model, full_state_dict=True, ...)

★ ★ FSDP2 Sharded Save/Load to CPU (1044-1142):
  fsdp2_sharded_save_to_cpu():
    → iterate all parameters → _local_tensor.detach().cpu() + DTensorSpec
  fsdp2_sharded_load_from_cpu():
    → verify device_mesh consistency → copy _local_tensor back from CPU
```

## 7. FSDPModelMerger — 离线合并→HF格式

```
model_merger/fsdp_model_merger.py (35-):

★ ★ 合并FSDP shards → 标准HF格式:
  1. 读取fsdp_config.json → world_size
  2. 加载rank-0 state dict → 提取DTensor/device_mesh信息
  3. ThreadPoolExecutor并行加载所有rank shards
  4. 合并DTensor shards: Shard(N) → torch.cat(dim=N)
  5. save_pretrained() → HF format(safetensors+config.json+tokenizer)

★ CLI:
  python -m verl.model_merger merge --backend fsdp --local_dir <path> --target_dir <path>
  python -m verl.model_merger merge --backend megatron ...

★ LoRA: lora_train_meta.json → 重建PEFT config → 分离adapter和base weights
```

### 复制分片的存储隔离

离线合并中的`Replicate()`表示取一个副本的值，不自动表示返回值拥有独立存储。[verl #7503](https://github.com/verl-project/verl/pull/7503)将该分支的`return tensors[0]`改为`.clone()`；2026-10-23读取open，作者amanyagami，固定head `e2d9f145d9176f8a0763bb3cd43ed430a4ca6d8b`、目标base `3f9d4c0d32cc20c7ec31c8137e91bb5673ce9e5e`、实际祖先 `8df88be746801b6d87c42e15f9a5a0ec1d5eeeae`（ahead2/behind89），两文件104增1删。当前版本只改变复制分支，已撤去此前广泛拒绝共享storage的guard；不能把旧版guard或目标分支的其他变更算进本候选。

[原函数](https://github.com/verl-project/verl/blob/e2d9f145d9176f8a0763bb3cd43ed430a4ca6d8b/verl/model_merger/fsdp_model_merger.py)处理的是已从DTensor取出的普通Tensor。`Shard(dim)`仍cat，`Partial()`仍不支持；复制分支仍只选首副本，clone不比较各rank值是否一致。它隔离的是存储所有权，不验证原checkpoint、模型权重绑定或HF文件键完整性。HF保存前另有显式`drop_tied_target_keys`路径，不能仅凭相同storage地址就把两个key判为错误重复。公开评审提出的plain-tensor分支是另一条路径，未给出实际共享storage的checkpoint复现；不把它自动扩张为已证实残余缺陷或另开修复。

#### 共享存储不等于重叠元素

一个缓冲的前四个元素和随后四个元素是两个不重叠view：`data_ptr()`不同，`untyped_storage().data_ptr()`相同。旧分支让输出继续引用源缓冲；更改首view会改源对应元素，但不会改第二view。新分支clone后，两输出与源缓冲各自独立。实际loader的`.bfloat16()`也有条件：输入已是BF16时可原样返回，FP32转BF16会新分配，不能假定所有输入都以相同方式保留别名。

[CPU复查](../../experiments/scripts/verify_verl_merger_storage.py)使用torch2.9.1的真实Tensor和Replicate/Shard/Partial，从固定对象提取整个原merger类，以object替代HF依赖基类，仅调用不读self的合并方法；作者三个测试函数体保持不变。基线2项通过、独立storage断言预期失败；head3项通过。另四组控制覆盖真实torch.save/load的BF16共享view往返、dtype转换、存储字节与源持有者、首副本/Partial规则，两次执行一致。没有原生verl导入、pytest收集、DTensor构造/进程组、完整merger CLI或HF保存；作者19项全套报告不是本机结果。

#### 输出占用与合并峰值

1024元素的BF16源缓冲共2048字节，两段各4元素的输出合计只有16字节有效值。对唯一storage去重计数得到：

| 仍可达的Tensor集合 | 基线共享view | head独立clone |
|---|---|---|
| 只计两个输出 | 2048字节 | 16字节 |
| 源view也仍被持有 | 2048字节 | 2064字节 |

这是实际storage.nbytes的CPU对照，不是RSS、allocator回收或峰值测量。clone可以避免**输出自身**长期保留整个源缓冲，却需要复制时的新分配；完整CLI还要结合源对象何时释放、并行加载、dtype转换和保存阶段判断峰值，不能由第一行直接断言整个合并峰值更低。

[原报告#6259](https://github.com/verl-project/verl/issues/6259)是64卡训练后离线转换出现跨safetensors文件的重复key；作者及公开评审已收窄根因声明。当前CPU控制只证明存储隔离，不复现原文件症状，也不把不同key共享storage等同于文件key重复。已有作者修复覆盖，不新增竞争PR；完整转换首先需要匹配checkpoint及Transformers/HF依赖，离线核验本身不要求重跑64卡训练，不因此新增M5算力任务。

复现：`python -B experiments/scripts/verify_verl_merger_storage.py /path/to/verl-git`，需Python3.12+/torch及本地固定Git对象，无新增依赖。脚本另断言目标base的合并方法AST与祖先相同，完整模块并不相同。原merger SHA256：base `2ae5f3f7fc95871b08a2e185def4f1a5cec09164579cd15eeb2314cb73ef1f0f`、head `29be8e51f7e6bc36a062ff0dbb7a9feb27186a1c3b6cba432c6d8661e0f4d1e1`；原测试 `890253f9060f545948bb9e87cdb8e7bf6d4d8665965bc29e0348db2ac0703c1f`。

## 8. 关键设计洞察

```
1. 2层架构 → 磁盘持久化+实时同步 → 分离关注点!
   → CheckpointManager → 磁盘 → 恢复训练 → 离线合并 → 下游使用
   → CheckpointEngineManager → 实时 → trainer→rollout → 不写磁盘
   → 分离 = 不同需求不同路径 → 磁盘慢但持久 → 实时快但临时 → 设计正确!

2. Per-rank sharded → 不gather完整模型 → 省内存!
   → 每rank保存自己的shard → torch.save → 不需要AllGather
   → Load: ShardedStateDictConfig(offload_to_cpu=True) → CPU加载 → GPU OOM安全!
   → vs consolidated: rank0 gather → 大模型=70GB+ → OOM风险 → sharded更安全!

3. HF export → rank 0 only → 生产部署桥梁!
   → get_fsdp_full_state_dict → rank0_only=True → 只1个rank gather → 省通信
   → init_empty_weights → 空骨架 → 不占GPU内存 → 填入state_dict → save_pretrained
   → safetensors格式 → vLLM/SGLang直接加载 → 生产部署无缝!

4. Megatron async save → 不阻塞训练 → 大模型关键!
   → AsyncCallsQueue → async_save_request → finalize callback → 继续训练
   → ckpt_contents.json最后写入 → completion signal → 检测是否完整
   → vs FSDP: torch.save阻塞 → 大模型=分钟级 → 训练暂停 → 性能损失!

5. CheckpointEngineManager → vs rLLM Tinker → 两种weight sync范式!
   → rLLM: save_checkpoint → disk → new SamplingClient → 磁盘中转 → 简单但慢
   → verl naive: generator零拷贝 → 同进程 → 等价rLLM in-process → 极快
   → verl NCCL: collective send/receive → 跨GPU → 不写磁盘 → 比磁盘快!
   → verl NIXL: RDMA → 跨节点 → 极快 → 需SM90/NVLink → RTX 4090不能用!

6. GRPO checkpoint → 省critic → 50%存储节省!
   → PPO: actor + critic + dataloader → 3×
   → GRPO: actor + dataloader → 2× → 无critic → 省1整套模型存储!
   → 这与GRPO省50%内存/compute一致 → 全系统优化!

7. FSDPModelMerger → 离线工具 → 训练→推理桥梁!
   → 训练产出FSDP shards → 不直接可用于推理 → 需合并
   → merger: ThreadPoolExecutor + torch.cat → 离线合并 → HF format → vLLM直接用
   → 这是训练→推理的完整pipeline → 训练checkpoint → merge → vLLM serve → 闭环!

8. LoRA-only checkpoint需要区分adapter和基座身份
   → 固定a4b93549：save_lora_only且模型有LoRA时，按adapter键规则过滤保存
   → r/alpha/task_type元数据不是base权重hash或恢复完整性证明
   → #8350拒绝缺adapter键，但检查发生在模型写入后，不保证回滚
   → 实际大小取决于目标层/结构/dtype，不沿用旧段未测的固定GB数字
```

---

Sources:
- verl/utils/checkpoint/checkpoint_manager.py (BaseCheckpointManager)
- verl/utils/checkpoint/fsdp_checkpoint_manager.py (FSDPCheckpointManager)
- verl/utils/checkpoint/megatron_checkpoint_manager.py (MegatronCheckpointManager)
- verl/utils/checkpoint/checkpoint_handler.py (CheckpointHandler for SFT)
- verl/checkpoint_engine/base.py (CheckpointEngineManager)
- verl/utils/fsdp_utils.py (FSDP1/FSDP2 state dict helpers)
- verl/trainer/ppo/ray_trainer.py (training loop checkpoint)
- verl/workers/engine/fsdp/transformer_impl.py (FSDP engine save/load)
- verl/model_merger/fsdp_model_merger.py (FSDP shard consolidation)
- Background agent research (verl checkpoint system)
