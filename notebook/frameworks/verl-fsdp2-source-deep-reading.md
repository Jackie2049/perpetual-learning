# verl FSDP2 Engine Deep Source Reading — Beyond PR Descriptions

2026-10-30补充：[packed边界与尾部padding](#packed-边界的显式传递与尾部-padding)核#8464：15个原CPU参数项在隔离原函数上通过，静态pad形成独立段；配置门控不是forward兼容性检查，空行offset与位置推导表示有别。非attention／梯度或GPU提速验收。

## Packed 边界的显式传递与尾部 padding

**已有offset可以免去重复推导，但必须描述模型实际收到的完整packed布局。** [#8464](https://github.com/verl-project/verl/pull/8464)读取OPEN，固定head `fc66f413f71c0c85a37847aa9e2b50e9bbea225b`、base `8e48d72de4766228c36623c90f9f5c49ee5b45f3`，五文件含两个新增测试。这里是FSDP训练端将多条序列平铺后的attention边界，不是[shared-prefix packing](verl-shared-prefix-packing.md)的KV前缀复用，也不是rollout调度优化。

[原helper](https://github.com/verl-project/verl/blob/fc66f413f71c0c85a37847aa9e2b50e9bbea225b/verl/workers/engine/utils.py)把offset转int32，最大真实段长通过一次`.max().item()`转为Python int，再按需追加静态pad的终点；Q／K使用同一offset张量与最大段长。原pad_packed_inputs给尾部pad的位置编号从0重启，因此它是独立尾段，不应扩入最后一条真实序列。例如真实长度[5,3]、静态pad8，边界为[0,5,8,16]、max_length为8，而非只取真实最大长度5；pad4时边界[0,5,8,12]、max仍5。无padding时不额外加段，单offset的空微批返回空dict。

### 原 CPU 断言与两个限定对照

本机Python3.12.13／Torch2.9.1／pytest8.4.2，单CPU线程；未安装Transformers。对[15项原CPU测试](https://github.com/verl-project/verl/blob/fc66f413f71c0c85a37847aa9e2b50e9bbea225b/tests/workers/test_fsdp_flash_attn_varlen_kwargs_on_cpu.py)保留参数装饰、helper与断言，通过AST隔离原函数后直接调用，而非原生pytest收集或导入FSDPEngine。对照是[Transformers5.16.1固定提交93c8b7b4的原helper](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/modeling_flash_attention_utils.py)；同样只选完整定义、不替换数学或全包导入。六种布局（[5,3]配pad0／4、[1,7,1,2]配0／9、[16]配0／3）均与原position_ids推导得到的offset／max一致，另九项核空微批与配置门控。

两个本地控制单独记账，不算原测试：一是给FA2文本config夹具加不接收额外关键字的strict forward，原配置helper仍返回True，inspect.Signature.bind则拒绝cu_seq_lens_q；没有构造真实remote-code模型或执行forward。二是长度[5,0,3]的offset helper保留[0,5,5,8]，position_ids中空行没有元素，原HF helper推导为[0,5,8]。这是空段的表示差别，不独自证明attention输出不同；本轮未确立生产端会送空行，也不将它定性训练缺陷或扩非法offset矩阵。

[脚本](../../experiments/verl-fa-varlen-metadata/verify.py)与[结果](../../experiments/verl-fa-varlen-metadata/result.json)保存四个完整verl输入hash和固定HF输入hash；两个最终新进程完整JSON一致。HF完整文件SHA256必须为`dea5b555b5bb8fba1b4ab2c62c8df9f96019b17555b3714f75be17c54e801f95`，脚本锁定它、禁止Git隐式fetch，不自动联网／安装。复现需预缓存上述Git对象，并先将固定链接的HF完整源码保存为外部输入：

```bash
python3.12 -B experiments/verl-fa-varlen-metadata/verify.py /path/to/verl-git /path/to/modeling_flash_attention_utils.py
```

### 启用条件与完整模型的边界

[engine接线](https://github.com/verl-project/verl/blob/fc66f413f71c0c85a37847aa9e2b50e9bbea225b/verl/workers/engine/fsdp/transformer_impl.py)要求remove-padding分支有packed offsets、FA2／FA3文本config、无vision_config属性、模型不已接收cu_seqlens，并在实际准备输入时排除Ulysses SP。该SP路径仍从all-gather后的position_ids推导边界，不能将本候选结果移作SP支持；vision_config即使值为None，只要属性存在也被排除。[dense_common](https://github.com/verl-project/verl/blob/fc66f413f71c0c85a37847aa9e2b50e9bbea225b/verl/models/transformers/dense_common.py)还让torch／triton fused forward把额外kwargs继续传入base model；只修改helper不能覆盖原先吞掉kwargs的fused路径。本机这三份完整模块只compile／读取接线，没有执行模型或fused算子。

配置门控没有检查四新参数或**kwargs的有效forward签名，作者已在正文披露，另有[机器人行内反馈](https://github.com/verl-project/verl/pull/8464#discussion_r4422782832)；不是维护者批准或本机发现的真实模型故障。包装器接收kwargs也不证明底层模型接收，不能据本地Signature示例实现一个通用signature补丁。

PR作者在RTX5090／Torch2.8／Transformers5.16.1报告默认行对精确父版训练耗时约少9.4%，并有GPU logits／梯度及60步SFT比较；它们不是本机复跑或普遍收益。本轮仅元数据与配置控制，没有FSDP实例、TensorDict／nested输入接线、attention、forward／backward、模型质量、同步次数或速度测量。硬件缺口记[M5条件参考](../../experiments/handoff-m5.md#verl-packed-attention-边界参考)，只供将来已有匹配工作时独立选择，不部署、升级或新增必跑队列。

2026-10-23 补充：[参数转换与缓冲区精度](#参数转换与缓冲区精度)固定于 #8189，用原 CPU 测试与原转换语句区分初始化数值和存储 dtype；本节之后的 2026-06 分析未在此次重核。

2026-10-26 补充：[NPU卸载的复制策略与固定内存](#npu-卸载的复制策略与固定内存)区分FSDP1／优化器复制开关与FSDP2显式pin策略，CPU调用测试不认证NPU内存收益。

## NPU 卸载的复制策略与固定内存

**阻塞D2H复制不等于关闭所有CPU offload的pinned memory。** 新[#8372](https://github.com/verl-project/verl/pull/8372)、[#8373](https://github.com/verl-project/verl/pull/8373)、[#8374](https://github.com/verl-project/verl/pull/8374)在2026-10-26读取时均OPEN/non-draft、同一CodeRabbit app创建，均自述未运行验证；评论仅bot跳过审查／CLA未签提示，reviews与行内空，不作人工验收。完整正文、diff、原测试及查重均已核，不另开第四份。

| 候选 | 固定head | 相对共同父版的生产改动 |
| --- | --- | --- |
| #8372 | `648423c5a5cece57a5412f8c1d96bee7d068d512` | FSDP1模型分片及优化器tensor卸载，NPU用non_blocking=False |
| #8373 | `340680d39c84293c2bebdaf292a3941b77958b47` | 仅FSDP1模型分片，优化器保留原True |
| #8374 | `8b7aa26c055e8f4cbbfc8989e429e4265b5dcb67` | 与#8372完整fsdp_utils模块AST相同，测试不同 |

三者父版／目标base均为`e009360c03e708e75d5628c9b7ad584af8a53e4d`。相对当轮main `3c6d9b43b67c74ec343d77874cd8fa6e841523da`均ahead4／behind0、实际祖先main；中间三个LoRA sleep/resync提交未在本轮审核或应用，不能把整支候选当单一卸载补丁。父版与main的完整fsdp_utils相同。三个PR各两文件，依次90增2删、66增1删、108增2删。

### 两个不同的分配入口

固定[Ascend/pytorch `0ed76164180e419b6c33569d05ae48b2627803ca` 的ToKernelNpu.cpp](https://github.com/Ascend/pytorch/blob/0ed76164180e419b6c33569d05ae48b2627803ca/torch_npu/csrc/aten/common/ToKernelNpu.cpp)：`_to_copy`在non_blocking为真、源为NPU、目标为CPU且layout为strided时令pin_out为真，再把它传入目标empty／empty_strided的pinned_memory选项。改变该复制flag会影响**这次目标分配**，不只是主机等待方式。[同版CopyKernel.cpp](https://github.com/Ascend/pytorch/blob/0ed76164180e419b6c33569d05ae48b2627803ca/torch_npu/csrc/aten/common/CopyKernel.cpp)另区分异步任务／事件处理与同步stream后memcpy。两原文件blob分别`a7a2a212e4d5e19379090ffe46da66d145d8e59d`／`2f69e6bd1383e89142b582da3bf9ddb268aa16a1`。这是固定后端源码，不代表M5安装版、所有布局或实际allocator行为已验收。

新候选的模型修改只在FSDP1分支，原`_offload_params=True`的handle仍跳过，原data／local_shard存储别名更新保留；两份另改通用optimizer tensor，非tensor元数据不动。FSDP2／普通模块提前分派到另一个helper，该helper在main已用`model.to("cpu", non_blocking=False)`；三候选没有改它，也没有改模型／optimizer重载函数。

另一入口是[main FSDPEngine构建](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/verl/workers/engine/fsdp/transformer_impl.py#L454)：offload_policy或forward_only时显式构造`CPUOffloadPolicy(pin_memory=True)`，并非上述D2H默认目标分配。三个新候选未触及它。已有[#8057](https://github.com/verl-project/verl/pull/8057) head `cbe21f44a4a537fd7d7ebe0da92dc88c1db47d91`另加默认True的offload_pin_memory，FSDP-Turbo nightly另传其已有pin_memory=False；完整diff还含CI、runner及依赖pin，不能按标题拼接。其A3故障／cgroup／RSS观测为作者现场报告，不作本项目或所有平台结论；本轮只读正文／diff，未执行其配置或nightly。

### CPU 测试的验证范围

[复查脚本](../../experiments/verl-npu-offload-policy/verify.py)隔离加载完整原四函数及真实FSDP类型，选#8374原三个测试函数、原参数共7项：head全过，父版4过、3项NPU复制flag断言预期失败。测试体不改，FlatParameter和CPU clone真实，但handle、device module及optimizer tensor仍是原Mock；没有构造FSDP实例、进程组或进行NPU／CUDA传输。设备发现导入由必须被原测试替换的fail-closed guard保护，未导入verl或torch_npu。基线失败限定原call-list断言位置／non_blocking调用断言，不将任意错误当复现。

AST比较忽略注释与行号：#8372与#8374完整模块同，#8373模型函数同但optimizer未改；五版FSDP2／重载函数不变。未另外运行#8372／#8373测试。[JSON](../../experiments/verl-npu-offload-policy/result.json)保存原参数、source hash及Python3.12.13／torch2.14.1／pytest9.1.1环境，重复结果一致。这不是完整pytest收集、pin分配峰值／耗时、并发rank或NPU内存耗尽修复验收。

```bash
python -B experiments/verl-npu-offload-policy/verify.py /path/to/verl
```

真机缺口写入[M5交接入口](../../experiments/handoff-m5.md#verl-npu-参数卸载参考)，仅在安全、获准、匹配的NPU工作中接续；M1不探测服务器、不以4090替代NPU、不故意耗尽allocator或另启训练。后续根据真实故障栈确定分配入口，不扩Mock矩阵或另造第四份修复。

## 参数转换与缓冲区精度

[#8154](https://github.com/verl-project/verl/issues/8154) 指出，显式 BF16 加载后再调用 `module.to(torch_dtype)`，会连模型以 FP32 构造的 RoPE 频率 buffer 一起舍入。报告者提供的三模型、九条固定 token 轨迹、四条件 CUDA replay 属外部证据，不能当本项目训练结果。已有 [#8189](https://github.com/verl-project/verl/pull/8189) 修复，读取时 open/non-draft：head `4e830c72bd0eb2b0d6ecfbd4c2c4144040637324`、目标 base 与共同祖先 `1d367a4f459a43768f76312c0d3430a99ec46b8c`，ahead 2 / behind 0，3 文件 118 增 3 删；不另开竞争候选。

新 helper `cast_module_params_to_dtype` 遍历 `module.parameters()`，只对 dtype 不同的浮点或复数参数赋值 `param.data.to(dtype)`，不遍历 buffer。这样保留的是**调用前的 buffer 数值和 dtype**，不是一律将 buffer 升成 FP32，更不是重建 canonical RoPE。原测试中的 `inv_freq` 从 FP32 经 BF16 再升回 FP32，最大频率绝对误差为 `9.765475988388062e-05`；之后再调用 helper 也仍保留舍入值。正确的先后关系是避免破坏初始值，不能靠后来标签变回 FP32 来认证精度恢复。

在固定 `_build_module` 中，转换位于 HF/value 模型加载、Liger/forward patch 之后，gradient-checkpointing 开启之前。显式 `model_dtype` 与后续 mixed-precision policy 是不同边界：`None` 时训练加载 dtype 选择 FP32，forward-only 选择 BF16；FSDP1 把 `buffer_dtype` 交给 `MixedPrecision`，而此处 FSDP2 的 `MixedPrecisionPolicy` 只传 param/reduce dtype 与输入 cast，不能把 FSDP1 的 buffer 参数原封不动当作 FSDP2 行为保证。这些是调用点源码核对，不是 FSDP 运行验证。

参数身份、存储内容与训练状态也应分开：原测试确认两个模块绑定同一个 `Parameter` 时身份仍保持，meta 参数可转换，整数参数/buffer 不变。helper 本身不看 `requires_grad` 或 HF `_keep_in_fp32_modules` 名单，因此冻结 FP32 参数仍转换；这不是实现 FP32 参数例外的完整策略。对已经带 FP32 `.grad` 的参数调用它，参数可变 BF16 而 `.grad` 保持原对象/FP32。该控制说明它不是训练中途的梯度/optimizer 状态迁移器，不把初始化路径里没有出现的状态当作新缺陷，也不自动拼接相关 keep-list PR #7165。

[复查脚本](../../experiments/scripts/verify_verl_param_cast.py)在 Python 3.12.13、torch 2.9.1、显式 CPU 默认设备下运行两次，原 6 个测试体均通过；测试覆盖 buffer 精度、浮点参数、meta 参数、整数、绑定参数身份及旧 blanket cast 反例。另 4 组控制核对原 base/head 转换调用、已有低精度 buffer、冻结/keep-list 属性以及已有梯度。完整 head `torch_dtypes.py` 与原测试模块执行，仅跳过根包初始化；两版 engine 只做语法及提取出的原转换语句执行，没有运行 `_build_module`、HF RoPE、FSDP 包装、真实模型或 logp。旧版没有新 helper，基线验证限于原 cast 语句对照。

```bash
/path/to/python3.12 -B experiments/scripts/verify_verl_param_cast.py /path/to/verl
```

真实 FSDP/model 的数值边界保留在[原 M5 策略一致性参考](../../experiments/handoff-m5.md#verl-异步策略一致性补证)：仅在日后独立选题且资源已匹配时核同权重、同 token/position 的 buffer 与 logp；不因 CPU 通过直接部署补丁、拼接 PR 或开启训练。

> 2026-06-18 | Source-level analysis | verl/fsdp_utils.py + engine_workers.py + rollout_corr_helper.py + bucketed_weight_transfer.py
> ★★★★★★★★ Going beyond PR-level analysis to actual implementation understanding
> ★★★★★★★★ 6 key implementation findings with RTX 4090 implications
> ★★★★★★★★ Reveals fragile patterns, hard-coded paths, and optimization opportunities

---

## 1. layered_summon_lora_params: Hard-Coded Prefix Fragility

```
★★★★★★★★★ verl/utils/fsdp_utils.py:631-670 — LoRA weight extraction for FSDP:

def layered_summon_lora_params(fsdp_module) -> OrderedDict:
    prefix_list = [
        # fsdp (FSDP v1)
        "_fsdp_wrapped_module.base_model.model.",
        "_fsdp_wrapped_module.base_model.model.model.",
        "_fsdp_wrapped_module.base_model.model.model.layers.",
        "_fsdp_wrapped_module.base_model.model.model.language_model.layers.",
        # fsdp2 (FSDP v2)
        "base_model.model.",
        "base_model.model.model.",
        "base_model.model.model.layers.",
        "base_model.model.model.language_model.layers.",
    ]

★★★★★★★★★ FINDING: 8 hard-coded prefix strings!
  → 4 for FSDP v1 (with _fsdp_wrapped_module wrapper)
  → 4 for FSDP v2 (direct module access)
  → Covers 2 model architectures: plain layers + language_model.layers

★★★★★★★★★ PROBLEMS:
  1. Fragile: any model with different module nesting → NOT found → LoRA params silently MISSING
  2. Maintenance burden: every new model architecture → need to add prefix
  3. No dynamic detection: could use model.named_modules() to auto-discover LoRA layers
  4. Already covers Qwen + Llama → but NOT Phi, Gemma, Mistral-NeMo, etc.

★★★★★★★★★ RTX 4090 impact:
  → If using non-standard model architecture → LoRA sync SILENTLY FAILS
  → Must verify model module nesting matches prefix_list before training
  → Safe for Qwen/Llama (most common RTX 4090 models) → others need prefix check

★★★★★★★★★ Alternative approach (better):
  → Use PEFT's get_peft_model_state_dict with auto module discovery
  → Or: introspect model.named_modules() for LoraLayer instances dynamically
  → This would eliminate hard-coded prefixes entirely
```

---

## 2. _is_root = False Hack Outside Summon Context

```
★★★★★★★★★ verl/utils/fsdp_utils.py:668 — after summon_full_params block:

                with FSDP.summon_full_params(submodule, writeback=False):
                    sub_lora_params = get_peft_model_state_dict(...)
                    ...
                get_torch_device().empty_cache()
            # ← HERE: submodule._is_root = False is set INSIDE summon
            #   but the pattern is fragile: _is_root should only be set
            #   inside summon_full_params context manager

★★★★★★★★★ FINDING: FSDP v1 _is_root manipulation pattern
  → FSDP v1 uses _is_root flag to track whether module is in summoned state
  → Setting _is_root = False is part of the summon_full_params cleanup
  → The code works because summon_full_params context manager handles this
  → BUT: if someone uses the pattern outside summon → FSDP state corruption

★★★★★★★★★ RTX 4090 impact:
  → LOW risk for FSDP v1 path → _is_root handled correctly inside context
  → FSDP v2 uses different mechanism (FSDPModule) → no _is_root
  → But: fragile pattern → potential bug if refactored without understanding
```

---

## 3. No Degenerate-Unit Filtering in LoRA Summon

```
★★★★★★★★★ verl/utils/fsdp_utils.py:654-667 — summons ALL submodules matching prefix:

    for prefix in prefix_list:
        for name, submodule in __prefix_submodules(fsdp_module, prefix):
            ...
            if fsdp_version(submodule) > 0:
                with FSDP.summon_full_params(submodule, writeback=False):
                    sub_lora_params = get_peft_model_state_dict(...)
                    lora_params.update(sub_lora_params)
                    submodule._is_root = False
                get_torch_device().empty_cache()

★★★★★★★★★ FINDING: No degenerate-unit filtering!
  → Iterates ALL 32+ decoder layers even if most have NO LoRA params
  → Each summon → materializes full parameters → GPU memory spike
  → Most layers: LoRA A+B = 0 params → summon is pure overhead
  → Empty_cache() after each → but spike still occurs momentarily

★★★★★★★★★ Memory spike calculation (7B model, LoRA rank=32):
  → Per decoder layer full params: ~350 MiB (bf16)
  → 32 layers × 350 MiB = ~11.2 GiB peak spike
  → With #6512 per-unit summon: only summon layers WITH LoRA → ~4 layers → ~1.4 GiB
  → ★★★★★★★★ 8x memory reduction potential from filtering!

★★★★★★★★★ verl #6512 (per-unit LoRA summon) addresses this:
  → But layered_summon_lora_params still summons ALL layers
  → #6512 uses different path: FSDP2 per-unit summon → only materializes needed units
  → ★★★★★★★★ layered_summon path is pre-#6512 → still has spike problem on FSDP v1

★★★★★★★★★ RTX 4090 impact:
  → 11.2 GiB spike on 24 GiB GPU → 46% of total memory → dangerous!
  → With CPU offload: spike still occurs before empty_cache → risk of OOM
  → MUST: use per-unit LoRA summon (#6512) OR filter to only LoRA-containing layers
```

---

## 4. Bypass 概率锚点与 loss 配置

将 `rollout_log_probs` 写入 `old_log_probs` 能跳过独立old概率重算，但不自动改变已经创建的actor worker的loss配置。2026-10-23核对已有[#8348](https://github.com/verl-project/verl/pull/8348)后，完整配置时机、原19项CPU用例和固定版本见[主笔记](verl-async-logprob-provenance.md#bypass-字段替换与-worker-的-loss-配置)。候选在worker创建前选择默认bypass loss，显式其他loss则保留并拒绝不生效的校正选项；不是所有bypass配置都运行同一loss。

两策略/三策略的区分描述概率锚点数量，不是正确性的充分条件。staleness drop/wait限制版本距离，不能保证参数、分布或logprob数值接近；还需核对实际配置、概率口径和同步内容。撤回旧段没有测量依据的 `18Ψ → 3.8Ψ` 显存比例，以及“低staleness确保bypass有效”的推断。当前CPU配置测试不提供显存、训练收益或真实worker验收。

---

## 5. HYBRID Weight Sync: 6-Step Flow with LoRA

```
★★★★★★★★★ verl/workers/engine_workers.py:667-746 — update_weights() in HYBRID mode:

Step 0: Determine sync mode
  → effective_mode = "auto" → config.rollout.checkpoint_engine.backend
  → If NOT "naive" → async path → checkpoint_engine.send_weights() → return

Step 1: Resume rollout weights (released during sleep)
  → await self.rollout.resume(tags=["weights"])
  → Rollout was sleeping → now wake up to receive weights

Step 2: Probe for LoRA config
  → get_per_tensor_param(layered_summon=True, base_sync_done=True)
  → Returns per_tensor_param + peft_config
  → If peft_merge=False AND peft_config exists → LoRA path

Step 3: Base weight sync (LoRA adapter path, FIRST TIME only)
  → do_lora_base_sync = not self.base_sync_done
  → First sync: get_per_tensor_param(base_sync_done=False) → base weights only
  → await self.rollout.update_weights(base weights, base_sync_done=False)
  → ★★★★★★★★ base_sync_done flag prevents redundant base sync → saves 50% time!

Step 4: Adapter/merged weight sync
  → await self.rollout.update_weights(all params, base_sync_done=True)
  → Includes LoRA params (or merged weights if peft_merge=True)

Step 5: Offload + cleanup
  → if is_param_offload_enabled: engine.to("cpu", model=True, optimizer=False)
  → aggressive_empty_cache(force_sync=True)

Step 6: Resume KV cache
  → await self.rollout.resume(tags=["kv_cache"])
  → self.base_sync_done = True → future syncs skip Step 3

★★★★★★★★★ RTX 4090 implications:
  → Steps 2-4 involve GPU memory spikes (summon + transfer)
  → Per-unit summon (#6512) reduces Step 3 spike from 11 GiB to ~1.4 GiB
  → CPU offload in Step 5 frees GPU for rollout KV cache in Step 6
  → ★★★★★★★★ This is why HYBRID mode works: sleep/wake lifecycle manages memory!
```

---

## 6. BucketedWeightSender: ZMQ IPC Transfer Mechanism

```
★★★★★★★★★ verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py:

class BucketedWeightSender:
    → ZMQ REQ/REP socket pattern for weight transfer
    → Default bucket_size = 512 MB (configurable via bucket_size_mb)
    → Supports CUDA IPC handles AND shared memory fallback (NPU compat)

★★★★★★★★★ Weight transfer flow:
  1. Pack tensors into fixed-size bucket (512 MB)
  2. When bucket full: synchronize GPU → send metadata via ZMQ
  3. Receiver reconstructs tensors from CUDA IPC handles or shared memory
  4. Each tensor gets metadata: {name, shape, dtype, offset, handle}

★★★★★★★★★ MoE gate precision handling:
  → Line 122-126: "we should not force cast weight here because some parameters
    (such as moe gate) have to keep fp32 precision"
  → ★★★★★★★★ MoE gate stays fp32 → correctly preserved → NOT cast to bf16!
  → Rollout side auto-casts on demand → but increases transfer volume

★★★★★★★★★ CUDA IPC handle mechanism:
  → rebuild_ipc(): changes device_id in handle for different CUDA_VISIBLE_DEVICES
  → This is key for HYBRID mode where trainer+rollout may see different GPU IDs
  → ★★★★★★★★ trainer GPU 0 → rollout GPU 0 → IPC handle device_id corrected

★★★★★★★★★ RTX 4090 implications:
  → 512 MB bucket size → reasonable for 24 GiB GPU → ~2% memory per bucket
  → MoE gate fp32 → slightly more transfer → but correctness matters more
  → CUDA IPC → zero-copy transfer → fast weight sync → essential for GRPO
```

---

## 7. FSDP2 LoRA Merge/Unmerge: Layer-by-Layer Pattern

```
★★★★★★★★★ verl/utils/fsdp_utils.py:848-871 — fsdp_merge_unmerge():

FSDP v1: Whole-model summon → merge → reshard
  → with FSDP.summon_full_params(module, writeback=True):
  →     _merge_or_unmerge_lora_(module, merge=do_merge)
  → ★★★★★★★★ DANGER: whole-model summon → ALL params materialized → OOM risk!

FSDP v2: Layer-by-layer summon → merge → reshard
  → for name, submodule in module.named_modules():
  →     if isinstance(submodule, FSDPModule) and name != "":
  →         with FSDP.summon_full_params(submodule, writeback=True):
  →             _merge_or_unmerge_lora_(submodule, merge=do_merge)
  → ★★★★★★★★ SAFE: per-layer summon → only one layer materialized at a time

★★★★★★★★★ RTX 4090 implications:
  → FSDP v1: whole-model summon → 27B model → ~54 GiB (fp32) → OOM on 24 GiB GPU!
  → FSDP v2: per-layer summon → ~350 MiB per layer → fits easily in 24 GiB
  → ★★★★★★★★ MUST use FSDP v2 for LoRA merge on RTX 4090 → FSDP v1 OOM!
```

---

## 8. Cross-Framework FSDP Summon/Release Pattern

```
★★★★★★★★★ Universal pattern found in 4 implementations:

Pattern: summon → extract → release → empty_cache
  → verl FSDP2: FSDP.summon_full_params(submodule) → extract → release
  → verl FSDP v1: FSDP.summon_full_params(module) → extract → release
  → PyTorch FSDP2: same summon_full_params pattern
  → DeepSpeed ZeRO-2: partition gathering → extract → partition release

★★★★★★★★★ Memory lifecycle:
  → Before summon: GPU holds only shard (or nothing with offload)
  → During summon: GPU holds full parameters → PEAK MEMORY
  → After release: GPU returns to shard state → memory freed
  → empty_cache(): explicit GPU cache cleanup after each summon cycle

★★★★★★★★★ Three optimization levels (from cross-framework synthesis):
  1. Whole-model summon (FSDP v1) → ALL params → OOM on RTX 4090
  2. Per-unit summon (FSDP v2, verl #6512) → ONE unit → fits RTX 4090
  3. Fractional offload (#187620) → partial CPU → dp>=2 ONLY → NOT RTX 4090!

★★★★★★★★★ RTX 4090 MUST use Level 2: per-unit summon → verified safe
```

---

## Key Findings Summary

★★★★★★★★★ 8 hard-coded LoRA prefix strings → fragile → silently fails on non-standard architectures
★★★★★★★★★ _is_root = False hack inside FSDP v1 summon → fragile pattern → works but dangerous if refactored
★★★★★★★★★ No degenerate-unit filtering → summons ALL 32 layers even if only 4 have LoRA → 8x memory waste
★★★★★★★★★ Bypass mode = 1-line substitution (old_log_probs = rollout_log_probs) → BUT infrastructure matters
★★★★★★★★★ HYBRID weight sync = 6 steps → base_sync_done flag prevents redundant base sync → 50% time savings
★★★★★★★★★ BucketedWeightSender ZMQ IPC → 512MB bucket → MoE gate fp32 preserved → CUDA IPC zero-copy
★★★★★★★★★ FSDP2 LoRA merge = layer-by-layer → SAFE for RTX 4090 → FSDP v1 = whole-model → OOM!
★★★★★★★★★ Universal summon/release pattern → 3 levels → RTX 4090 MUST use per-unit (Level 2)

---

## References

- verl fsdp_utils.py: layered_summon_lora_params (line 631), fsdp_merge_unmerge (line 848), backup_base_model_weights (line 961)
- verl engine_workers.py: update_weights HYBRID flow (line 667)
- verl rollout_corr_helper.py: bypass_mode substitution (line 1131)
- verl bucketed_weight_transfer.py: ZMQ IPC mechanism (line 74)
- verl #6512: per-unit LoRA summon → peak memory bounded by largest FSDP unit
- verl #6699: detach model_output → 4x memory reduction
- verl #6717: Tinker training worker primitives
- verl #6765: OptimStepParams per-step overrides (MERGED June 18)
- Cross-framework synthesis: notebook/fundamentals/cross-framework-fsdp-evolution-synthesis.md
- verl FSDP2 vs DeepSpeed ZeRO-2: tools/fsdp2_vs_zero2_decision_guide.py
