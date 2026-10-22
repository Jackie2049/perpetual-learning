# verl：训练与 rollout 的量化层审计

2026-10-21，M1 只读学习已有作者的 [PR #8132](https://github.com/verl-project/verl/pull/8132)。核心收获：**配置一致、导出格式一致、引擎实际选层一致是不同证据；选层一致仍不等于权重数值或 logprob 一致。** 本文不是新缺陷报告或竞争实现。

## 版本与问题

读取时 PR open，作者 wengeezhang；固定 head `0defd9a231cb876ce56812611b0afb3b2b25831a`，base `cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3`。这是量化支持堆叠的第 7 部分，依赖前序 PR，并按 #8287 的导出接口重做；整个 PR diff 包含前序提交，不能全算作本次审计新增代码。

作者旧版 B200/ModelOpt 观察发现：训练与 rollout 的忽略规则看似相同，dense `gate_up_proj` 却可能受 `mlp.gate` 前缀匹配影响而走不同精度。作者据此将新版改为观察真实导出权重流，而非重建引擎忽略规则。**这些旧版真机结果不是当前 head 的验收**；作者明确新版尚未复跑硬件。本机也没有做真机验证。

完整阅读固定 head 的 [审计模块](https://github.com/verl-project/verl/blob/0defd9a231cb876ce56812611b0afb3b2b25831a/verl/utils/quant_layer_audit.py)、[原始 CPU 测试](https://github.com/verl-project/verl/blob/0defd9a231cb876ce56812611b0afb3b2b25831a/tests/utils/test_quant_layer_audit_on_cpu.py)、[QAT 导出模块](https://github.com/verl-project/verl/blob/0defd9a231cb876ce56812611b0afb3b2b25831a/verl/utils/qat/compressed_tensors.py)；worker、Megatron、vLLM 只读相应接线片段，不代表审过整个训练/推理栈。

## 三类证据各自证明什么

| 来源 | 实际观察 | 不能直接推出 |
|---|---|---|
| 训练模块 | Transformer layer 子模块存在非空 TE `_fp8_workspaces`，或 offload 前保留的 `_verl_fp8_workspace_seen` 标记 | 每个权重、当前 step 都实际运行同一种 FP8 kernel |
| 导出权重流 | 原生 FP8 dtype，或 `.weight_packed` 规范化为量化权重；普通 weight 记未量化 | packed 数值、scale、布局、量化 recipe 或内容相等 |
| live vLLM 模块 | 解析实际 `quant_method`/scheme，处理名称映射、packed QKV/gate-up、MoE holder、LoRA base layer | 所有 worker 一致，或该方法已执行正确 kernel |

训练信号是 **layer 粒度**：靠类名与 `layer_number` 识别层，将全局 1-based 层号转为 0-based；不存在于本地 PP stage 的层返回未知。标记可跨 workspace 清理保留，不是逐次 GEMM trace。层外参数还受 `outside_layers_quantized` 判断影响。

导出记录只看格式，不比较 tensor 内容；引擎侧把方法描述中不以 `unquantized` 开头的项当量化。两阶段比较的是“量化/未量化”布尔值，不能认证相同位宽、recipe 或误差。

`kept_high_precision_by_design` 复用 QAT 模块的真实 `DEFAULT_IGNORE` 与匹配函数，覆盖 lm_head、embedding、visual、mtp、router/shared-expert gate 等。训练/导出比较遇到这些权重会**先跳过**；`by_design` 计数不是“已经观测它们全为 BF16”。非本地 PP 层的未知项也不判一致或不一致。

## 完成、沉默与失败的边界

审计只为量化 rollout + TE FP8 recipe 条件创建；默认 `warn`，配置也可关闭。不能从“没有告警”反推审计已创建、运行且匹配。

1. `record` 包装导出 generator，逐个记录并原样 yield；**完全消费流之后**才进行训练/导出比较。提前停止消费不等于完成比较。
2. 首次没有 FP8 workspace 信号时等待。两次不同同步仍无信号，或禁用 parameter transpose cache 时，结束并解释 **no verdict**；未知状态不是一致性通过，`raise` 模式也不把这种说明变成 mismatch 异常。
3. 有信号后完成一次判定，设置 `done`/verdict，再告警或抛错；后续同步不重复审计。它不是每步持续验证，也不保证后续切换精度仍被发现。
4. `raise` 位于流被消费之后，不能代表“权重从未应用”或自动回滚；异常完成点与接收端事务是两件事。

引擎复查需要已有 verdict，且只复查一次。无法解析的名称/PPMissingLayer 返回未知并跳过；应保留 checked、unresolved 的覆盖范围。空/None 返回也会消耗这次复查机会，不可写作“全量匹配”。合并 worker 返回时取首个有效非 None 方法，不做跨 worker 冲突检测或多数共识。

这些是观察边界，不在缺少运行证据时把它们定性成新 bug。

## 接线路径并非同一覆盖范围

固定 head 的 Megatron 导出在 quantizer 之后记录，adapter-only 路径不做同样审计；workspace 清理前留下标记。`engine_workers.update_weights` 的 colocated/naive 分支在 rollout 更新返回后、恢复 cache 前尝试引擎复查；非 naive checkpoint 路径发送后提前返回，不经过该复查。获取引擎方法的 RPC 异常只警告退出，不能视为匹配证明。

SGLang MXFP8 loader 使用独立 engine guard，逐请求写入前检查格式；接续源码分析见下节。未穷尽 delta/replica 等全部路线，不能将 colocated vLLM 的覆盖外推为所有 rollout 后端。

## SGLang 分桶加载与失败清理

2026-10-21 接续阅读同一 head 的完整 [MXFP8 loader](https://github.com/verl-project/verl/blob/0defd9a231cb876ce56812611b0afb3b2b25831a/verl/workers/rollout/sglang_rollout/mxfp8_refit_loader.py)、[数值探针 helper](https://github.com/verl-project/verl/blob/0defd9a231cb876ce56812611b0afb3b2b25831a/verl/utils/mxfp8_refit_check.py)，以及 [rollout 调用片段](https://github.com/verl-project/verl/blob/0defd9a231cb876ce56812611b0afb3b2b25831a/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L364)。PR/API 再核仍 open，head/base 未变。以下是源码结论，没有运行 loader 或其单测。

同一权重同步会分为多个请求。普通 bucket 只加载，最后的 `FINALIZE_MARKER` 请求才统一派生 kernel 布局；逐 bucket 派生会碰到后续尚未写入的专家 scale，也可能重复处理已变换的数据。调用方先依次 await bucket，再 await finalize，正常返回后才 flush cache、更新 global step。这个先后关系不证明网络中断后的远端状态或全 rank 原子提交。

| 阶段 | 本地动作 | 证据边界 |
|---|---|---|
| 请求预检 | 对本请求可解析的 weight 比较输入/引擎 dtype 类别，再调用 `load_weights` | 拒绝的是本请求，不撤销前面成功写入的 bucket |
| 首 bucket 暂存 | 所需 MoE canonical scale buffer 全部分配成功后，才重绑参数；随后加载 | 分配失败不遗留部分重绑，不代表加载过程事务化 |
| finalize | 取 MoE 参考快照，处理 dense/MoE 布局，回写 live storage，再做抽样自检 | 处理/拷贝依次发生，中途异常可能已修改前面的层 |
| try 内异常 | `_abandon_sync` 移除同步标记，令暂存 scale 参数重新指向原 live storage，然后重抛 | 恢复指针，不恢复全部旧权重/scale 数值，也不是安全继续生成许可 |

格式 guard 只比较 `{BF16, FP16, FP32}` 与“其他 dtype”，并非验证精确 FP8 recipe。scale 等非 weight 叶子不走这项比较；未知名称跳过，解析 dtype（包括未知结果）按名字缓存在 model 上。因而首条一致日志或返回的 checked 数量不是全同步、全参数覆盖证明。

### Canonical 数据与图捕获存储

MoE scale 只有在 live shape/dtype 不再是 canonical `[... , K/32]` uint8 时才暂存；未改变 shape/dtype 的 runner 可能直接覆写 live buffer。暂存 buffer 填 `0xFF`，检查开启时，finalize 在处理对应 MoE 层前检查残留哨兵；它发现未写条目，但不认证写入值来自正确版本，也不覆盖所有未暂存 scale。

重新派生后，结果必须恢复为原 live shape/dtype；必要时 `live.copy_(new)`，再将参数指回 live storage，并检查所有暂存项均已回绑。这里分开维护两个条件：**内容已更新**与**CUDA graph 捕获的存储仍可读到内容**。CPU `data_ptr` 断言不能替代真实 graph replay。源码已有 dense DeepGEMM→FlashInfer fallback 可能重绑 swizzled scale 的 TODO，不能只凭 MoE 的回绑逻辑宣称所有 backend 安全。

### 自检通过与跳过必须分开

自检选最小 dense 层、最小 MoE 层的 local expert 0，默认 seed 0 的 8 行输入。比较的是 kernel 输出与 canonical 权重/scale 的反量化参考，不是训练 actor 与 rollout 的参数相等；默认相对误差阈值 0.25 也不是通用健康界或完整模型认证。

MoE EP>1、TP>1 且无 reduce_results、无法用 logits 固定单专家的路由或非 canonical scale 会跳过。helper 对非有限 reference 和超阈值/非有限输出报错，但对探针内部的一般 Exception 警告并返回 NaN；loader 不将这类返回重新判失败。因此 finalize 正常返回仍可能伴随 skipped probe。`VERL_MXFP8_REFIT_CHECK=0` 同时绕过格式 guard、暂存 scale 哨兵检查和数值自检，不仅关闭日志。

对应[原始测试](https://github.com/verl-project/verl/blob/0defd9a231cb876ce56812611b0afb3b2b25831a/tests/utils/test_sglang_mxfp8_refit_loader_on_cpu.py#L421)的分桶、失败清理、分配失败和预检案例，只读核对；测试中的 SGLang backend、TopK、quant method 与 swizzle 本身为替身（如 clone+1）。本轮不新增替身、不声称复跑作者 loader 测试。现有 CPU 环境缺 SGLang、hydra、omegaconf、tensordict、compressed_tensors，未补装或启动服务。

源码 SHA256：loader `073b917564abd783e724aed3759912ded9a74eb65444ce6a633989d2e4f35b30`；loader 测试 `65945e46ef916fe4e059d03a7df2dd7089d6b7ee934bf050fdcf60b6d6132247`。与下方 24 项 audit 单测是不同执行范围。

## 新旧 SGLang refit 路线的覆盖关系

2026-10-21 接续核对新 [#8306](https://github.com/verl-project/verl/pull/8306)，作者 wuxibin89，open；本轮固定 head `1beb35554a763d18ed76fcbefaac1b224eaa1eeb`，不是上一轮列表看到的 `457aa56d…`。旧 #8132 仍 open/head `0defd9a2…`。**两条是有重叠的并行方案，不能按编号较新就视为替代，更不能把测试和配置直接拼接。** #8306 作者也明确标注与 #8130 的 MXFP8 export/refit 重叠，以及与 #7971 NVFP4 方向的概念重叠；没有据此新增我方实现。

API 的目标 base 为 `cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3`，但与新 head 的 merge-base 是 `5a8dd519ab32bfd36d2509a137f830ba79e89c54`。以 `base...head` 核对8个PR改动文件，与API列表一致；没有新增/修改测试。直接 `base head` 两点比较还会显示目标分支独有的 optimizer-offload 测试差异，不能误说是本PR撤销了该修复。

### 扩展格式不等于继承全部审计

| 范围 | 旧笔记固定 #8132 路径 | 新 #8306 固定 head |
|---|---|---|
| 专用 refit 格式 | 已读的 whole-model MXFP8 loader | MXFP8，以及 BF16 dense + MXFP4/NVFP4 routed experts；拒绝 MXFP8 dense + MXFP4 MoE |
| 关闭同步 | bucket加载后专门marker触发派生 | 同样要求完整bucket序列；显式拒绝custom refit的delta_flush |
| 存储处理 | MoE scale回绑；dense fallback留有TODO | 加入dense swizzled/deepgemm scale捕获与copy-back，以及FP4参数/派生量回绑 |
| 保护与证据 | 选层审计、dtype guard、未写scale哨兵、抽样数值自检；本机24项audit单测 | 已读三个新模块未接入这些审计/guard/数值probe；作者报告Blackwell观测，本PR diff没有提交配套CPU测试 |

这是不同分支的覆盖对照，不是说新PR从旧实现“删除”了保护。旧24项测试和作者旧loader测试也不能算作新路线回归通过。

完整阅读固定 head 的 [sglang_quant_utils.py](https://github.com/verl-project/verl/blob/1beb35554a763d18ed76fcbefaac1b224eaa1eeb/verl/utils/sglang/sglang_quant_utils.py)、[FP8 helper](https://github.com/verl-project/verl/blob/1beb35554a763d18ed76fcbefaac1b224eaa1eeb/verl/utils/sglang/sglang_fp8_utils.py)、[FP4 helper](https://github.com/verl-project/verl/blob/1beb35554a763d18ed76fcbefaac1b224eaa1eeb/verl/utils/sglang/sglang_fp4_utils.py)，以及export/worker接线差异和文档相关片段；没有审完整SGLang或全部训练路径。

### 配置和消息边界必须成套核对

新配置入口优先解析rollout override，再看模型native配置；refit kind由配置结构判定，不只凭“MXFP8”名字。all-MXFP8 compressed-tensors配置转native SGLang schema，已知regex ignore映射为引擎名称，未知regex拒绝；部分ignore还经 `SGLANG_FP8_IGNORED_LAYERS` 传递。导出复用CT量化器，将 `.weight_scale` 名字改为 `.weight_scale_inv`，该包装器没有把tensor数值取倒数。

新loader FQN是 `verl.utils.sglang.sglang_quant_utils.load_and_reprocess`，marker也与旧 `mxfp8_refit_loader` 不同。非NVFP4配置的接线要求DeepGEMM，NVFP4留给SGLang选择；FP4 helper另检查支持的runner条件。启动配置、exporter、每bucket的load_format、finalize marker及实际backend必须属于同一版本路线，不能混用旧请求或根据另一分支的默认值推定兼容。

新 `_finish_sync` 按 FP8 dense、FP8 MoE、MXFP4、NVFP4 顺序重建，完成后移除同步标记。FP4 staging可直接复用live tensor或同一存储的dtype view，并不总是额外副本；`restore_*_live` 恢复参数/派生量引用与alias，未备份全体旧内容。因此失败清理仍不等于参数回滚或跨rank事务。dense捕获/copy-back分支解决的是源码中的存储身份处理，不能仅凭该分支宣称真实graph replay已经验证。

### 本轮只确认源码与语法

作者报告Qwen3.5-35B-A3B、Megatron EP8/SGLang TP2、Blackwell默认CUDA Graph条件下的三类量化mismatch观测；这是作者证据，不是本项目复跑，也不是全部配置的通用阈值或长期训练证明。作者说明CPU套件受torchcodec/libc10_cuda.so导入问题阻挡，最终diff未带配套测试。本机的torchcodec本身未安装，没有复现作者同一故障。

M1只在Python3.12.13对三个固定Git blob做AST解析和内存compile（272/211/260行），均通过，不导入或执行模块、不生成pyc；现有环境也缺SGLang、compressed_tensors、hydra/omegaconf。没有新造fake workspace或用旧单测认证新loader。

SHA256：quant helper `9557c1c4ae3b65c41d36b0a462e14c66e737691cf99e3aad1c4f6431050c08a6`；FP8 helper `cd167e604bf66231673d7571664592a6c21ad3f5c4edb9aab80f2a93f3679b57`；FP4 helper `266f9c4fa4a2501fab01d04fe67d87b91c0a766d9b71832c27ece05bb5db28b0`。真机接续补入[同一M5条目](../../experiments/handoff-m5.md#verl-量化层审计新版复验)，不要求M5同时验证两套或自动merge/cherry-pick；先按实际资源与已选路线决定。M5已有4090记录不等于具备本PR所述Blackwell验证条件。

## 本机复现：24 项原始 CPU 单测

[复现入口](../../experiments/scripts/verify_verl_quant_layer_audit.py) 从指定只读 Git 仓库取旧 #8132 固定 head `0defd9a2…`，不覆盖 #8306，不检出、不安装、不下载模型：

```bash
python experiments/scripts/verify_verl_quant_layer_audit.py /path/to/verl-git
```

2026-10-21，Python 3.12.13、torch 2.9.1、pytest 8.4.2：**24 passed**。完整原样执行审计、QAT 模块与测试体/fixture，只用 namespace shell 隔离 `verl`/`utils`/`qat` 的包初始化，并用 collector 指向原始测试模块；禁用外部 pytest 插件、conftest 和缓存。复用已有 Git 加载 helper，打印 SHA256：

- `quant_layer_audit.py`：`d4c02f4b317e37b7f0d12c84c87f3d8bfb44faa39fb6287cdc6f982dc823a27d`
- `test_quant_layer_audit_on_cpu.py`：`fbd62da8919235d9b90993dc21f66fabd77b4f539e02cde67342ffdb2062dfce`
- `qat/compressed_tensors.py`：`bd466f3abca89b642cc5af5637236b1920466c432be9e0d45b9fd320f4bc6c83`

真实 CPU torch tensor/Module/dtype 参与，但原测试自己构造 Megatron/TE workspace、vLLM quant method 等替身。它验证记录、跳过、告警/抛错、等待、映射等**单元逻辑**，不是 TE forward/offload、实际 compressor、引擎/RPC、SGLang loader 或真实训练。外部 compressed_tensors、Transformer Engine、vLLM/SGLang 未导入；现有环境也缺这些包。本轮未运行作者声称的 63/137 项组合，不补装依赖来伪装完整栈验收。

## 如何用到后续诊断

量化层审计补充的是“选层/格式”的可观测性；[策略来源笔记](verl-async-logprob-provenance.md)讨论的权重身份、raw/processed 概率与配对 token 仍需独立证据。这里的 Megatron/TE 量化路径不能直接解释 #8250 私有 FSDP2 异步训练退化，也没有推出新修复。

两条固定路线的真机接续统一登记 [M5 真机交接](../../experiments/handoff-m5.md#verl-量化层审计新版复验)，先选版本路线，再记录分桶/finalize、实际 backend、graph，以及该路线实际具备的 guard/probe 覆盖。状态是待匹配环境，不代表已接单。M1 的本项源码学习收束，不探测服务器、不竞争已有 PR、不改 AReaL，不能用更多替身替代缺失的真机证据。

## ModelOpt 模块忽略与子量化器

**权重导出并正确加载，不保证 actor forward 已按要求跳过 fake quantization。** 2026-10-22 只读已有 [#8329](https://github.com/verl-project/verl/pull/8329)，作者LZC-BELIEVER，读取时open。固定head `5c7e2d2ed49bd9f448043883d96b6e191e187e95`、目标base `16ec093cc4a235c603c29aed1f2c4e9f551fab48`、merge-base `f2923cbf154731918b1e36db8c567433f349fede`；真正PR diff为2文件66增5删：生产helper新增3行，另一项是测试扩充及cpu→gpu文件改名。不是#8132的审计实现或#8306的refit路线更新，不能自动叠加或借用其验证。

### 模块名和量化器名属于不同层次

[`_ignore_patterns_to_quant_cfg`](https://github.com/verl-project/verl/blob/5c7e2d2ed49bd9f448043883d96b6e191e187e95/verl/utils/qat/modelopt.py#L281)旧版只追加 `{"quantizer_name": key, "enable": False}`。给模块模式 `*self_attention.linear_proj`，实际叶子名却是 `...linear_proj.input_quantizer` 和 `...linear_proj.weight_quantizer`，后缀使旧模式不能匹配。候选保留原规则，再追加 `key + ".*"`；点号使精确结尾in_proj_a不会顺带选择in_proj_ab，而不是粗略追加星号。

[ModelOpt 0.44.0 官方消费代码](https://github.com/NVIDIA/Model-Optimizer/blob/c897fbeaaff66d53d61033f107885b7c5432f235/modelopt/torch/quantization/conversion.py)按列表顺序应用规则，后匹配项可覆盖前项；只对TensorQuantizer/SequentialQuantizer匹配名称，并使用fnmatch，另有专家名称归一化。只有enable=False而没有cfg的规则切换启用状态，不重写其他数值属性。verl将ignore_cfg放在preset/default规则之后，因而它能覆盖前面的enable；不是修改一份未被消费的配置字符串。

本机两版原helper配标准库fnmatch的名称选择对照如下，输入为合成名称，不构造量化器：

| 用户模式 | base选择 | head选择 | 范围 |
|---|---|---|---|
| `projection` / `projectio?` | 无 | projection的input、weight | 模块模式扩到后代 |
| `projection.*` | 两个叶子 | 两个叶子 | 已有通配用法不变 |
| `projection.weight_quantizer` | 仅weight | 仅weight | 保留直接叶子规则，input不被关闭 |
| `*linear_attn.in_proj_a` | 无 | 仅in_proj_a的两个叶子 | in_proj_ab不被选中 |
| `*self_attention.linear_proj` | 无 | 对应两个叶子 | 实际Megatron模块名 |

每版共11组精确选择集合控制及空列表检查通过，另覆盖 `*input_quantizer`、3个旧alias和一个未映射名称。alias字典只对完整pattern相等时替换：`lm_head`→`*output_layer*`、`*mlp.gate`→`*router*`、`*self_attn*`→`*self_attention*`。它不是任意HF→Megatron名字翻译器；例如 `*self_attn.o_proj`不会自动变成 `*self_attention.linear_proj`。actor使用glob，rollout示例使用显式re:正则，必须对应到同一层，不能照抄同一字符串就认为两边选层相同。`.*`可匹配更深后代，也不是“只允许一层”的路径解析器。

最小原helper复查（本地须已有固定Git对象；这里只展示模块边界与直接叶子两组）：

```bash
python3 -B - /path/to/verl-git <<'PY'
import ast, fnmatch, subprocess, sys
for rev in ['16ec093cc4a235c603c29aed1f2c4e9f551fab48',
            '5c7e2d2ed49bd9f448043883d96b6e191e187e95']:
    src = subprocess.check_output(['git', '-C', sys.argv[1], 'show',
                                   rev + ':verl/utils/qat/modelopt.py'])
    f = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
             and n.name == '_ignore_patterns_to_quant_cfg')
    ns = {}
    exec(compile(ast.Module(body=[f], type_ignores=[]), 'original-helper', 'exec'), ns)
    for pattern in ['projection', 'projection.weight_quantizer']:
        rules = ns[f.name]([pattern])
        print(rev[:8], pattern, [n for n in ['projection.input_quantizer',
              'projection.weight_quantizer', 'projection_extra.weight_quantizer']
              if any(fnmatch.fnmatch(n, r['quantizer_name']) for r in rules)])
PY
```

### 配置检查与真实行为证据

[原测试](https://github.com/verl-project/verl/blob/5c7e2d2ed49bd9f448043883d96b6e191e187e95/tests/utils/test_modelopt_mxfp8_on_gpu.py)不再只断言配置末尾出现ignore字符串，而是调用真实mtq.quantize后检查两个叶子的is_enabled、兄弟隔离、直接叶子模式、alias和原preset未变；BF16例要求forward、weight.grad和input.grad与未量化参考精确相等。关闭量化器不等于冻结该层参数。这些是原测试的行为要求，本机未执行它们。

作者PR正文报告ModelOpt0.44.0、compressed-tensors0.17/0.18两环境各34项通过；H100/Qwen3-0.6B、单seed、两臂各10步GRPO中，被忽略28层的56个启用量化器变成0，首层QDQ误差由0.0859375到0，参数仍逐步更新。与此同时，选定首层导出BF16与vLLM实际加载权重在**修复前后均11/11相等**。这说明字节refit检查与actor计算精度可以给出不同层面的答案，不能只凭前者认证forward一致。作者未把不同轨迹的reward/K3均值当改进证明，也未跑多GPU、MXFP8端到端或Qwen3.5/GDN；这些都是作者报告，不是本项目真机结果或当前head全栈验收。

固定head的[GPU单测workflow](https://github.com/verl-project/verl/blob/5c7e2d2ed49bd9f448043883d96b6e191e187e95/.github/workflows/gpu_unit_tests.yml)排除`*on_cpu.py`；改名让文件进入该收集范围，并不保证任务实际触发、依赖存在或测试通过。新增BF16例未显式将tensor送到CUDA，文件名本身也不是每个断言都在GPU执行的证据。

M1 Python3.12.13只执行两版完整原helper AST及标准库名称控制，对两版modelopt.py和原测试文件共4模块内存compile通过；未导入torch/ModelOpt/compressed-tensors、运行原pytest、量化器、梯度或设备。已有环境find_spec确认缺modelopt/compressed_tensors，不补装、不以stub替代真实mtq；临时控制不新增永久一次性工具。SHA256：生产base `8ef392c4592728c77cd693d0762564ee74b5c2f7867ac83fc2aad151f3801e9e`、head `c926b4987e48c5e487865c62046de0547bff75cb551bbd50691a3d16129b1828`；测试base `27987bd731b460fb69d13e67bcabb58b041caba1ee80255fed6b7591f1fd8c1f`、head `76ccbb40aa951c1695b1912ccb520d4be5f42fa6bd1608afa3d1b56137382b9b`。真实行为缺口补[原M5量化条目](../../experiments/handoff-m5.md#verl-量化层审计新版复验)，不另排队/应用或开展H100训练；本题收束，不据此诊断#8250或扩写量化器替身。
