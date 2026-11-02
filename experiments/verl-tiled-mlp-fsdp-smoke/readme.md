# verl `tiled_mlp` 接线补丁的 FSDP 真机冒烟（TM-006 的 M5 侧补证）

一句话结论：补丁把「配置完全无效」修成「FSDP2 下真实生效」；**FSDP1 下开启会硬报错**（不是静默无效），**且在 verl 默认的梯度检查点开启时几乎没有显存取收益**，只有关掉检查点才从 16.39 GB 降到 8.74 GB。

2026-11-02，4090-1（4×RTX 4090 之一，单卡）。上游版本 `6879e1ab07`（GitHub tarball 直取），补丁为 [tiled-mlp-wiring.patch](../../open-source/verl/patches/tiled-mlp-wiring/tiled-mlp-wiring.patch)（sha256 `6bf60406a41bbac0da787030b53797064d33c4ac8c666c9c7cc74f5027e8d7b8`）。

## 一、为什么要跑

M1 的本地验证只到静态层：`verify_patch.py` 从源码里抽出 `_build_module`，把 `apply_monkey_patch` 换成记录器，证明「配置被传下去了」。这不能回答三个会决定能不能提 PR 的问题：

1. 补丁之后 `tiled_mlp.enabled=True` 真的会 patch 到 MLP 类吗？
2. 打了补丁与不打补丁，数值是否一致？
3. 文档说 TiledMLP 只在 FSDP2 下兼容——**FSDP1 下开启是静默无效，还是直接崩？**

## 二、方法

受测树：`6879e1ab07` 原样 + 打补丁后各一份。构造真实 `FSDPEngine`（`EngineRegistry.new` → `engine.initialize()`），走的就是 `_build_module → apply_monkey_patch` 这条生产路径；然后跑一次真实 `forward_backward_batch`（`ppo_loss`）。

- 小模型：`Qwen3Config(hidden=64, inter=128, layers=2)`，从 config 建，不需下载权重；固定 seed 保存一次，每次运行加载同一份权重（`param_fingerprint` 各次完全一致，见 `results.json` 的 `same_init`）。
- 大模型显存对照：`Qwen2.5-0.5B-Instruct`（本地已有），8192 token 单条。
- 检测「是否真的 patch 上」用模型真实 MLP 类（`qwen2 → Qwen2MLP`、`qwen3 → Qwen3MLP`），看 `forward.__name__` 是否为 `tiled_forward`，并抓 `TiledMLP monkey patch applied to: ...` 日志。
- 数值对照：同一份权重、同一批输入，`enabled=0` 与 `enabled=1` 各跑一次，比较 MLP 权重梯度范数、其余参数梯度范数、logprob 总和。

脚本：[smoke_tiled_mlp.py](smoke_tiled_mlp.py)、[run_smoke.sh](run_smoke.sh)、[cpu_equiv.py](cpu_equiv.py)、[summarize.py](summarize.py)。原始产物在 [raw/](raw/) 与 [logs/](logs/)，md5 见 [md5sums.txt](md5sums.txt)。

## 三、结果

### 3.1 接线是否生效（FSDP2）

| 运行 | `enabled` | 补丁 | MLP 类 forward | 日志 |
|---|---|---|---|---|
| `base_fsdp2_e0` | False | 有 | `forward` | 无 TiledMLP 行 |
| `unwired_fsdp2_e1` | **True** | **无** | `forward` | 无 TiledMLP 行 |
| `wired_fsdp2_e1_s4` | True | 有 | **`tiled_forward`** | `TiledMLP monkey patch applied to: Qwen3MLP (shards=4)` |
| `wired_05b_e1s8` | True | 有 | **`tiled_forward`** | `... Qwen2MLP (shards=8)` |

不打补丁时 `enabled=True` 与 `enabled=False` 的梯度、logprob **逐位相同**（`mlpΔrel = 0.00e+00`），这就是 TM-006 所报「配置被静默吞掉」的直接数值证据。

### 3.2 数值一致性

CPU fp64（`cpu_equiv.py`，同一权重、同一输入）下，tiled 与普通 MLP 的 forward、输入梯度、权重梯度差异在 `1e-15 ~ 1e-16`（浮点舍入级），说明**分片前反向的数学是精确等价的**。包括 `torch.chunk` 返回块数少于 `num_shards` 的三种边界（1/3/5/9 个 token 配 4 片，实际只切出 1/3/3/3 块）仍然精确——那种情况下 `GradientAccumulator` 的「最后一片」分支不触发，梯度退化为常规 autograd 累加，只是少了一次 fp32 累加，不影响结果。

GPU 侧（FSDP2，bf16 混合精度）：

| 用例 | MLP 梯度相对差 | 其余梯度相对差 | logprob 相对差 |
|---|---|---|---|
| tiny 16 tok / 4 片 | 9.1e-06 | 0 | 0 |
| tiny 16 tok / 8 片 | 6.0e-05 | 0 | 0 |
| tiny 16 tok / 16 片 | 3.1e-05 | 0 | 0 |
| tiny 18 tok / 4 片 | 1.5e-03 | 1.2e-03 | 2.0e-05 |
| tiny 18 tok / 4 片，fp32 混合精度 | **1.1e-07** | **6.2e-08** | **0** |
| 0.5B 8192 tok / 8 片 | 1.1e-03 | 2.6e-02 | 1.2e-04 |

同一形状下把混合精度从 bf16 换成 fp32，差异从 1e-3 掉到 1e-7、logprob 逐位相同——**GPU 上的偏差来自 bf16 下不同 GEMM 形状的舍入，不是逻辑错误**。这也意味着开启该选项后，bf16 训练的数值会有 1e-3 量级的漂移（不是位对齐的），做严格复现实验时要知道这一点。

### 3.3 显存（Qwen2.5-0.5B，8192 token，单卡）

| 配置 | baseline 峰值 | tiled 峰值 | 变化 |
|---|---|---|---|
| 梯度检查点关，4 片 | 16.39 GB | 8.74 GB | **−46.7%** |
| 梯度检查点关，8 片 | 16.39 GB | 8.74 GB | −46.7% |
| 梯度检查点关，16 片 | 16.39 GB | 8.74 GB | −46.7% |
| **梯度检查点开（verl 默认），8 片** | 5.46 GB | 5.46 GB | **±0.0%** |

两点值得记：

1. 收益在 4 片就饱和，再增加分片不继续下降——剩下的 8.7 GB 是参数、梯度和 logits，不是 MLP 激活。
2. **verl 默认 `enable_gradient_checkpointing=True`（`hf_model.yaml:34`），此时 TiledMLP 一点显存都不省**。原因是两者机制重叠：TiledMLP 自己就是「前向不存激活、反向按片重算」，而梯度检查点已经做了整层重算。换句话说，这个选项的实际价值等于「用更细粒度的分片重算替代整层重算，且只在没开检查点时才轮得到它」。

### 3.4 FSDP1（`strategy=fsdp`）

三种变体全部**硬报错**，不是静默无效：

| 变体 | 结果 |
|---|---|
| 默认（`use_orig_params=False`） | `RuntimeError: setStorage: sizes [64, 128], strides [1, 64], storage offset 131328, ... out of bounds for storage of size 0` |
| `use_orig_params=True` | `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float` |
| `reshard_after_forward=False` | 与默认同一条 storage 错误 |

机制：TiledMLP 的反向要在**反向阶段重新前向**每一片，而 FSDP1 在前向结束后已经把参数 storage reshard 掉了，重算时读到的是已释放的存储。这与模块 docstring「FSDP2-compatible」一致，但错误信息完全看不出是这个原因。

## 四、对 TM-006 的含义

1. **补丁有效**：FSDP2 下配置从「完全无效」变成「真实生效」，默认行为不变（`enabled=False` 时梯度逐位相同）。
2. **补丁把一个静默问题换成了一个响亮的崩溃**：以前 FSDP1 用户开这个选项什么也不会发生；打补丁后会在第一个反向挂掉，错误信息是 storage 越界。所以如果要提 PR，**建议同时加一条配置期校验**（`tiled_mlp.enabled=True` 且 `strategy != "fsdp2"` 时直接抛明确错误），否则等于给 FSDP1 用户埋了一个难懂的运行期错误。
3. **价值判断要修正**：选项本身在 verl 默认配置下（梯度检查点开启）不省显存。PR 描述里不能写「开启后省内存」，只能写「接线修复 + 说明适用条件」。这一点也影响该问题值不值得提——它是配置项名不副实的证据，而不是一个能带来收益的优化。
4. 上游 `tiled_mlp.py` 自带边界（分片数 > token 数时切块数不足）**不影响正确性**，不构成独立 issue。

## 五、边界

- 单卡、单机、无 optimizer step（只做前反向），没有覆盖多 rank 下的梯度 reduce 与 shard 交互。
- 0.5B 结论是单模型单形状；更大模型下 MLP 激活占比更高，收益比例会不同。
- fp32 隔离实验在 8192 token 下 OOM（`raw/base_05b_fp32.json`、`raw/wired_05b_e1s8_fp32.json` 记录了该边界），最终用 tiny 模型完成精度归因。
- 没有测吞吐/step 时间收益（tiny 模型 0.9–1.4 s/步，差异在噪声内）。
- 未修改上游、未开 issue/PR；一切对外动作待用户逐项授权。

## 六、复现

```bash
# 服务器 /tmp/m5-tiled-mlp
curl -sL -o verl.tar.gz https://codeload.github.com/verl-project/verl/tar.gz/6879e1ab07
tar xzf verl.tar.gz
cp -a verl-6879e1ab07 verl-6879e1ab07-patched
cd verl-6879e1ab07-patched && patch -p1 < /path/to/tiled-mlp-wiring.patch

cd /tmp/m5-tiled-mlp
bash run_smoke.sh                       # 全矩阵
PYTHONPATH=<tree> python cpu_equiv.py --out out/cpu_equiv.json
python summarize.py                     # 汇总为 raw/results.json
```
