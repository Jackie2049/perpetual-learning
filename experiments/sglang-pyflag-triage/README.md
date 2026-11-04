# SGLang pyflag 第二批：34 条静默失效里还有几条是真的（2026-11-04）

## 一句话结论

SGLang `27aedf7f` 上 pyflag 报的 **34 条 `silently_inert` 里，源码层真缺陷只有 1 条新发现**：
`logits_processor._get_logits(..., use_logits_buffer=True)` 的默认值让**唯一调用点拿到的
永远是 `False`**，而它不是内部开关——它正是 `InputLogprobProcessor` 那个「逃生口」。

外加 1 条已登记（TM-013 的 `enable_deterministic_inference`）。其余 32 条**不是用户缺陷**，
分五类结构原因，每类都有可机读的判据。⇒ **大仓库上 pyflag 的 `silently_inert` 总数不能当缺陷清单**，
但「分桶 + 逐条绑事实」之后，剩下的数目是可信的。

## 做了什么

1. **自动分桶**（[build_triage.py](build_triage.py) → [triage-buckets.json](triage-buckets.json)）：
   按消费者所在位置把 34 条分到 `kernel_launcher`(2) / `config_dataclass`(3) /
   `runtime_kill_switch`(8) / `source_candidate`(21)。
2. **逐条判定**（[findings.json](findings.json)）：34 条每条一个标签，标签与分桶**对账**
   （内核桶不许写成源码桶的判定，反之亦然）。
3. **读取点分类**（[verify_findings.py](verify_findings.py)）：把每条候选的形参在函数体内的
   每一处读取分进 `branch` / `forward` / `self_attr`，转发型的条目据此排除。
4. **新候选的行为分叉**：取 `_copy_logits_to_buffer` 的**真实源码**在 CPU 上执行，
   两条分支各跑一次，看产物是不是真的不同。
5. **负向对照**（`--selftest`）：3 条扰动全部变红。

## 34 条的分类账

| 标签 | 条数 | 是什么 | 判据 |
|---|---|---|---|
| `KERNEL_CONSTEXPR` | 10 | Triton/cute 核的 `constexpr` 形参 | 消费者在 `srt/kernels/**`、`srt/layers/attention/**`；不传即用默认，两条路径等价 |
| `INTERNAL_OR_DEFAULT` | 12 | `__init__` 里按形参分支，真正的生产者是模块级常量 / `**kwargs` / 默认值 | 读取点全是 `self.*` 或分支，生产者不在前端选项里 |
| `ATTR_PASSING` | 6 | 属性传参（`obj.flag`），工具按名字找关键字，看得见名字、看不见值 | `get_global_server_args().enable_cfg_parallel` 一类 |
| `CONFIG_DEFAULT` | 3 | HF 配置 dataclass 的字段 | `srt/configs/**`，由 `from_dict` 填 |
| `INTERNAL_PRODUCER` | 1 | `use_dsa`：`DSATokenToKVPool.__init__` 内部传 `use_dsa=True`（`memory_pool.py:5245`） | 生产者在同一个类的子类里 |
| `TM013_DUP` | 1 | `enable_deterministic_inference`，已在 TM-013 逐条取证 | 见 [TM-013 证据页](../../open-source/sglang/evidence/grok-router-detflag.md) |
| **`TOPIC_ESCAPE_FAILS_OPEN`** | **1** | **`use_logits_buffer`——本批唯一新发现** | 见下节 |

1+12+6+3+1+1+10 = **34**，逐条以 [findings.json](findings.json) 为准，脚本核对不漏不重。

其中两条值得单独看（都在 `KERNEL_CONSTEXPR` 里，但原因不同）：

- **`use_zigzag_page_table` 是被「转发」而不是被忽略**：`trtllm_mha_backend.forward_extend` 在
  CP 激活时把内部函数 `_trtllm_context_attn` 交给 `cp_strategy.run_attention(...)`，
  `cp/zigzag.py:388` 再用 `use_zigzag_page_table=True` 回调。⇒ 名字看不见生产者，
  是因为生产者藏在一个**不透明的回调参数**里；判据在 `verify_findings.py::validate_zigzag_chain`。
- **`use_clc_scheduler`** 同理但更彻底：它被 `self.use_clc_scheduler = False` 直接写死
  （`python/sglang/kernels/ops/attention/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py:143`），形参只影响一行 `self.sched_warp_id` 的取值。

## 新发现：`use_logits_buffer` 是一个「永远走不到」的逃生口

**它是什么。** `logits_processor._get_logits(..., use_logits_buffer: bool = True)` 控制一件事：
算完 lm_head 之后，结果要不要直接收进 graph 复用的那块显存（`next_token_logits_buffer`），
还是老老实实 `logits.float()` 另行分配。

**为什么可疑。** 形参默认 `True`，但唯一调用点（`python/sglang/srt/layers/logprob_processor.py:657` 的
`get_logits_fn(...)`）显式传了 `use_logits_buffer=num_chunks == 1`。而 `num_chunks` 的定义是

```python
num_chunks = (total_size + chunk_size - 1) // chunk_size
```

`chunk_size = min(max(pruned_states.shape[0], 1), SGLANG_LOGPROB_CHUNK_SIZE)`——
默认 `SGLANG_LOGPROB_CHUNK_SIZE=2048`（`python/sglang/srt/environ.py:1601`），所以只要输入 logprob
那些行超过 2048，**`chunk_size` 就小于总数、`num_chunks > 1`、形参拿到 `False`**。

**原注释说的理由不成立。** 注释（`python/sglang/srt/layers/logits_processor.py:1410-1411`）写的是

> The shared logits buffer is keyed by vocab width and rows; skip it
> when this batch has a different logits shape than the graph buffer.

——这是**形状比对**能占用的那条路径。调用点用「是不是只有一块」代替了它：
单块时不比对形状就占用 buffer，多块时连形状完全匹配也不用。两种偏差都有：

| 场景 | 形状是否匹配 graph buffer | 应该怎么做 | 现在怎么做 |
|---|---|---|---|
| 单块、形状不匹配 | 否 | 跳过的理由成立 | 仍然传 `True`，靠函数体内的形状比对兜住 |
| 多块、某块形状恰好匹配 | 是 | 本来可以用 | 一律 `False`，退回 `.float()` |

⇒ **少了的是「多块里形状匹配的那一块直接用 buffer」**，不是「用了不该用的 buffer」。
所以对外只能说「一个 `bool = True` 的形参在唯一调用点拿不到 `True`、注释里的形状判据被绕开」，
**不能写成「导致显存出错」**——函数体内的形状比对还在，不会写坏东西。

## 行为上确实分叉（不是死代码）

[verify_findings.py](verify_findings.py) 把 `_copy_logits_to_buffer` 的**真实源码**取出来、
桩掉 `torch` 的四个成员，在 CPU 上执行原文件的真实分支，两条分支各跑一次：

| 传值 | 产物 | 退回 `.float()` |
|---|---|---|
| `use_logits_buffer=True` | `graph_buffer<-logits`（直接收进复用的 buffer） | 否 |
| `use_logits_buffer=False` | `logits.float`（另行分配） | 是 |

⇒ 这个形参**改变的是可观测量**（显存里的去向与是否多一次转换），不是等价的两条路。

## 负向对照

`--selftest` 三条，全部变红后恢复（都作用在快照副本上，不动原始快照）：

| 扰动 | 期望 | 实测 |
|---|---|---|
| 让两条分支退化成同一操作 | 判据报「产物相同」 | 报 |
| 删掉 `_copy_logits_to_buffer` | 判据失效被**点名**，不是静默通过 | 点名 |
| 把内核桶的条目写成源码桶的判定 | 分桶对账报不一致 | 报 |

另有两处注入由人工验：把 `use_zigzag_page_table` 判成 `INTERNAL_OR_DEFAULT` ⇒ 被拦；
34 条全改错 ⇒ 只报出 5 条（说明对账是逐条对，不是一条挂掉全挂）。

## 判定不了的部分（必须一起写）

- **没有真机读数**：没跑 SGLang、没上 GPU。`use_logits_buffer` 两条分支的**显存峰值差异**
  与「本该省下的那次转换值多少」都没测，只证明产物不同。
- **是否会被修**：调用点写 `num_chunks == 1` 也可能是「多块时有意不复用 buffer」的
  保守选择（多块会互相覆盖同一块 buffer，注释里也提到 chunk 之间会 alias）。**这一点
  本仓库没有定论**，对外只能问，不能断言该改。
- `use_dsa` 的内部生产者是**同一个类的子类**（`DSATokenToKVPool` 调 `MLATokenToKVPool.__init__`），
  从外部看仍然是「这个形参没人从外面传」——但它的两条分支在子类里被用，不算缺陷。

## 复现

```bash
# 1. 取快照（40 MB，网络不通时走 gh-proxy）
curl -sSL -o /tmp/sglang.tar.gz \
  https://gh-proxy.com/https://github.com/sgl-project/sglang/archive/27aedf7f.tar.gz
mkdir -p /tmp/sglang-src && tar xzf /tmp/sglang.tar.gz -C /tmp/sglang-src --strip-components=1

# 2. 分桶（读 pyflag 已有的扫描结果，不需要重扫 10 分钟）
python3 build_triage.py /tmp/sglang-src \
  --pyflag-json ../../tools/pyflag/scan-sglang-main.json \
  --json triage-buckets.json

# 3. 逐条对账 + 行为分叉 + 读取点分类
python3 verify_findings.py /tmp/sglang-src --json findings-readings.json   # exit 0
python3 verify_findings.py /tmp/sglang-src --selftest                      # exit 0
```

环境：macOS（Apple Silicon）、Python 3.9.6、**未装 SGLang、不需要 torch**——脚本只解析 AST，
行为分叉那段用桩替换 torch 的四个成员。快照为 `27aedf7f` 的 tarball，与 pyflag 扫描用的是同一 ref。

## 自错

1. **分桶脚本第一版把「`__init__` 里存 self」整类说成「调用点走 `*_ENABLED` 常量」**——
   这只对了 `enable_linear_replayssm` 一条，`use_dsa` 是同一形状却有内部生产者。
   已改成「线索不是结论」，并把 `use_dsa` 单独取证。
2. **第一次跑负向对照用的是「等价分支」的单点扰动**：只把 `return cast_to_float32_logits(...)`
   改成 `return logits_buffer`，产物名不同但都没有 `.float()`，判据仍说「分叉」。
   ⇒ 判据改成「产物名相同 **或** False 分支不再退回 `.float()`」，两条都要看。
3. **误判注入最初选 `disable_recompute` / `use_safe_gate`**，两条都在源码桶，改了标签对账也不报；
   换成内核桶的 `enable_bmm` 才真的把「分桶与判定不一致」这条判据点亮。

## 主文档与相关

- [证据页](../../open-source/sglang/evidence/logits-buffer-escape-hatch.md)
- [TM-013（第一批）](../../queue_tech-miner.md) · [第一批实验](../sglang-grok-router-detflag/README.md)
- [pyflag 方法](../../tools/pyflag/readme.md)
