# AReaL `use_cpu`：CPU 回退开关**没有人打开，也打不开**

对 [AReaL](https://github.com/areal-project/AReaL) `3b936798`（2026-10-19）里
`archon/moe/kernels.py` 的 CPU 回退路径做行为核对。来源是
[扩池第七批](../areal-pyflag-triage/README.md) 里唯一被判"真不可达"的那条 `use_cpu`——
上一批只证明了"没有生产者"，这一批要回答它的后半句：**那段 CPU 实现本身是不是好的**。

- 证据页：[open-source/areal/evidence/use-cpu-fallback-2026-11.md](../../open-source/areal/evidence/use-cpu-fallback-2026-11.md)
- 上游状态：**不对外**（见文末"价值与判断"）

## 结论（先说结果）

| 问题 | 答案 |
|---|---|
| 用户能不能打开 CPU 回退？ | **不能**。唯一调用点 `utils.py:212` 传 5 个位置实参，不含 `use_cpu`；全仓（含 `tests/`）零生产者 |
| 没有 Triton 的机器上这条路能不能用？ | **不能**。`kernels.py` 顶层 `import triton`，整模块进不来；其测试用 `skipif(not HAS_TRITON)` 静默跳过 |
| 那段 CPU 实现本身对不对？ | **在上游自己的缓冲口径下是对的**（逐位等于 GPU 排布） |
| 那两个偏移口径没对齐会怎样？ | 只在**缓冲被压到 token 总数**时才暴露：抛 `RuntimeError`；上游给的缓冲比这大，所以正常不会碰到 |

⇒ 这是一个"**开关没人打开、打开了也用不上**"的设计面问题，不是坏掉的回退。
上一批"CPU 实现从未执行过"的结论成立，本批补的是它是**可用但不可达**。

## 读数

| 项 | 值 |
|---|---|
| 快照 | `3b93679888e9f4929f6f155f13c6a286a62dfd72`（`git status` 干净） |
| 判据 | `probe.py` **6 类 15 条**，全部 PASS |
| 抄本一致性 | `check_snapshot.py` 4 条 PASS，注入自检 **3/3** |
| 随机扫描 | 2550 组输入（R∈{1,2,4,8}、E∈{1,4,16,64}、对齐∈{1,8,16,64}）在**上游缓冲口径**下：跑通 2550、写入条数恒等于 token 总数 2550 |

### P1 开关不可达

`areal/experimental/models/archon/moe/utils.py:212`：

```python
) = generate_permute_indices(
    num_tokens_per_expert,
    num_local_experts,
    ep_degree,
    padded_max_len,
    TOKEN_GROUP_ALIGN_SIZE_M,
)
```

5 个位置实参，`use_cpu` 走默认 `False`；`grep -rn use_cpu` 全仓只剩 `kernels.py` 自己那几行。

### P2 没有 Triton 就进不来

`kernels.py:5-7` 是顶层 `import torch / import triton / import triton.language as tl`。
在无 Triton 的机器上 `import areal.experimental.models.archon.moe.kernels` 直接
`ModuleNotFoundError`（本机实测）。同目录的 `tests/experimental/archon/test_moe_common.py`
用 `skipif(not HAS_TRITON)` 包住这一批用例 ⇒ **缺 Triton 时测试静默略过，CI 不会红**。

### P3 两个偏移口径不同（但不致命）

`fill_indices_cpu` 用两组不一致的数：起笔是**对齐后**的 `write_offsets[e]`（来自
`m_offsets - m_sizes`），累加却是**未对齐**的 `length`。读数里能直接看到后果：
56 格缓冲里 36 格留 `-1`（对齐填充段整段空着）。

### P4 缓冲容量决定成败

`_permute` 给的缓冲是 `round_up(总 token + E × 对齐, 对齐)`（`utils.py:204-205`），
比"对齐后总量"还大。在这个尺寸下抄本与 GPU 排布**逐位相同**（A/B/C 三档均 0 点差异）。
把缓冲压到 token 总数（Triton 内核在上游布局下也放得下）时才会抛
`RuntimeError: upper bound and lower bound inconsistent with step sign`。

⇒ **"两个口径不一致"是真的，"因此这段代码是坏的"不成立。**

## 本批的自错（都由对照抓出，已改）

1. **把"只在窄缓冲下才崩"写成了"一跑就崩"**。第一版用 `max_len = m_offsets[-1]`（对齐后总量）
   当缓冲，于是每档都抛；把缓冲换成上游真正的口径后是 0 点差异。⇒ **报"崩溃"前先确认
   缓冲是调用方实际会给的那个**。
2. **先入为主地假设 CPU 抄本与 GPU 排布不同**，写了一条"必须有差异"的判据，结果三档全是
   0 差异、判据自己红。改成中性陈述（报差异点数）后才看清真实结论。
3. **"与位置脱节"检查误伤正确实现**——它隐含假设排布是 rank-major，而实际是
   expert-major、段内 rank 序。已删掉，改用"与 GPU 排布逐位比对 + 两个改坏对照"。
4. **抄本与快照的比对一度假绿/假红并存**：子集比对看不到"多出来的行"，而多出一个
   `device=` 行又被当内容差异 ⇒ 最终改成"剥 docstring + 括号续行合并 + 两侧同一套
   canon 归一化"再双向比对。

## 脚本

| 文件 | 作用 | 怎么跑 |
|---|---|---|
| `probe.py` | 6 类 15 条判据（开关可达性 / Triton 依赖 / 口径差异 / 容量两档 / 两个对照臂） | `/tmp/verify-py312/bin/python probe.py` |
| `check_snapshot.py` | 抄本与 `raw/upstream/` 逐行双向比对；`--selftest` 3 条注入 | 同上加 `--selftest` |
| `_cpu_impl.py` | 上游 `fill_indices_cpu` 的函数体抄本（三处差异见文件头） | 被 import |
| `_naive_split.py` | 上游 `generate_permute_indices` 算偏移那几行的抄本 | 被 import |
| `_oracle.py` | GPU 排布的 CPU 复刻（语义定义） | 被 import |
| `raw/upstream/` | `kernels.py` / `utils.py` 逐字节快照（sha256 见 `raw/check-snapshot.txt`） | — |

**环境**：本机无 Python 3.12 与 torch，用 `uv venv --python 3.12` + `torch 2.15.0`（CPU 轮子）
建的一次性环境；**无 Triton**（macOS 上 PyPI 无 `cp312` 轮子，这也正是 P2 能被实测的原因）。

```bash
uv venv --python 3.12 /tmp/verify-py312
uv pip install --python /tmp/verify-py312/bin/python torch
cd experiments/areal-use-cpu-fallback
/tmp/verify-py312/bin/python check_snapshot.py && /tmp/verify-py312/bin/python check_snapshot.py --selftest
/tmp/verify-py312/bin/python probe.py
```

## 边界

- 抄本是逐字搬的，读数可信的是"这两段代码自身的行为"；**真实模块在本机 import 不进来**
  （`kernels` 缺 triton、`utils` 随之失败），所以没有"AReaL 在本机跑起来"的读数。
- 只证明 CPU 分支无人打开、且打开也进不来；**没有证明"CPU 实现有错"**——恰恰相反，
  在上游缓冲口径下它是对的。
- 纯 CPU、未装 areal、未编译、未上 GPU/NPU、**未触上游**。

## 价值与判断

**3/10，不对外。** 没有用户受影响（没有配置字段、没有 CLI、默认就关着）；`use_cpu`
是开发期调试开关。真要提，最多是"删掉形参和分支"或"给它接线"，价值都低于 3 分。
本批留下的是一条方法结论：**判"回退路径坏了"之前，先把调用方实际给的缓冲/参数口径
对齐**——本次三次误判里有两次都是这里翻车。
