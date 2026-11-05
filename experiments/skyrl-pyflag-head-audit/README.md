# SkyRL HEAD 七条「静默失效」重扫：结论不变，但「从没接通」这件事从 18 个 ref 扩到整条历史

## 一句话结论

在 SkyRL `f854d6d9`（2026-11-04）上重跑 [pyflag](../../tools/pyflag/readme.md)，报出 **41 个开关 / 7 条 `silently_inert`**，
逐条查证后**真失效仍然只有 1 条**（`use_liger_kernel`），与 2026-11-04 那批的判定**逐条相同**。
新增的读数全落在同一句上：**这个形参从仓库第一个提交（2025-06-24）到 HEAD，1458 条提交里没有任何一处以关键字实参传过它**，
中间文件改过三次名（`skyrl_train/models.py` → `skyrl_train/model_wrapper.py` → `backends/.../workers/model_wrapper.py`），每一版都带着这个形参。
⇒ **不是"某次重构漏接线"，是从第一天就没人接。**

## 为什么值得再扫一遍

上一批（[experiments/skyrl-pyflag-triage](../skyrl-pyflag-triage/readme.md)，2026-11-04）自己写了边界：
"能说该文件自 2026-02-16 起没接过线，**不能说整个仓库历史上从没有过**"——因为当时是浅克隆 + contents API 逐版本取，
18 次提交就到顶了。而 TM-007 的停止条件又写着"再扫应换新框架而非重扫"。
这两句放在一起有一个具体的缺口：**同一个仓库、换一次**整仓历史**，仍可能有新信息**。

这一轮就是补这件事，顺带把 7 条判据从"读代码读出来"变成"可执行的 20 条判据 + 11 条注入"。

## 读数

**扫描**（`raw/scan-skyrl-f854d6d9.json`，972 个 `.py`）：41 个开关、7 条 `silently_inert`。
七条的形态与判定（`probe.py`，**13/13 判据通过**）：

| # | 开关 | mode | 判定 | 决定性事实 |
|---|---|---|---|---|
| 1 | `use_liger_kernel` | never | **DEAD** | `model_wrapper.py:148` 有形参、`:181` 有分支，但全仓零关键字实参、`config.py` 无字段、`pyproject.toml` 无依赖 |
| 2 | `use_remove_padding` | always | NOT_USER_FACING | `monkey_patch.py:153` 是 `if use_remove_padding or ulysses_sp_size > 1:`；唯一调用点 `worker.py:169` 在 `sequence_parallel_size > 1` 守卫内且不传它 |
| 3 | `enable_sample_support_replay` | never | INDIRECT_PRODUCER | `worker.py:1127` 读配置字段、4 处按关键字传给 `forward` |
| 4 | `enable_pd` | mixed | INDIRECT_PRODUCER | `setup.py:94` 读 `ie_cfg.enable_pd`，`:133`／`:155` 显式传 `True`；`main_base.py:297` 另读一次 |
| 5 | `use_combine` | always | FALSE_POSITIVE | 基类默认 `True`，子类 `Qwen4ExpFinalMixer` 用**位置参数** `super().__init__(config, use_combine=False)` |
| 6 | `use_signals` | mixed | DEPRECATED_BY_DESIGN | `py_functional.py:57` docstring 标 `(Deprecated)`，默认走多进程 |
| 7 | `disable_print` | never | PUBLIC_HELPER_API | `registration.py:113` 的 `pprint` 是公开 helper，docstring 明写 `disable_print=True` 时返回字符串 |

**历史归属**（`history-liger.py`，`raw/history-liger.json`）：H1–H4 四条判据全过，外加一条判据敏感性自检。

- 仓库 **1458 个提交**全部扫过（每条提交一次 `git grep`，27 秒）；
- 该文件 **38 条提交**，**38 条**的版本里都带 `use_liger_kernel=False,`；
- 历史提及该名字 **4647 处**，其中**关键字实参 0 处**（其余是形参声明、docstring、`if flag:` 分支）；
- rename 链三段：`skyrl_train/models.py`（起于 `8805e756`，2025-10-01 前后）→ `skyrl_train/model_wrapper.py`（起于 `0c18fb5b`，2026-01-05）→ `skyrl/backends/skyrl_train/workers/model_wrapper.py`（起于 `b408bafa`，2026-10-30）；
- 最早出现该形参的提交是 `b3280d84`（2025-06-24，仓库第一个提交）。

**可达性**（`reach-liger.py`，`raw/reach-liger.json`）：六条判据全过。

- 配置侧：`config.py` 里没有这个字段 ⇒ Hydra 的 structured config 会在**构配置**这一步拒掉，用户连"传进去"的机会都没有；
- 构造侧：`fsdp_worker.py` 两处 `HFModelWrapper(...)` 都不传（`E2_call_sites=2`，`any_call_passes_it=false`）；
- 依赖侧：`pyproject.toml` 与根 `uv.lock` 都没有 `liger-kernel`；**唯一提到它的是 `skyrl-agent/uv.lock`**——那是另一个子工程（agent 侧）的锁文件，与本包无关，归因时必须写清；
- 本机 `liger_kernel` 不可 import（正向对照：`json` 可 import，证明这条判据不是死的）；分支里是**懒 import**，所以即使有人传 `True`，也只会得到 `ModuleNotFoundError`。

## 判据怎么写才算数

这一批真正的产出是**判据的可注入性**。`probe.py --selftest` 有 11 条注入，每条都改源码里的一处、
要求**指定的那条判据**变红、其余不变；`history-liger.py --selftest` 用 3 正 3 负共 6 个样例钉住"什么算把值传进去"。

自检当场抓出的错误（都在本批内修掉）：

1. **`L5` 一开始把声明文件也算进去**：全仓扫 `use_liger_kernel` 必然命中 `model_wrapper.py` 自己那 3 行，
   于是它只能是"命中恰好 3 处"，任何别处的提及都让它红——包括**注入本该只红 L3 的情况**。改成"声明文件之外有没有人传"。
2. **`S1`／`P1` 数的是行数不是出现次数**：把 `enable_sample_support_replay=sample_support_replay,` 从 4 处删 1 处，
   剩下的行还在 ⇒ 判据不变红。改成 `sum(l.count(...))` 并写死期望值（4／3）。
3. **`D1` 的 `_find(...)` 忘了传 `start`**：docstring 在 `pprint` 之前也有同名短语，`fn[0] < doc[0]` 判据直接不成立。
4. **自检临时树建在本目录里，被 `L5` 的 `os.walk` 扫进去**：五条注入里有四条因此多红一条（`L5`）。
   这是"**负向对照本身被自己的产物污染**"，修法是把临时树拷到本目录下的 `.selftest-tmp/`、并让 `l5` 往上找到含 `skyrl/` 的那一层再扫。
5. **`_pred`（历史脚本里判"是不是关键字实参"）三次写错**：先漏了"形参声明独占一行"、再误伤 `name=self.cfg.x,` 这种真实参。
   最后一次改成"整行就是 `name=<标识符>`（可带逗号）⇒ 形参"，并用 3 正 3 负样本钉住。

## 边界（别当结论引用）

- **扫描只跑了 HEAD 一个点**；历史部分只回答了"`use_liger_kernel` 有没有被传过"，**没有**对 7 条逐一做全历史。
- **历史上限**：`git rev-list HEAD` 的 1458 个提交，不含未合入的分支（该仓库是 trunk 开发，影响很小但要说清）。
- **没有训练读数**：`use_liger_kernel` 的"配不了"是**读代码 + 本机 import** 得出的，没有真跑一次 Hydra 配置校验去确认报错文案。
- **没触上游**：未建 issue、未评论、未开 PR；TM-007 的对外动作仍待用户逐项授权。
- 上一批的一条边界在本批**仍然成立且未变**：`use_remove_padding` 是"形参不产生观测差异"，不是"开关坏了"。

## 复现

```bash
git clone https://github.com/NovaSky-AI/SkyRL.git /tmp/skyrl   # 历史部分需要完整克隆（不带 --depth）

python3 probe.py --snapshot raw/snapshot                    # 13 条判据（快照随仓库提供，无克隆也能跑）
python3 probe.py --snapshot raw/snapshot --selftest          # 11 条注入，退出码 0 = 全部按预期变红
python3 history-liger.py /tmp/skyrl --json raw/history-liger.json --selftest
python3 reach-liger.py --checkout /tmp/skyrl --json raw/reach-liger.json
```

## 产物

- `probe.py` — 13 条判据（7 组）+ 11 条注入自检；`raw/findings-skyrl-f854d6d9.json`、`raw/selftest.txt`
- `history-liger.py` — 整仓历史归属（1458 个 rev）+ 6 例判据敏感性自检；`raw/history-liger.json`、`raw/history-liger.txt`
- `reach-liger.py` — 可达性 6 条判据；`raw/reach-liger.json`
- `raw/scan-skyrl-f854d6d9.json` — pyflag 原始扫描
- `raw/snapshot/` — 11 个源文件逐字节快照，`probe.py` 不依赖克隆

入口：[证据页](../../open-source/skyrl/evidence/pyflag-head-audit-2026-11.md) · [上一批](../skyrl-pyflag-triage/readme.md) · [pyflag](../../tools/pyflag/readme.md) · [TM-007](../../queue_tech-miner.md) · [diary](../../diary/2026-11-05.md)
