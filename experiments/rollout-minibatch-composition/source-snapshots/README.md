# 源码快照（只读，用于 anchor 可核验）

本目录保存 §13.32 用到的**四个框架主分支**的关键文件副本，用途只有一个：
让 `train_side_composition.py` 的 `check_anchor` 能在本机离线核验"结论依赖的那几行代码还在"。
不带任何上游写操作，也没有克隆整仓库。

| 目录 | 仓库 | commit | 取用文件 |
|---|---|---|---|
| `verl/main/` | volcengine/verl | `1a9e711e`（本地 checkout，与 §13.30/§13.31 同一锚点） | `trainer/ppo/core_algos.py`、`utils/groupwise.py`、`workers/engine_workers.py` |
| `openrlhf/main/` | OpenRLHF/OpenRLHF | `dc2a7ad3`（2026-09-17 主分支头） | `trainer/ppo_utils/experience_maker.py` |
| `skyrl/main/` | NovaSky-AI/SkyRL | `40606751`（2026-11-01 主分支头） | `backends/skyrl_train/utils/ppo_utils.py` |
| `nemo_rl/main/` | NVIDIA-NeMo/RL | `6e68b179`（2026-11-01 主分支头） | `algorithms/utils.py`、`algorithms/advantage_estimator.py` |

**局限**：

- 三个非 verl 快照通过 GitHub contents API 逐个取 RAW 文件，**没有克隆仓库**，
  因此无法用 `git rev-parse` 复核；commit 用 `repos/<org>/<repo>/commits?per_page=1`
  取得，取文件时也显式带了该 commit 作为 `ref`。
- 只取与"优势 mean/std 按什么分组、loss 用什么分母"相关的文件；被引用的其它模块
  （如 OpenRLHF 的 deepspeed strategy、SkyRL 的 torch_utils）未取，正文里对它们的
  描述来自同一次只读查阅，未在本目录留存副本。

## 许可与转载说明

- 上游四个项目均为 **Apache License 2.0**（OpenRLHF／SkyRL／NeMo RL 由 GitHub 仓库元数据确认；verl 仓库 header 与 LICENSE 文件均为 Apache-2.0，GitHub 元数据未标 SPDX 标识）。
- 本目录的文件是**原样副本**，保留上游版权头，未做任何修改；仅用于本仓库内的只读锚点核验，不参与构建、不被 `train_side_composition.py` 以外的地方引用。
- Apache-2.0 要求转载时保留版权与许可声明，故文件头的声明原样保留；如需在这些文件基础上修改再分发，须按 Apache-2.0 附加变更说明。
- 若后续只想要"证据"而不想要上游代码副本，可改为只在 `run-info.json` 记录 `(repo, commit, 文件, 行号, sha256)` 而不存正文；当时选择存副本是为了让 `--verify` 真正能离线复跑。
