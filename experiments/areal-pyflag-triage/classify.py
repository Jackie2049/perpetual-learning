#!/usr/bin/env python3
"""42 个开关 / 7 条 `silently_inert` 逐条归因，每条一个档位。

输入是扫描器原始输出（`raw/scan-areal.json`，**故意保留未修复版**——
"扫描器为什么漏"本身就是本批的读数）。有候选没档位就 exit 1。

档位：
  R1 真实接线（生产者存在，扫描器盲区）
  R2 真不可达（说明仍宣称可用）
  R3 结构内部（无用户面）
"""
import json
import pathlib
import sys

# 每条候选：档位 + 一句可复核的理由（行号对着 3b936798）
VERDICTS = {
    "use_sapo_loss": (
        "R1",
        "生产者在 functools.partial(grpo_loss_fn, ...) 的关键字实参里"
        "（actor.py:612 的 partial、:622 的 use_sapo_loss= 实参）；"
        "扫描器只认『直接调用点里同名关键字』，partial 把它隔开一层 ⇒ 漏判",
    ),
    "use_cispo_loss": (
        "R1",
        "同 use_sapo_loss：actor.py:625 在同一个 functools.partial 里传值；"
        "配置字段在 cli_args.py:2076",
    ),
    "use_decoupled_loss": (
        "R1",
        "同形（actor.py:626），且另有一处真分支 actor.py:170 `if not config.use_decoupled_loss:`；"
        "配置字段 cli_args.py:2115",
    ),
    "use_cpu": (
        "R2",
        "kernels.py:155 的形参 + :211 的 `if use_cpu:` 分支；"
        "唯一调用点 moe/utils.py:212 只传 5 个位置实参、不传它，"
        "全仓（含 tests）无任何生产者 ⇒ 该分支恒走 else（GPU 实现）；"
        "而 docstring:172 仍写『use_cpu: whether to use CPU implementation』",
    ),
    "use_local_output": (
        "R1",
        "parallel_styles.py:42 的形参在 :48 存成 self.use_local_output、"
        "在 :80 由 functools.partial 喂给 _prepare_output_fn；"
        "调用方 fsdp_utils/parallel.py:318 ReplicateParallel(...) 不传 ⇒ 吃默认 True。"
        "默认值就是设计意图（score 层要 to_local），不是失效",
    ),
    "use_dist_ckpt": (
        "R3",
        "checkpointer.py:315 形参、:343 分支；唯一调用点 :459 `self.get_rng_state()` "
        "不传 ⇒ 恒 True（把 rng_state_list 包成 ShardedObject）。"
        "但它没有配置字段、没有 CLI、跨模块无调用者 ⇒ 是内部默认，不是用户开关",
    ),
    "with_rng": (
        "R3",
        "checkpointer.py 4 处形参（:410/:499/:548/:641）+ 真分支（:457/:628）；"
        "生产调用点 megatron_engine.py:1406（save）与 :1437（load）都不传 ⇒ 恒 True。"
        "没有配置字段 ⇒ 内部默认（总是存/读 RNG 状态），不是用户开关",
    ),
}


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    data = json.loads(pathlib.Path(argv[0]).read_text(encoding="utf-8"))
    inert = [f for f in data["flags"] if f["silently_inert"]]
    rows, missing = [], []
    for f in inert:
        v = VERDICTS.get(f["flag"])
        if v is None:
            missing.append(f["flag"])
            continue
        rows.append({"flag": f["flag"], "mode": f["mode"],
                     "defaults": f["defaults"], "producers": f["producers_total"],
                     "verdict": v[0], "reason": v[1]})
    for r in rows:
        print("%-22s %-7s %-8s %s" % (r["flag"], r["mode"], r["verdict"], r["reason"]))
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("\n%d 条候选：%s" % (len(rows), counts))
    if missing:
        print("有候选没档位：%s" % missing)
        return 1
    if len(rows) != len(inert):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
