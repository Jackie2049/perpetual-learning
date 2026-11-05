#!/usr/bin/env python3
"""负向对照：把判据/夹具各弄坏一次，确认 probe 会变红。

自检（`--selftest`）只证明「当前代码下判据成立」；本脚本反过来证明
「代码被改坏时判据会响」。四种破坏，每种都必须被对应检查抓到：

  A. 夹具把 extra_info 写丢            → ② 正向对照应失败
  B. 夹具把 uid 写成含下划线           → ③ 负向对照应失败（证明 uid 约定不是装饰）
  C. 让 canonical 也走 extra_info 分支 → ④ 应失败
  D. 让所有名字都"可读"（不做键检查）   → ③ 与 ⑦ 应失败

用法：python3 negative_control.py --tree /path/to/verl
退出码：0 = 四种破坏全部被抓到；1 = 有破坏没被抓到（判据不可信）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe_metric_tables as pm  # noqa: E402
import stub_env  # noqa: E402


def _checks(tree=None, monkey=None) -> set[str]:
    """跑一次自检，返回失败项。

    破坏只能施加在 `selftest` 的**真内层**：把 `selftest` 自己替换掉，
    让它在原实现外面套一层替换——这样 `selftest` 内部引用的 `run_metric`
    等模块级名字仍然是它自己的，替换才会生效。
    """
    real_selftest = pm.selftest
    if monkey is not None:
        def wrapped(inner_root: Path) -> list[dict]:
            saved = {}
            for name, value in monkey.items():
                saved[name] = pm.__dict__[name]
                pm.__dict__[name] = value
            try:
                return real_selftest(inner_root)
            finally:
                for name, value in saved.items():
                    pm.__dict__[name] = value

        pm.selftest = wrapped
    try:
        checks = (pm.selftest if monkey is None else pm.selftest)(tree)
    finally:
        pm.selftest = real_selftest
    return {c["check"] for c in checks if not c["ok"]}


def run_case(root: Path) -> list[dict]:
    results: list[dict] = []
    original_run = pm.run_metric
    original_canonical = stub_env.load_replay_buffer_module

    # A. 写盘时把 extra_info 丢掉
    def drop_info(module, fake_tq, metric, extra_info, sessions=2, metric_missing_since=None):
        return original_run(module, fake_tq, metric, None, sessions, metric_missing_since)

    results.append({"case": "A 夹具丢 extra_info", "failed": _checks(root, {"run_metric": drop_info})})

    # B. uid 含下划线
    real_run = original_run

    def bad_uid(module, fake_tq, metric, extra_info, sessions=2, metric_missing_since=None):
        fake_tq.partitions.clear()
        fake_tq.put_group("bad_uid", sessions, extra_info)
        from omegaconf import OmegaConf
        rb = module.ReplayBuffer(
            trainer_mode="sync", trainer_config=OmegaConf.create({}), max_off_policy_threshold=None,
            max_off_policy_strategy="drop", sampler_kwargs=OmegaConf.create({}),
            filter_groups_metric=metric, refill_fn=lambda n: 0)
        rb._sync_metadata_from_transfer_queue()
        try:
            rb._dapo_filtered_keys("train")
            return {"metric": metric, "outcome": "ran"}
        except RuntimeError as exc:
            return {"metric": metric, "outcome": "raised", "error": str(exc)}

    results.append({"case": "B uid 含下划线", "failed": _checks(root, {"run_metric": bad_uid})})

    # C. canonical 走 extra_info 分支：把常量改成一个没人写的名字
    def broken_canonical(tree: Path):
        module, fake_tq = original_canonical(tree)
        module.FILTER_GROUPS_REWARD_METRIC = "canonical_broken"
        return module, fake_tq

    results.append({"case": "C canonical 常量被改坏",
                    "failed": _checks(root, {"load_replay_buffer_module": broken_canonical})})

    # D. 「所有名字都可读」：把缺失判定短路掉
    def all_readable(tree: Path):
        module, fake_tq = original_canonical(tree)
        rb_cls = module.ReplayBuffer

        def patched(self, partition_id):
            fake_tq.partitions.clear()
            fake_tq.put_group("probeuid", 2, {"anything": 1.0})
            self._sync_metadata_from_transfer_queue()
            return set(), {}

        rb_cls._dapo_filtered_keys = patched
        return module, fake_tq

    results.append({"case": "D 所有名字都可读",
                    "failed": _checks(root, {"load_replay_buffer_module": all_readable})})

    return results


EXPECTED = {
    "A 夹具丢 extra_info": {"② acc 可读时 ran（正向对照）"},
    "B uid 含下划线": {"② acc 可读时 ran（正向对照）"},
    "C canonical 常量被改坏": {"⑧ 上游测试用的 metric 名都能在表①找到"},
    "D 所有名字都可读": {"③ 无人写的名字 raised（负向对照）", "⑤ 缺一条轨迹的键就整组抛"},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True)
    args = ap.parse_args(argv)
    root = Path(args.tree).expanduser().resolve()

    ok = True
    for res in run_case(root):
        want = EXPECTED[res["case"]]
        hit = want & res["failed"]
        caught = hit == want
        ok = ok and caught
        print(("PASS " if caught else "FAIL ") + res["case"]
              + f"   期望失败检查={sorted(want)} 实际失败={sorted(res['failed'])}")
    print(f"\nnegative_control {'4/4' if ok else '有漏网'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
