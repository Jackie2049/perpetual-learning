#!/usr/bin/env python3
"""真实 Hydra 复现：#6067 删掉 `trainer.use_legacy_worker_impl` 之后，两种写法的行为。

用最小树（不装 verl、不需要 GPU）复刻三种形状，逐条打印**实测**结果：

| 形状 | 树里有没有这个键 | 传法 | 期望 |
|---|---|---|---|
| A | 有（`044bbba2^` 的形状） | `trainer.use_legacy_worker_impl=disable` | compose 成功、值生效 |
| B | 没有（`044bbba2` 之后的形状） | `trainer.use_legacy_worker_impl=disable` | **compose 失败** |
| C | 没有 | `+trainer.use_legacy_worker_impl=disable` | **compose 成功、键被接受** |

C 成功 **不等于** 键被消费：verl 侧没有任何代码再读这个字段（`git grep` 计数 0，见 readme）。
再加一条负向对照 D（存在的键、unprefixed），确保 B 的失败不是"所有 override 都失败"。

依赖：`pip install hydra-core`（本轮用 hydra 1.3.2 / Python 3.12 实测）。
用法：python3 hydra_repro.py [--json out.json]     退出码 0 = 四条都符合期望
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

TREE_WITH_KEY = """trainer:
  device: cuda
  use_legacy_worker_impl: disable
  n_gpus_per_node: 8
"""
TREE_WITHOUT_KEY = """trainer:
  device: cuda
  n_gpus_per_node: 8
"""
WRAPPER = """hydra:
  searchpath:
    - file://{cfgdir}
defaults:
  - ppo_trainer
  - _self_
async_training:
  staleness_threshold: 0.1
"""

CASES = [
    ("A_present_unprefixed", TREE_WITH_KEY, "trainer.use_legacy_worker_impl=disable", "OK"),
    ("B_absent_unprefixed", TREE_WITHOUT_KEY, "trainer.use_legacy_worker_impl=disable", "FAIL"),
    ("C_absent_prefixed", TREE_WITHOUT_KEY, "+trainer.use_legacy_worker_impl=disable", "OK"),
    ("D_control_existing_key", TREE_WITHOUT_KEY, "trainer.device=npu", "OK"),
]


def run_case(workdir: pathlib.Path, tree: str, override: str) -> dict:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    cfgdir = workdir / "trainer" / "config"
    cfgdir.mkdir(parents=True, exist_ok=True)
    (cfgdir / "ppo_trainer.yaml").write_text(tree, encoding="utf-8")
    (cfgdir / "fa.yaml").write_text(WRAPPER.format(cfgdir=cfgdir), encoding="utf-8")
    out = {"override": override}
    with initialize_config_dir(config_dir=str(cfgdir), version_base=None):
        try:
            cfg = compose(config_name="fa", overrides=[override])
            out["result"] = "OK"
            out["trainer"] = dict(OmegaConf.to_container(cfg.trainer, resolve=True))
        except Exception as exc:  # noqa: BLE001 — 失败类型本身就是被测对象
            out["result"] = "FAIL"
            out["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, tree, override, expect in CASES:
            workdir = pathlib.Path(tmp) / name
            got = run_case(workdir, tree, override)
            got.update({"case": name, "expected": expect})
            got["matches"] = got["result"] == expect
            results.append(got)
            print(f"[{'PASS' if got['matches'] else 'FAIL'}] {name:<24} expect={expect:<4} got={got['result']:<4} {got.get('error', '')}")
    ok = all(r["matches"] for r in results)
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
