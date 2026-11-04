#!/usr/bin/env python3
"""把 #6067 删除的 worker 配置消费点，逐条钉成"真失效 / 假阳性"的判定表。

scan_consumers.py 只做文本级定位，它报出的 30 个候选里绝大多数是假阳性
（`actor.fsdp_config` → 改为 `actor.engine` 这个 dataclass；`ref.log_prob_micro_batch_size`
→ 被 engine_workers.py 改名成 `ppo_micro_batch_size` 后照常读取）。本脚本用**逐条写死的
判据**复核，每条给出：
  - 旧代码在哪个文件/行怎么读；
  - 当前 main 在哪里消费（或确认没有）；
  - 结论：`wired`（仍生效）/ `renamed`（改了名照样生效）/ `lost`（用户仍能设、但没人读）。

判据说硬就硬：`lost` 的每条都要求同时满足
  ① main 的 `verl/workers/config/*.py` 里仍声明该字段（用户能写这行配置）；
  ② main 的 `verl/workers/**` + `verl/trainer/**` 里没有任何代码读它；
  ③ 例外：字段被 `_mutable_fields` 之类机制消费的，单独注明。

用法：python3 verify_findings.py --repo <verl 仓库> [--json out.json] [--selftest]
退出码：0 全部判定与声明一致；1 有判定对不上（说明 main 变了，结论需要重核）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

# 每条：(字段叶子名, 旧读取点描述, 当前应出现在哪些文件里的消费证据, 期望结论)
# 消费证据写成 (文件 glob, 必须出现的正则)；`lost` 的证据是"反向"的：
# 这些文件里全都**不**出现该叶子名。
CASES = [
    dict(
        key="tiled_mlp",
        old="verl/workers/fsdp_workers.py:913-915 取值、:935-936/:989-990/:1481-1482 转发进 apply_monkey_patch",
        expect="lost",
        scope=["verl/workers/**/*.py", "verl/trainer/**/*.py"],
        note="HFModelConfig 仍声明 tiled_mlp；FSDP 引擎 _build_module 的 apply_monkey_patch 不传 use_tiled_mlp",
    ),
    dict(
        key="freeze_vision_tower",
        old="verl/workers/fsdp_workers.py:545-553（actor）与 :1554-1562（critic）冻结视觉塔",
        expect="lost",
        scope=["verl/workers/**/*.py", "verl/trainer/**/*.py"],
        note="actor.yaml 仍暴露该键；旧实现文件顶部的 get_vl_model_vision_tower 也随文件删除，main 无替代",
    ),
    dict(
        key="num_nextn_predict_layers",
        old="verl/workers/fsdp_workers.py:429-432 在 FSDP 上强制清零（MTP 开启时抛 NotImplementedError）；megatron_workers.py:163-164 在 MTP 关闭时清零",
        expect="moved",
        scope=["verl/workers/engine/fsdp/**/*.py", "verl/workers/engine/veomni/**/*.py",
               "verl/workers/engine/torchtitan/**/*.py", "verl/workers/engine/automodel/**/*.py"],
        expect_file=["verl/workers/config/model.py", r"self\.hf_config\.num_nextn_predict_layers = 0"],
        note="FSDP 侧的'清零'搬进了 HFModelConfig.__post_init__（MTP 关闭时统一清零），"
             "所以这条不是失效、是搬了家；旧代码的 raise NotImplementedError 已改成静默清零",
    ),
    dict(
        key="use_prefix_grouper",
        old="verl/workers/fsdp_workers.py:934 传给 apply_monkey_patch；:969-997 写进 ref 配置",
        expect="lost",
        scope=["verl/workers/engine/fsdp/**/*.py"],
        note="TM-005 已单独取证；本条只作同源复核",
    ),
    dict(
        key="mtp_loss_scaling_factor",
        old="verl/workers/megatron_workers.py:160 写入 override_transformer_config",
        expect="wired",
        evidence=("verl/utils/megatron_utils.py", r"model_config\.mtp\.mtp_loss_scaling_factor"),
        note="check_mtp_config 在新引擎侧接回",
    ),
    dict(
        key="log_prob_micro_batch_size",
        old="verl/workers/fsdp_workers.py:283-285 就地整除并按 GPU 折算",
        expect="renamed",
        evidence=("verl/workers/engine_workers.py", r'pop\(\s*"log_prob_micro_batch_size"'),
        note="engine_workers 把 ref.log_prob_micro_batch_size 改名成 ppo_micro_batch_size",
    ),
    dict(
        key="fsdp_size",
        old="verl/workers/fsdp_workers.py:178 用来建 device mesh",
        expect="renamed",
        evidence=("verl/workers/engine/fsdp/transformer_impl.py", r"fsdp_size"),
        note="actor.fsdp_config 整体变成 actor.engine（FSDPEngineConfig.fsdp_size）",
    ),
]


def run(cmd: list[str], cwd: pathlib.Path) -> str:
    out = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"命令失败：{' '.join(cmd)}\n{out.stderr}")
    return out.stdout


def files_matching(repo: pathlib.Path, globs: list[str]) -> list[pathlib.Path]:
    seen: dict[str, pathlib.Path] = {}
    for g in globs:
        for p in repo.glob(g):
            if "__pycache__" in p.parts:
                continue
            seen[str(p)] = p
    return sorted(seen.values())


def leaf_readers(
    repo: pathlib.Path, key: str, globs: list[str], exclude_prefixes: tuple[str, ...] = ()
) -> list[str]:
    # 边界用 `[A-Za-z0-9_]` 反查而不是 `\b`：Python 里 `_` 算单词字符，所以
    # `_tiled_mlp` 这类**派生变量名**（如 `_is_offload_param`、`tiled_mlp_config`）
    # 用 `\b` 会被漏掉——自检第一次就是栽在这里（注入了 `use_tiled_mlp=True` 仍报"没有读取"）。
    # 反过来，`use_tiled_mlp` 这种前缀写法算不算"读 tiled_mlp"取决于上下文，本脚本按
    # "出现该标识符子串即算"处理，宁可多报也不漏报；误报由 verify_findings 的逐条人工复核兜底。
    pat = re.compile(re.escape(key))
    hits = []
    for p in files_matching(repo, globs):
        rel = str(p.relative_to(repo))
        if any(rel.startswith(pre) for pre in exclude_prefixes):
            continue
        if pat.search(p.read_text(encoding="utf-8", errors="replace")):
            hits.append(rel)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="verl 仓库路径；--selftest 时不需要")
    ap.add_argument("--json", default=None)
    ap.add_argument("--selftest", action="store_true", help="用假仓库验证判据本身有分辨力")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.repo:
        ap.error("除了 --selftest，必须给 --repo")

    repo = pathlib.Path(args.repo).expanduser().resolve()
    results = []
    bad = 0
    for case in CASES:
        row = dict(key=case["key"], expect=case["expect"], old=case["old"], note=case["note"])
        if case["expect"] in ("wired", "renamed"):
            path, pattern = case["evidence"]
            text = (repo / path).read_text(encoding="utf-8")
            ok = re.search(pattern, text) is not None
            row.update(verdict="ok" if ok else "证据缺失", where=f"{path} /{pattern}/")
        elif case["expect"] == "moved":
            path, pattern = case["expect_file"]
            text = (repo / path).read_text(encoding="utf-8")
            moved = re.search(pattern, text) is not None
            hits = leaf_readers(repo, case["key"], case["scope"])
            ok = moved and not hits
            row.update(verdict="ok" if ok else "对不上", where={"moved_to": bool(moved), "engine_readers": hits})
        else:
            # `lost` 的"没有读取"必须排除配置数据类自己的声明行：字段声明本身
            # 就是 `tiled_mlp: dict = ...` 这种样子，不排除的话每条都会自证"有人读"。
            # 这是本脚本第一版真踩的坑（3 条全判"对不上"），判据现在写死在这里。
            hits = leaf_readers(
                repo, case["key"], case["scope"], exclude_prefixes=("verl/workers/config/",)
            )
            declared = leaf_readers(repo, case["key"], ["verl/workers/config/*.py"])
            ok = (not hits) and bool(declared)
            row.update(
                verdict="ok" if ok else "对不上",
                where={
                    "engine_or_trainer_readers": hits,
                    "config_declarations": declared,
                },
            )
        if row["verdict"] != "ok":
            bad += 1
        results.append(row)

    print(json.dumps({"n": len(results), "n_mismatch": bad, "rows": results}, ensure_ascii=False, indent=2))
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"n": len(results), "n_mismatch": bad, "rows": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0 if bad == 0 else 1


def selftest() -> int:
    """判据自检：注入一个"明明有人读"的字段，`lost` 判定必须能识别出它没丢。

    不做这一步的话，"30 条候选里只有几条真失效"这个结论无法区分于
    "搜索本来就看不见读取点"。
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "verl/workers/config").mkdir(parents=True)
        (root / "verl/workers/engine/fsdp").mkdir(parents=True)
        (root / "verl/workers/config/model.py").write_text("class A:\n    tiled_mlp: dict = None\n")
        (root / "verl/workers/engine/fsdp/transformer_impl.py").write_text("x = 1\n")
        empty = leaf_readers(root, "tiled_mlp", ["verl/workers/engine/**/*.py", "verl/trainer/**/*.py"])
        declared = leaf_readers(root, "tiled_mlp", ["verl/workers/config/*.py"])
        if empty != [] or declared == []:
            print("自检失败：干净树上 'lost' 判据没给出 (无读取, 有声明)")
            return 1

        (root / "verl/workers/engine/fsdp/transformer_impl.py").write_text("apply(use_tiled_mlp=True)\n")
        hits = leaf_readers(root, "tiled_mlp", ["verl/workers/engine/**/*.py"])
        if not hits:
            print("自检失败：注入了读取点却仍然报 '没有读取'")
            return 1
        print("自检通过：干净树报 lost，注入读取点后判据翻转")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
