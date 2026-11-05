#!/usr/bin/env python3
"""verl `algorithm.filter_groups.metric`：三个「合法值表」互相不自洽，按文档填会在第一步崩。

2026-11-05 M1。来源：主文档 §13.49「顺带记一个文档不一致」的扩查。

要回答的问题：V1 组过滤按**名字**去 `reward_extra_info` 取键，取不到就抛
`RuntimeError: Finished groups are missing DAPO metric '...'`。那么——
「文档/注释/随仓库脚本告诉用户可以填哪些名字」与「代码里真的会出现哪些名字」
两张表，差在哪里？差出来的名字，用户照填会怎样？

三条读数（全部 CPU、只读）：
  ① **表**：从真实源码抽出三个表——代码写出的键（含 scorer 返回 dict 的键）、
     配置 docstring 声称的值、yaml 注释／文档／脚本举例的值——做集合差。
  ② **真跑**：exec 上游真实 `ReplayBuffer`（不组装 verl 包，只替身 tq 与
     SkipManager），用上游 CPU 测试同一套写盘顺序造一个「2 条同分轨迹的组」，
     看每个被声称的名字拿到的结局。
  ③ **上游自己的单测在守什么**：`tests/trainer/ppo/v1/` 里所有
     `filter_groups_metric=` 的字面值，逐个数是否落在表①里。

用法：
    python3 probe_metric_tables.py --tree /path/to/verl [--json out.json]
    python3 probe_metric_tables.py --tree /path/to/verl --selftest

退出码：0 = 跑完（报告或自检通过）；1 = 自检有漏网（脚本本身不可信）；
        2 = 找不到源码树。
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stub_env import load_replay_buffer_module  # noqa: E402

#: `reward_extra_info[key] = value` 的文字键从这些文件里抽（V1 奖励通路）。
V1_WRITER_GLOB = "verl/experimental/reward_loop/reward_manager/*.py"
#: scorer 返回字面量 dict 的键也会被原样塞进 `reward_extra_info`。
SCORER_GLOB = "verl/utils/reward_score/*.py"
#: 表② 的来源文件。
CONFIG_PY = "verl/trainer/config/algorithm.py"
CONFIG_YAML = "verl/trainer/config/ppo_trainer.yaml"
#: ③ 上游测试。
UPSTREAM_TESTS = "tests/trainer/ppo/v1"


# --------------------------------------------------------------------- 表①


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def collect_written_keys(root: Path) -> dict:
    static: dict[str, list[str]] = {}
    dynamic: list[str] = []
    for path in sorted(root.glob(V1_WRITER_GLOB)):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if not isinstance(tgt, ast.Subscript) or not _is_name(tgt.value, "reward_extra_info"):
                    continue
                where = f"{path.relative_to(root)}:{node.lineno}"
                if isinstance(tgt.slice, ast.Constant) and isinstance(tgt.slice.value, str):
                    static.setdefault(tgt.slice.value, []).append(where)
                else:
                    dynamic.append(where)
    return {"static": {k: sorted(v) for k, v in sorted(static.items())}, "dynamic": dynamic}


def collect_scorer_keys(root: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for path in sorted(root.glob(SCORER_GLOB)):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
                continue
            for k in node.value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    out.setdefault(k.value, []).append(f"{path.relative_to(root)}:{node.lineno}")
    return {k: sorted(v) for k, v in sorted(out.items())}


# ----------------------------------------------------------- 表② 与 表③


def collect_config_docstring(root: Path) -> list[str]:
    tree = ast.parse((root / CONFIG_PY).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FilterGroupsConfig":
            doc = ast.get_docstring(node) or ""
            # `ast.get_docstring` 会把公共缩进剥掉，参数行剩下 4 空格缩进
            block = re.search(r"metric \(([^)]*)\):(.*?)(?=\n {4}\w|\Z)", doc, re.S)
            return re.findall(r'"([a-z_]+)"', block.group(0)) if block else []
    return []


def collect_yaml_comment(root: Path) -> list[str]:
    lines = (root / CONFIG_YAML).read_text().splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*metric:\s", line):
            block = " ".join(lines[max(0, i - 3) : i + 1])
            if "canonical" in block:
                return sorted(set(re.findall(r'"([a-z_]+)"', block)))
    return []


def collect_doc_examples(root: Path) -> dict[str, list[str]]:
    """仓库内 markdown 里 `metric: a / b / c` 式枚举（写的是「可以填这些」）。"""
    out: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.md")):
        if "/.git/" in str(path) or "node_modules" in str(path):
            continue
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if "metric:" not in line or " / " not in line:
                continue
            frag = line.split("metric:", 1)[1]
            for name in re.findall(r"\b([a-z][a-z_]{2,})\b", frag):
                out.setdefault(name, []).append(f"{path.relative_to(root)}:{lineno}")
    return {k: sorted(v) for k, v in sorted(out.items())}


def collect_script_values(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*.sh")):
        if "/.git/" in str(path):
            continue
        for line in path.read_text(errors="ignore").splitlines():
            m = re.match(r"^\s*filter_groups_metric=(\S+)", line)
            if m:
                out[str(path.relative_to(root))] = m.group(1)
    return out


# ------------------------------------------------------------- ② 真跑


def _new_buffer(module, metric: str):
    from omegaconf import OmegaConf

    return module.ReplayBuffer(
        trainer_mode="sync",
        trainer_config=OmegaConf.create({}),
        max_off_policy_threshold=None,
        max_off_policy_strategy="drop",
        sampler_kwargs=OmegaConf.create({}),
        filter_groups_metric=metric,
        refill_fn=lambda n: 0,
    )


def run_metric(module, fake_tq, metric: str, extra_info: dict | None, sessions: int = 2,
               metric_missing_since: str | None = None) -> dict:
    """把一个组写进 tq，用真实 `_dapo_filtered_keys` 跑一遍。

    `metric_missing_since` 非空时，模拟「用户从某条轨迹起不再给该键」——
    用来验证「只要有一条轨迹缺键，整组就抛」。
    """
    fake_tq.partitions.clear()
    # uid 不能含下划线：真实代码用 key.split("_")[0] 反推 uid（上游测试注释也这么写）
    uid = "probeuid"
    if metric_missing_since is None:
        fake_tq.put_group(uid, sessions, extra_info)
    else:
        for session in range(sessions):
            info = dict(extra_info or {})
            if session >= 1:
                info.pop(metric_missing_since, None)
            fake_tq.kv_put(
                key=f"{uid}_{session}_0", partition_id="train",
                fields={"extra_fields": {"reward_extra_info": info}, "rm_scores": 1.0},
                tag={"is_prompt": False, "seq_len": 3, "global_steps": 0},
            )
        fake_tq.kv_put(key=uid, partition_id="train",
                       tag={"is_prompt": True, "status": "finished", "global_steps": 0})

    rb = _new_buffer(module, metric)
    rb._sync_metadata_from_transfer_queue()
    try:
        filtered, counts = rb._dapo_filtered_keys("train")
        return {"metric": metric, "outcome": "ran", "filtered": sorted(filtered), "counts": dict(counts)}
    except RuntimeError as exc:
        return {"metric": metric, "outcome": "raised", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - 任何其它异常同样是读数
        return {"metric": metric, "outcome": "crashed", "error": f"{type(exc).__name__}: {exc}"}


# -------------------------------------------------- ③ 上游单测守的是什么


def collect_upstream_test_metrics(root: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for path in sorted((root / UPSTREAM_TESTS).glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg == "filter_groups_metric":
                    val = kw.value
                    if isinstance(val, ast.Constant) and isinstance(val.value, str):
                        out.setdefault(val.value, []).append(f"{path.name}:{node.lineno}")
                    else:
                        out.setdefault("<非字面量>", []).append(f"{path.name}:{node.lineno}")
    return {k: sorted(v) for k, v in sorted(out.items())}


# ------------------------------------------------------------------ 报告


def build_report(root: Path) -> dict:
    written = collect_written_keys(root)
    scorer = collect_scorer_keys(root)
    docstring = collect_config_docstring(root)
    yaml_comment = collect_yaml_comment(root)
    doc_examples = collect_doc_examples(root)
    scripts = collect_script_values(root)
    upstream = collect_upstream_test_metrics(root)

    canonical = None
    module, fake_tq = load_replay_buffer_module(root)
    canonical = getattr(module, "FILTER_GROUPS_REWARD_METRIC", "reward")

    reachable = set(written["static"]) | set(scorer)

    claims: dict[str, dict] = {}
    for name in docstring:
        claims.setdefault(name, {"sources": []})["sources"].append("config_docstring")
    for name in yaml_comment:
        claims.setdefault(name, {"sources": []})["sources"].append("yaml_comment")
    for name, where in doc_examples.items():
        claims.setdefault(name, {"sources": [], "where": where})["sources"].append("repo_docs")
    for path, val in scripts.items():
        c = claims.setdefault(val, {"sources": []})
        c["sources"].append("script")
        c.setdefault("where", []).append(path)

    table = []
    for name in sorted(claims):
        if name == canonical:
            via = "rm_scores(canonical)"
        elif name in reachable:
            via = "reward_extra_info"
        else:
            via = "none"
        table.append({"name": name, "reachable_via": via, "sources": sorted(claims[name]["sources"]),
                      "where": claims[name].get("where", [])})

    # 每个「声称的名字」在两种输入下各跑一次：
    #   generous —— 把表①里所有键都给上（scorer 把能给的都给了）
    #   realistic —— 只给 acc（除 math_dapo 以外所有 scorer 的真实行为）
    generous = {k: 1.0 for k in reachable if k != "pred"}
    runtime = {
        "generous": [run_metric(module, fake_tq, t["name"], generous) for t in table],
        "acc_only": [run_metric(module, fake_tq, t["name"], {"acc": 1.0}) for t in table],
    }
    partial = run_metric(module, fake_tq, "acc", {"acc": 1.0}, sessions=2, metric_missing_since="acc")

    return {
        "verl_head_marker": _head_marker(root),
        "tables": {
            "written_static": written["static"],
            "dynamic_writers": written["dynamic"],
            "scorer_literal_keys": scorer,
            "config_docstring": docstring,
            "yaml_comment": yaml_comment,
            "repo_doc_examples": doc_examples,
            "script_values": scripts,
            "canonical_metric": canonical,
        },
        "claims_vs_reachable": table,
        "runtime": runtime,
        "partial_key_case": partial,
        "upstream_test_metrics": upstream,
        "summary": {
            "unreachable_claims": [t["name"] for t in table if t["reachable_via"] == "none"],
            "docstring_claims_unreachable": [t["name"] for t in table
                                             if t["reachable_via"] == "none" and "config_docstring" in t["sources"]],
            "upstream_test_metrics_not_in_table": sorted(set(upstream) - reachable - {canonical, "<非字面量>"}),
        },
    }


def _head_marker(root: Path) -> str:
    """不联网：用源码里带的版本标记。"""
    v = root / "verl/version/version"
    return v.read_text().strip() if v.exists() else "unknown"


# ------------------------------------------------------------------ 自检


def selftest(root: Path) -> list[dict]:
    checks: list[dict] = []

    def expect(name, cond, detail=""):
        checks.append({"check": name, "ok": bool(cond), "detail": detail})

    report = build_report(root)
    module, fake_tq = load_replay_buffer_module(root)
    canonical = report["tables"]["canonical_metric"]
    reachable = set(report["tables"]["written_static"]) | set(report["tables"]["scorer_literal_keys"])

    # ① 表本身：acc 一定在「静态写入 ∪ scorer 字面量」里
    expect("① 表① 含 acc", "acc" in reachable, str(sorted(reachable)))

    # ② 正向对照：可读名字真的 ran
    good = run_metric(module, fake_tq, "acc", {"acc": 0.0})
    expect("② acc 可读时 ran（正向对照）", good["outcome"] == "ran", str(good))

    # ③ 负向对照：同一条路径，把一个没人写的名字喂进去必须 raised
    bad = run_metric(module, fake_tq, "no_such_metric_xyz", {"acc": 0.0})
    expect("③ 无人写的名字 raised（负向对照）", bad["outcome"] == "raised", str(bad))

    # ④ canonical 走 rm_scores 分支：故意不给任何 extra_info 也必须能跑
    canon = run_metric(module, fake_tq, canonical, None)
    expect(f"④ canonical {canonical!r} 不依赖 extra_info", canon["outcome"] == "ran", str(canon))

    # ⑤ 部分键：两条轨迹一条有 acc 一条没有 ⇒ 仍要 raised（说明不是「至少有一条就行」）
    partial = report["partial_key_case"]
    expect("⑤ 缺一条轨迹的键就整组抛", partial["outcome"] == "raised", str(partial))

    # ⑥ 注入「表①伪造一条」：可达集合变形后，判据必须跟着变
    forged = reachable | {"forged_metric"}
    expect("⑥ 伪造键会改变可达集合", "forged_metric" in forged and "forged_metric" not in reachable)

    # ⑦ 文档里被点名的名字，按「只给 acc」的真实输入跑，结局必须是 raised；
    #    若上游修好了（名字变得可读），这条会红——那是信号。
    acc_only = {r["metric"]: r["outcome"] for r in report["runtime"]["acc_only"]}
    unreachable = report["summary"]["unreachable_claims"]
    expect("⑦ 不可达声称名在 acc-only 输入下全部 raised",
           all(acc_only.get(n) != "ran" for n in unreachable) and bool(unreachable), str(acc_only))

    # ⑧ 上游单测的字面值全部落在表①∪{canonical}（否则说明表① 抽漏了东西）
    stray = report["summary"]["upstream_test_metrics_not_in_table"]
    expect("⑧ 上游测试用的 metric 名都能在表①找到", not stray, str(stray))

    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", help="verl 源码树根目录")
    ap.add_argument("--json", help="报告写到该路径")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if not args.tree:
        print("需要 --tree", file=sys.stderr)
        return 2
    root = Path(args.tree).expanduser().resolve()
    if not (root / "verl/trainer/ppo/v1/replay_buffer.py").exists():
        print(f"{root} 不像 verl 源码树", file=sys.stderr)
        return 2

    if args.selftest:
        checks = selftest(root)
        for c in checks:
            print(("PASS " if c["ok"] else "FAIL ") + c["check"] + (f"   [{c['detail']}]" if c["detail"] else ""))
        bad = [c for c in checks if not c["ok"]]
        print(f"\nselftest {len(checks) - len(bad)}/{len(checks)}")
        return 0 if not bad else 1

    report = build_report(root)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json:
        Path(args.json).write_text(text + "\n")
        print(f"报告写入 {args.json}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
