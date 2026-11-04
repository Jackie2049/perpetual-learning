#!/usr/bin/env python3
"""#6067 第三个体检信号族：**被删掉的启动配置键，还有谁在命令行里传**。

前两批只覆盖"被整份删除的文件"：
- 配置消费点体检（`verl-6067-config-consumer-scan`）看引擎代码还读不读；
- 接口面体检（`verl-6067-api-surface-scan`）看文档还教不教这些名字。

本批换到"被**改动**而非删除"的启动入口：`verl/trainer/config/ppo_trainer.yaml` 里
`trainer.use_legacy_worker_impl` 被删掉，而 Hydra 对**不存在的键**会直接报
`Could not override 'trainer.X'. Key 'X' is not in struct` —— 这是**运行期可见的硬失败**，
不是静默失效。问题只在"还有谁在传它"。

做法（纯静态，不跑 verl、不需要 hydra）：
1. 解析 `044bbba2^` 与 `044bbba2` 两份 `ppo_trainer.yaml` 的**叶子键全路径**（缩进栈，跳过列表项与注释），取差集；
2. 在当前树里找这些键的**全部引用点**（`*.sh` / `*.py` / `*.yaml` / `*.md` / `*.rst` / `*.json`），一行一处；
3. 按上下文分四档（判据写死在 classify() 里，不靠人眼）：
   - `EXEMPT_NEG_MARK`：同一行标了"已移除"（含 removed / migrated / deprecated 等否定词）；
   - `INVOKE`：同一行出现在 python/shell 命令位置（`python -m verl.`、`python xxx.py`、`bash xxx.sh`、`${ENGINE}` 等）；
   - `DOC_TABLE_CELL`：在 markdown/rst 表格行里（`| ... |`）；
   - `OTHER`：其余（注释、test 结构等）。

用法：
    python3 scan_shell_refs.py --repo <verl 仓库> [--json scan-result.json]
    python3 scan_shell_refs.py --repo <verl 仓库> --key trainer.use_legacy_worker_impl   # 只看一个键

退出码：0 = 扫描完成；2 = 参数/仓库问题。
**本脚本只做文本级定位；"这行传了键"不等于"该脚本现在会失败"**（见 readme「判据与边界」）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

REFACTOR = "044bbba2"
PARENT = f"{REFACTOR}^"
BOOT_CONFIGS = [
    "verl/trainer/config/ppo_trainer.yaml",
]
SCAN_SUFFIXES = {".sh", ".py", ".yaml", ".yml", ".md", ".rst", ".json"}
# 脚本的"入口身份"：同一行是否落在一次 hydra 启动里（python 解释器 + 主模块名）
ENTRY = re.compile(r"python3?\s+-m\s+([\w.]+)|python3?\s+(\S+\.py)|bash\s+(\S+\.sh)")
# 本行若是 "--config-name=xxx" / "--config-path=..."，入口取当前行解析出的值（覆盖行首入口）
CFG_NAME = re.compile(r"--config-name[=\s]+['\"]?([\w.]+)")
CFG_PATH = re.compile(r"--config-path[=\s]+['\"]?([\w./-]+)")
# 启动后多少行内仍算同一条命令（shell 续行 \ 展开后的 BODY 数组也算）
BLOCK_SPAN = 200
NAME_ONLY = re.compile(r"^\s*([A-Za-z_]\w*)=\(")
FALLBACK_ENTRY = "verl.experimental.fully_async_policy.fully_async_main"
# 行首若是脚本变量赋值（如 FULLY_ASYNC=( ... )），入口取解析出的变量值，而不是脚本本身
# 文档类后缀：只在表格行里才算"还在教用户怎么配"
DOC_SUFFIXES = {".md", ".rst"}
NEG_MARK = re.compile(r"removed|migrated|deprecated|not supported|no longer", re.I)
INVOKE = re.compile(r"(python3?\s+-m\s|python3?\s+\S+\.py|bash\s+\S+\.sh|\$\{ENGINE|\bverl\.trainer\.)")


def git(repo: pathlib.Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} 失败\n{out.stderr}")
    return out.stdout


def yaml_leaf_paths(text: str) -> set[str]:
    """缩进栈解析 YAML 的叶子键全路径（不支持 flow 集合，够用）。"""
    paths: set[str] = set()
    stack: list[tuple[int, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            continue
        if ":" not in stripped:
            continue
        indent = len(line) - len(line.lstrip())
        key = stripped.split(":")[0]
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parts = [k for _, k in stack] + [key]
        stack.append((indent, key))
        has_value = stripped.split(":", 1)[1].strip() not in ("", "|", ">")
        if has_value:
            paths.add(".".join(parts))
        else:
            paths.add(".".join(parts) + ".*")  # 空节点，允许有子键
    return paths


def removed_keys(repo: pathlib.Path) -> tuple[dict[str, set[str]], set[str]]:
    before: dict[str, set[str]] = {}
    after: dict[str, set[str]] = {}
    for cfg in BOOT_CONFIGS:
        before[cfg] = yaml_leaf_paths(git(repo, "show", f"{PARENT}:{cfg}"))
        after[cfg] = yaml_leaf_paths(git(repo, "show", f"{REFACTOR}:{cfg}"))
    gone: set[str] = set()
    for cfg in BOOT_CONFIGS:
        for key in before[cfg]:
            if key.endswith(".*"):
                continue
            if key not in after[cfg]:
                gone.add(key)
    return before, gone


def classify(suffix: str, line: str) -> str:
    if NEG_MARK.search(line):
        return "EXEMPT_NEG_MARK"
    if suffix in DOC_SUFFIXES:
        return "DOC_TABLE_CELL" if line.strip().startswith("|") else "OTHER"
    return "INVOKE"


def _launch_indices(lines: list[str]) -> list[int]:
    """所有"真启动行"的下标：含 python -m / python x.py / bash x.sh，且匹配不在引号内、不在注释里。"""
    out = []
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        for m in ENTRY.finditer(line):
            if not _inside_quotes(line, m.start()):
                out.append(idx)
                break
    return out


def entry_context(lines: list[str], lineno: int) -> str:
    """返回这一行"会以哪个 hydra 入口被消费"。两种落点：

    - `INVOKE`：落在一条启动命令（`python -m ... key=...`）附近，且在引号外；
    - `ARRAY`：命令行被拆进 `NAME=( … )` 数组，该数组又在某个启动行里被展开。

    这一步只做**定位**，不承诺"该脚本现在跑得起来"（模板路径、`${HOME}` 展开等
    都在本脚本之外，见 readme 边界）。
    """
    line = lines[lineno - 1]
    launches = _launch_indices(lines)
    if not launches:
        return "NO_ENTRY"
    near = [i for i in launches if abs(i - (lineno - 1)) <= BLOCK_SPAN]
    cands = near or launches
    idx = min(cands, key=lambda i: abs(i - (lineno - 1)))
    other = lines[idx]
    m = ENTRY.search(other)
    entry = m.group(1) or m.group(2) or m.group(3)
    return entry


def _inside_quotes(line: str, pos: int) -> bool:
    depth, quote = 0, None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if i == pos:
            return quote is not None or depth > 0
    return False


HYDRA_MAIN = re.compile(r"@hydra\.main\(config_path=[\"']([^\"']+)[\"'],\s*config_name=[\"']([^\"']+)[\"']")


def hydra_default(repo: pathlib.Path, entry: str) -> tuple[pathlib.Path, str] | None:
    """从入口模块的 `@hydra.main(...)` 读出默认配置目录与配置名（不猜）。"""
    mod = repo / (entry.replace(".", "/") + ".py")
    if not mod.is_file():
        return None
    m = HYDRA_MAIN.search(mod.read_text(encoding="utf-8"))
    if not m:
        return None
    cpath, cname = m.group(1), m.group(2)
    d = (mod.parent / cpath).resolve()
    return (d, cname) if (d / f"{cname}.yaml").is_file() else None


def _config_name_from_callers(repo: pathlib.Path, entry: str) -> tuple[pathlib.Path, str] | None:
    """入口靠 `--config-name` 选择配置时，从调用它的脚本里读出这个名字（取最常出现的那个）。

    形如 `python -m <entry> --config-path=config --config-name='fully_async_ppo_megatron_trainer.yaml'`。
    """
    stem = entry.rsplit(".", 1)[0]
    if not (repo / (stem.replace(".", "/") + ".py")).is_file():
        return None
    names = git(repo, "grep", "-h", "-e", "--config-name", "--", "*.sh").splitlines()
    cands: dict[str, int] = {}
    for line in names:
        m = CFG_NAME.search(line)
        if m:
            n = m.group(1).removesuffix(".yaml")
            cands[n] = cands.get(n, 0) + 1
    if not cands:
        return None
    name = max(cands, key=lambda k: cands[k])
    for d in {repo / (stem.replace(".", "/")), repo / "verl/trainer/config"}:
        cfgdir = d / "config" if d.name != "config" else d
        for base in (cfgdir, d):
            if (base / f"{name}.yaml").is_file():
                return base, name
    return None


def config_chain(repo: pathlib.Path, entry: str | None, log: list[str]) -> dict | None:
    """沿入口的默认 hydra 配置展开 `defaults:`，返回这条链的形状。

    返回 {"files": [...], "root_keys": set(根键), "has_record_root": bool}。
    `has_record_root` 表示链里出现过 `ppo_trainer.yaml`——onfig 由它构成，
    键空间因此是 **struct 模式**（顶层 unprefixed override 会被 Hydra 拒）。

    返回 None = 链条没接上（不是 python -m、读不到 `@hydra.main`、有包含项缺文件），
    调用方必须标 UNKNOWN，**不能**当成"没有该键"。
    """
    if not entry or entry in ("NO_ENTRY",) or entry.endswith((".py", ".sh")):
        log.append(f"{entry}: runner 不是 python -m 模块，配置链不解析")
        return None
    default = hydra_default(repo, entry) or _config_name_from_callers(repo, entry)
    if default is None:
        log.append(f"{entry}: 读不到 @hydra.main 的 config_path/config_name")
        return None
    d0, name = default
    root_keys: set[str] = set()
    files: list[str] = []
    has_record = False
    queue = [(d0, name)]
    while queue:
        d, nm = queue.pop()
        f = d / f"{nm}.yaml"
        if not f.is_file():
            log.append(f"配置链缺文件：{f.relative_to(repo)}")
            return None
        text = f.read_text(encoding="utf-8")
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("- ") and not s.split("- ", 1)[1].startswith("_") and not any(c in s for c in "{}@:"):
                # 子目录里的配置用 Hydra searchpath 找到（fully_async 的 searchpath 指向 verl/trainer/config）
                sub = d / f"{s[2:].strip()}.yaml"
                target = d if sub.is_file() else repo / "verl/trainer/config"
                queue.append((target, s[2:].strip()))
            m = re.match(r"^([A-Za-z_][\w]*):", line)
            if m:
                root_keys.add(m.group(1))
        rel = str(f.relative_to(repo))
        if rel not in files:
            files.append(rel)
        if rel == "verl/trainer/config/ppo_trainer.yaml":
            has_record = True
    return {"files": files, "root_keys": root_keys, "has_record_root": has_record}


def scan_refs(repo: pathlib.Path, key: str) -> list[dict]:
    hits: list[dict] = []
    globs = ["*.sh", "*.py", "*.yaml", "*.yml", "*.md", "*.rst", "*.json"]
    files = git(repo, "grep", "-l", "-e", key, "--", *globs)
    for rel in sorted({f for f in files.splitlines() if f}):
        path = repo / rel
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lines = text.splitlines()
        for lineno, line in enumerate(lines, 1):
            if key not in line:
                continue
            hits.append(
                {
                    "file": rel,
                    "line": lineno,
                    "kind": classify(path.suffix, line),
                    "pkg_entry": entry_context(lines, lineno) if path.suffix in {".sh", ".py"} else None,
                    "text": line.strip()[:200],
                }
            )
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--json")
    ap.add_argument("--key")
    args = ap.parse_args()
    repo = pathlib.Path(args.repo).resolve()

    _, gone = removed_keys(repo)
    keys = [args.key] if args.key else sorted(gone)
    if args.key and args.key not in gone:
        raise SystemExit(f"{args.key} 不在 #6067 的删除集里：{sorted(gone)}")

    report = {"refactor": REFACTOR, "boot_configs": BOOT_CONFIGS, "removed_keys": sorted(gone), "keys": {}}
    log: list[str] = []
    for key in keys:
        hits = scan_refs(repo, key)
        chains: dict[str, list[str] | None] = {}
        for h in hits:
            if h["kind"] not in ("INVOKE",):
                continue
            entry = h["pkg_entry"]
            if entry not in chains:
                chains[entry] = config_chain(repo, entry, log)
            ch = chains[entry]
            h["config_chain"] = {**ch, "root_keys": sorted(ch["root_keys"])} if ch else None
            root = key.split(".", 1)[0]
            prefix = "+" in h["text"]
            if ch is None:
                h["consumed_by"] = "UNKNOWN"
            elif root not in ch["root_keys"]:
                h["consumed_by"] = "UNKNOWN"  # 入口记的不是这个根键，链条可能接错
            elif ch["has_record_root"] and not prefix:
                h["consumed_by"] = "COMPOSE_FAIL"  # open dict 下 unprefixed override 会被 Hydra 拒
            elif ch["has_record_root"] and prefix:
                h["consumed_by"] = "SILENT_OK"  # 加了 + 就真能传进去，不报错
            else:
                h["consumed_by"] = "FREE"
        by_kind: dict[str, int] = {}
        for h in hits:
            k = h["kind"] + ("/" + h["consumed_by"] if "consumed_by" in h else "")
            by_kind[k] = by_kind.get(k, 0) + 1
        report["keys"][key] = {"n_hits": len(hits), "by_kind": by_kind, "hits": hits}

    print(f"#6067 ({REFACTOR}) 删除的 boot 配置键：{sorted(gone) or '（无）'}")
    if log:
        print("解析日志：")
        for line in dict.fromkeys(log):
            print(f"  - {line}")
    for key, info in report["keys"].items():
        print(f"\n== {key}: {info['n_hits']} 处  {info['by_kind']}")
        for h in info["hits"]:
            ch = h.get("config_chain") or {}
            tail = "  chain=" + ",".join(x.split("/config/")[-1] for x in ch.get("files", [])) if ch else ""
            print(f"  [{h['kind']:<16}{('/' + h['consumed_by']) if 'consumed_by' in h else '':<14}] "
                  f"{h['file']}:{h['line']}{tail}\n      {h['text']}")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写出 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
