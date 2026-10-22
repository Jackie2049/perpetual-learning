"""五续：从 PDL 源文件生成 nosync 变体——删除 T.pdl_sync() 行、T.pdl_trigger() 替换为 pass。
依赖型 pdl_sync 在顺序 launch 下由 stream 顺序天然保证，正确改写是删除而非替换 sync_grid
（sync_grid 引入 cooperative-launch grid 上限，四续已实测 4090 大 grid 启动失败）。

用法: python3 gen_nosync_variants.py <src_dir> <out_dir>
"""
import ast
import re
import sys
from pathlib import Path

def resolve_path(arg_index, default_path):
    """Resolves a path from sys.argv, ensuring it's within a safe base directory."""
    if len(sys.argv) > arg_index:
        # Sanitize input to prevent path traversal
        user_input = sys.argv[arg_index]
        if ".." in user_input or user_input.startswith("/"):
            raise ValueError(f"Invalid path input: {user_input}. Path traversal is not allowed.")
        return Path(__file__).parent / user_input
    return default_path

SRC = resolve_path(1, Path(__file__).parent / "pdl-sources")
OUT = resolve_path(2, Path(__file__).parent / "pdlfree-variants-nosync")

OUT.mkdir(parents=True, exist_ok=True)

SYNC_RE = re.compile(r"^\s*T\.pdl_sync\(\)\s*$")
TRIG_RE = re.compile(r"^\s*T\.pdl_trigger\(\)\s*$")


def rewrite(lines):
    """删除 pdl_sync 行；pdl_trigger 行替换为 pass。删除后若产生空块（上一行以 ':' 结尾）补 pass。"""
    out = []
    for i, line in enumerate(lines):
        if SYNC_RE.match(line):
            indent = len(line) - len(line.lstrip())
            prev = lines[i - 1] if i > 0 else ""
            # 若上一行是块头（以 : 结尾）且缩进更小，则本行是块内唯一语句 → 补 pass 保持合法
            if prev.rstrip().endswith(":") and (len(prev) - len(prev.lstrip())) < indent:
                out.append(" " * indent + "pass  # PDL removed (nosync)\n")
            continue  # 否则直接删除
        if TRIG_RE.match(line):
            indent = len(line) - len(line.lstrip())
            out.append(" " * indent + "pass  # PDL removed\n")
            continue
        out.append(line)
    return out


def main():
    changed = 0
    for src in sorted(SRC.glob("*.py")):
        lines = src.read_text().splitlines(keepends=True)
        new = rewrite(lines)
        try:
            ast.parse("".join(new))
        except SyntaxError as e:
            print(f"[SYNTAX] {src.name}: {e}")
            continue
        residue = [l for l in new if re.search(r"T\.(sync_grid|pdl_sync|pdl_trigger)\(", l)]
        out_path = OUT / src.name.replace(".py", "_nosync.py")
        out_path.write_text("".join(new))
        if new != lines:
            changed += 1
        status = "OK" if not residue else f"RESIDUE {len(residue)}"
        # 统计替换
        n_sync = sum(1 for l in lines if SYNC_RE.match(l))
        n_trig = sum(1 for l in lines if TRIG_RE.match(l))
        print(f"[{status}] {src.name}: {n_sync} pdl_sync, {n_trig} pdl_trigger -> {out_path.name}")
    print(f"\n===== {changed} files changed =====")


if __name__ == "__main__":
    main()