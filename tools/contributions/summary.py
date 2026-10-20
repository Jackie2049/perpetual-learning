#!/usr/bin/env python3
"""Parse the shared Markdown contribution header; validate working tree or Git index."""
import argparse
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('仓库', '贡献点', '类型', '价值评估', '进展', '大白话讲明白', 'What–Why–How')
# Owner boundary: these existing M5 documents await explicit header-only authorization.
# New AReaL files are still required to follow WRITING.md. Never silently exempt a folder.
LEGACY = frozenset({
    'open-source/areal/readme.md',
    'open-source/areal/tracker.md',
    'open-source/areal/handoff-m5.md',
})
START = '<!-- contribution-summary:start -->'
END = '<!-- contribution-summary:end -->'


def parse_summary(source):
    starts = list(re.finditer(r'^' + re.escape(START) + r'\s*$', source, re.M))
    ends = list(re.finditer(r'^' + re.escape(END) + r'\s*$', source, re.M))
    if len(starts) != 1 or len(ends) != 1 or starts[0].end() >= ends[0].start():
        raise ValueError('需要且只能有一组 contribution-summary 起止标记')
    if not re.fullmatch(r'# [^\n]+\s*', source[:starts[0].start()]):
        raise ValueError('抬头必须紧跟唯一的一级标题，位于正文之前')
    block = source[starts[0].end():ends[0].start()].strip()
    matches = list(re.finditer(r'^\*\*【([^】]+)】\*\*', block, re.M))
    if tuple(m[1] for m in matches) != FIELDS:
        raise ValueError('字段及顺序必须是：' + '、'.join(FIELDS))
    values = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        value = block[match.end():end].strip()
        if not value or value in ('…', '...', '待填写'):
            raise ValueError(f'{match[1]} 不得留空或使用占位符')
        values[match[1]] = value
    for field in ('仓库', '贡献点'):
        if not re.search(r'\[[^\]]+\]\((?:https://|\.{0,2}/|[\w-]+[./])[^)]+\)', values[field]):
            raise ValueError(f'{field} 必须提供可点击的项目或证据入口')
    if not re.match(r'^(?:10|[0-9]) / 10 — \S', values['价值评估']):
        raise ValueError('价值评估使用“分数 / 10 — 一句理由”，分数为 0–10')
    if len(values['大白话讲明白']) < 40:
        raise ValueError('大白话说明过短，应独立说明背景、问题和影响')
    lines = values['What–Why–How'].splitlines()
    points = [re.fullmatch(r'- \*\*(What|Why|How)：\*\* (.+)', line) for line in lines if line.strip()]
    if len(points) != 3 or any(p is None for p in points) or [p[1] for p in points] != ['What', 'Why', 'How']:
        raise ValueError('What–Why–How 仅用三点：- **What：**、- **Why：**、- **How：**')
    values['what'], values['why'], values['how'] = (p[2] for p in points)
    return values


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true', help='仅检查暂存区修改的 Markdown')
    args = parser.parse_args()
    paths = (git('diff', '--cached', '--name-only', '--diff-filter=ACMR', '--', 'open-source')
             if args.staged else git('ls-files', '--', 'open-source')).splitlines()
    errors, count, skipped = [], 0, []
    for path in paths:
        if not path.endswith('.md'):
            continue
        source = git('show', ':' + path) if args.staged else (ROOT / path).read_text()
        if path in LEGACY and START not in source:
            skipped.append(path)
            continue
        try:
            parse_summary(source)
            count += 1
        except ValueError as error:
            errors.append(f'{path}: {error}')
    for error in errors:
        print(error)
    if skipped:
        print('保留原样（M5 文档，抬头整改待授权）：' + '、'.join(skipped))
    if errors:
        raise SystemExit(1)
    print(f'开源文档抬头检查通过：{count} 份')


if __name__ == '__main__':
    main()
