#!/usr/bin/env python3
"""Append work records and check the Git index before committing. Python 3.9+."""
import argparse
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import posixpath
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

TZ = ZoneInfo('Asia/Singapore')
ENTRY = re.compile(r'^## (\d{2}:\d{2}) — ([^\n]+)\n(.*?)(?=^## |\Z)', re.M | re.S)
DAY = re.compile(r'diary/(\d{4}-\d{2}-\d{2})\.md$')
HOOKS = 'tools/diary/hooks'


def git(*args, optional=False):
    result = subprocess.run(['git', *args], text=True, capture_output=True)
    if result.returncode and not optional:
        raise ValueError(result.stderr.strip())
    return result.stdout if result.returncode == 0 else ''


def entries(text):
    result = {}
    for match in ENTRY.finditer(text):
        key = (match[1], match[2])
        if key in result:
            raise ValueError('日志中有重复的时间与标题，请合并同一工作或明确区分标题。')
        result[key] = match[3].strip()
    return result


def without_fenced_code(text):
    prose, fence = [], None
    for line in text.splitlines(keepends=True):
        marker = re.fullmatch(r' {0,3}(`{3,}|~{3,})(.*)', line.rstrip('\r\n'))
        if fence is not None:
            if (marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1]
                    and not marker[2].strip()):
                fence = None
            continue
        if marker and (marker[1][0] == '~' or '`' not in marker[2]):
            fence = (marker[1][0], len(marker[1]))
        else:
            prose.append(line)
    return ''.join(prose)


def append(title, body):
    title, body = title.strip(), body.strip()
    if not title or '\n' in title or not body or re.search(r'^## ', body, re.M):
        raise ValueError('需要单行标题和非空正文；正文使用段落或三级标题。')
    now = datetime.now(TZ)
    day, time = now.strftime('%Y-%m-%d'), now.strftime('%H:%M')
    path = Path(f'diary/{day}.md')
    text = path.read_text() if path.exists() else f'# {day} 工作日志\n'
    if (time, title) in entries(text):
        raise ValueError('这一时间与标题已存在，请更新原记录，避免重复追加。')
    index = Path('diary/readme.md')
    overview = index.read_text()
    link = f'[{day}]({day}.md)'
    if link not in overview:
        marker = '## 已有日志\n'
        if marker not in overview:
            raise ValueError('diary/readme.md 缺少“已有日志”入口，请先修复索引。')
        overview = overview.replace(marker, f'{marker}\n- {link}\n', 1)
    path.write_text(text.rstrip() + f'\n\n## {time} — {title}\n\n{body}\n')
    index.write_text(overview)
    print(f'已追加 {path}（{time} +08:00）。核对正文和当日回看后，与本项成果一起暂存。')


def check(now=None):
    now = now or datetime.now(TZ)
    # Treat a renamed day as removal + addition so the source history is still checked.
    changed = git('diff', '--cached', '--no-renames', '--name-only', '-z').split('\0')
    if not any(changed):
        return
    tracked = set(git('ls-files', '-z').split('\0'))
    overview = git('show', ':diary/readme.md', optional=True)
    recent = []
    for name in filter(None, changed):
        day_match = DAY.fullmatch(name)
        if not day_match:
            continue
        old = entries(git('show', f'HEAD:{name}', optional=True))
        new = entries(git('show', f':{name}', optional=True))
        # Reheading is allowed only when the original body survives, one-to-one.
        # A fresh work record must not excuse replacing the rest of the day's history.
        reheaded = Counter(body for key, body in new.items() if key not in old)
        for key, body in old.items():
            if key not in new:
                if reheaded[body] == 0:
                    raise ValueError(f'{name} 移除已有日志：{key[0]} — {key[1]}。请保留原记录并追加纠正；需要迁移/删除时先由用户决定。')
                reheaded[body] -= 1
            elif body and not new[key]:
                raise ValueError(f'{name} 清空已有日志：{key[0]} — {key[1]}。请保留原记录并追加纠正。')
        for (time, title), body in new.items():
            if body == old.get((time, title)) or body in old.values():
                continue
            stamp = datetime.strptime(f'{day_match[1]} {time}', '%Y-%m-%d %H:%M').replace(tzinfo=TZ)
            if stamp > now:
                raise ValueError(f'{name} {time} 是未来时间，请使用系统时间。')
            # Yesterday is allowed for work staged across midnight, without relabeling its time.
            if stamp.date() < now.date() - timedelta(days=1) or not body:
                continue
            if f']({day_match[1]}.md)' not in overview:
                raise ValueError(f'{name} 尚未进入已暂存的 diary/readme.md 索引。')
            for target in re.findall(r'\]\(([^\s)]+)\)', without_fenced_code(body)):
                url = urlsplit(target)
                if url.scheme or url.netloc or not url.path:
                    continue
                path = posixpath.normpath(posixpath.join('diary', unquote(url.path)))
                if path.startswith('/') or path == '..' or path.startswith('../'):
                    raise ValueError(f'日志链接应指向仓库内成果：{target}')
                if path not in tracked and not any(p.startswith(path + '/') for p in tracked):
                    raise ValueError(f'日志链接目标未纳入本次提交或已有版本：{target}')
            recent.append(f'{name} {time} — {title}')
    if not recent:
        raise ValueError('本次提交缺少对应日志：请追加工作记录，或补充同一项工作的近期记录，并一起 git add。仅修改索引、回看或旧日志不能代替本项工作记录。')
    print('日志检查通过：' + '; '.join(recent))


def install():
    configured = git('config', '--get', 'core.hooksPath', optional=True).strip()
    if configured == HOOKS:
        print('日志提交检查已启用。')
        return
    if configured:
        raise ValueError(f'已有 core.hooksPath={configured}，未覆盖；请在现有 pre-commit 中接入本工具 check。')
    existing = Path(git('rev-parse', '--git-path', 'hooks/pre-commit').strip())
    if existing.exists() or existing.is_symlink():
        raise ValueError('已有 pre-commit，未覆盖；请在现有 hook 中接入本工具 check。')
    git('config', '--local', 'core.hooksPath', HOOKS)
    print('已为当前克隆启用日志提交检查。其他克隆需各执行一次 install。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    add = commands.add_parser('append', help='正文从标准输入或 --body-file 读取，自动取时并维护日期索引')
    add.add_argument('--title', required=True)
    add.add_argument('--body-file', type=Path)
    commands.add_parser('check', help='只检查暂存区，不自动暂存或修改文件')
    commands.add_parser('install', help='为当前克隆启用 pre-commit；已有 hooks 时拒绝覆盖')
    args = parser.parse_args()
    try:
        root = Path(git('rev-parse', '--show-toplevel').strip())
        if Path.cwd().resolve() != root.resolve():
            raise ValueError('请在项目根目录执行。')
        if args.command == 'append':
            append(args.title, args.body_file.read_text() if args.body_file else sys.stdin.read())
        elif args.command == 'check':
            check()
        else:
            install()
    except (ValueError, OSError) as exc:
        print(f'日志收尾检查：{exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
