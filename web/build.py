#!/usr/bin/env python3
"""Build a read-only Pages snapshot from committed repository documents."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import posixpath
import re
import shutil
import subprocess
from urllib.parse import quote, urlsplit
from zoneinfo import ZoneInfo

import bleach
import markdown

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web'
OUT = WEB / 'dist'
REPO = 'https://github.com/Jackie2049/perpetual-learning'
TZ = ZoneInfo('Asia/Singapore')
TAGS = {'p', 'br', 'hr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'em', 'del', 'blockquote', 'ul', 'ol', 'li', 'pre', 'code', 'a', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'span', 'div'}


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True)


def clean_text(text):
    text = re.sub(r'!?\[([^\]]+)\]\([^)]+\)', r'\1', text)
    return re.sub(r'[*`>~]', '', text).strip()


def doc_id(path):
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def github_url(path, revision):
    return f'{REPO}/blob/{revision}/{quote(path)}'


def render(source, path, revision):
    # Repository notes contain math cardinalities such as |o_i| inside bold table cells.
    # Escape inline delimiters before the table parser can treat them as extra columns.
    lines = []
    for line in source.splitlines():
        if line.lstrip().startswith('|'):
            line = re.sub(r'(`[^`]+`|\*\*[^\n]+?\*\*|(?<!\\)\$(?:\\.|[^$])+?\$)', lambda m: re.sub(r'(?<!\\)\|', r'\\|', m[0]), line)
        lines.append(line)
    source = '\n'.join(lines)
    rendered = markdown.markdown(source, extensions=['fenced_code', 'tables', 'sane_lists', 'toc', 'pymdownx.arithmatex'], extension_configs={'pymdownx.arithmatex': {'generic': True}})
    def rewrite(match):
        url = match[1]
        parsed = urlsplit(url)
        if not parsed.scheme and not parsed.netloc and not url.startswith('#'):
            target = posixpath.normpath(posixpath.join(posixpath.dirname(path), parsed.path))
            url = github_url(target, revision) + ('#' + parsed.fragment if parsed.fragment else '')
        return f'href="{url}"'
    rendered = re.sub(r'href="([^"]*)"', rewrite, rendered)
    return bleach.clean(rendered, tags=TAGS, attributes={'a': ['href', 'title'], '*': ['id'], 'code': ['class'], 'span': ['class'], 'div': ['class']}, protocols=['https', 'http', 'mailto'], strip=True)


def main():
    revision = git('rev-parse', 'HEAD').strip()
    paths = git('ls-tree', '-r', '--name-only', revision).splitlines()
    docs = {}
    catalog = []
    categories = {'notebook': '知识笔记', 'open-source': '开源贡献', 'diary': '工作日志'}
    for path in paths:
        if not path.endswith('.md') or path.split('/')[0] not in categories:
            continue
        if Path(path).name.lower() == 'readme.md':
            continue
        source = git('show', f'{revision}:{path}')
        docs[path] = source
        heading = re.search(r'^#\s+(.+)', source, re.M)
        title = clean_text(heading[1]) if heading else Path(path).stem
        headings = ' · '.join(clean_text(h) for h in re.findall(r'^#{2,3}\s+(.+)', source, re.M)[:6])
        catalog.append({'id': doc_id(path), 'path': path, 'title': title, 'category': categories[path.split('/')[0]], 'outline': headings[:350], 'url': github_url(path, revision)})
    dates = sorted(p for p in docs if re.fullmatch(r'diary/\d{4}-\d{2}-\d{2}\.md', p))
    if not dates:
        raise RuntimeError('No committed diary entries found')
    latest = dates[-1]
    featured = json.loads((WEB / 'featured.json').read_text())
    pending_paths = [
        ('open-source/verl/candidate-8237-offload-fraction.md', '优化器卸载修复', 'verl · #8237'),
        ('open-source/verl/candidate-8250-async-entropy-collapse.md', '异步训练熵坍塌诊断', 'verl · #8250'),
        ('open-source/verl/verl-clip-semantics-rfc-draft.md', 'clip 语义文档澄清', 'verl · 文档提案'),
        ('open-source/vllm/vllm-39096-batch-invariance-comment-draft.md', '批次不变性增量评论', 'vLLM · #39096'),
    ]
    pending = []
    for path, title, tag in pending_paths:
        source = docs.get(path, '')
        status_lines = re.findall(r'^>\s*(?:状态|Status|发布形态)[：:]\s*(.*)', source, re.M)
        status = clean_text(status_lines[0]) if status_lines else ''
        if not status or 'POSTED' in status or '评论已发布' in status:
            continue
        if not re.search(r'待|需授权|需用户授权|授权|DRAFT', status):
            continue
        pending.append({'id': doc_id(path), 'path': path, 'title': title, 'tag': tag, 'status': status, 'label': '待上机验证' if '待 GPU' in status else '待发布决定'})
    selected = {f['path'] for f in featured} | {p['path'] for p in pending} | {latest}
    # Only explicitly selected reading pages are copied. Other historical files are title-indexed and linked to GitHub.
    for path in selected:
        if path not in paths:
            raise RuntimeError(f'Selected document is not committed: {path}')
        if path not in docs:
            docs[path] = git('show', f'{revision}:{path}')
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'content').mkdir(parents=True)
    for name in ['index.html', 'styles.css', 'app.js', 'favicon.svg']:
        shutil.copyfile(WEB / name, OUT / name)
    shutil.copytree(WEB / 'vendor' / 'katex', OUT / 'vendor' / 'katex')
    (OUT / '.nojekyll').touch()
    for path in selected:
        date = git('log', '-1', '--format=%cI', revision, '--', path).strip()
        document = {'path': path, 'updated': date, 'url': github_url(path, revision), 'html': render(docs[path], path, revision)}
        (OUT / 'content' / (doc_id(path) + '.json')).write_text(json.dumps(document, ensure_ascii=False))
    for item in catalog:
        item['readable'] = item['path'] in selected
    for item in featured:
        item['id'] = doc_id(item['path'])
    diary_source = docs[latest]
    section = re.search(r'^## 当日回看[^\n]*\n(.*?)(?=^## |\Z)', diary_source, re.M | re.S)
    review = render(section[1].strip(), latest, revision) if section else '<p>当日尚未整理回看摘要，请阅读完整日志。</p>'
    events = [{'time': m[1], 'title': clean_text(m[2]), 'summary': clean_text(m[3].strip().split('\n\n')[0])[:170]} for m in re.finditer(r'^## (\d\d:\d\d) — ([^\n]+)\n(.*?)(?=^## |\Z)', diary_source, re.M | re.S)]
    counts = {category: sum(d['category'] == category for d in catalog) for category in categories.values()}
    data = {'revision': revision, 'built': datetime.now(TZ).isoformat(), 'sourceDate': git('log', '-1', '--format=%cI').strip(), 'repository': REPO, 'counts': counts, 'featured': featured, 'pending': pending, 'diary': {'path': latest, 'id': doc_id(latest), 'date': Path(latest).stem, 'review': review, 'events': list(reversed(events))}, 'library': catalog}
    (OUT / 'data.json').write_text(json.dumps(data, ensure_ascii=False))
    print(f'Built {len(catalog)} document titles, {len(selected)} reading pages, {len(events)} diary entries; snapshot {revision[:7]}')


if __name__ == '__main__':
    main()
