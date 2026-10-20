#!/usr/bin/env python3
"""Build a read-only Pages snapshot from committed repository documents."""
from datetime import datetime
from pathlib import Path
import hashlib
import html
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


def render_math(markup):
    # Sanitize repository HTML first; only trusted KaTeX output is added afterwards.
    pattern = re.compile(r'<(span|div) class="arithmatex">(.*?)</\1>', re.S)
    matches = list(pattern.finditer(markup))
    if not matches:
        return markup
    formulas = []
    for match in matches:
        source = html.unescape(match[2]).strip()
        display = match[1] == 'div'
        opening, closing = (r'\[', r'\]') if display else (r'\(', r'\)')
        if not source.startswith(opening) or not source.endswith(closing):
            raise ValueError('Invalid math delimiters')
        formulas.append({'source': source[2:-2], 'display': display})
    result = subprocess.run(
        ['node', str(WEB / 'render-math.cjs')], input=json.dumps(formulas),
        text=True, capture_output=True, check=True,
    )
    rendered = iter(json.loads(result.stdout))
    return pattern.sub(lambda m: f'<{m[1]} class="arithmatex">{next(rendered)}</{m[1]}>', markup)


def asset_url(name):
    digest = hashlib.sha256((WEB / name).read_bytes()).hexdigest()[:16]
    return f'./{name}?v={digest}'


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
    cleaned = bleach.clean(rendered, tags=TAGS, attributes={'a': ['href', 'title'], '*': ['id'], 'code': ['class'], 'span': ['class'], 'div': ['class']}, protocols=['https', 'http', 'mailto'], strip=True)
    return render_math(cleaned)


def section_source(source, prefix):
    """Select a heading and its nested content, never neighbouring sections."""
    headings = list(re.finditer(r'^(#{2,6})\s+(.+)$', source, re.M))
    for i, match in enumerate(headings):
        if clean_text(match[2]).startswith(prefix):
            end = next((h.start() for h in headings[i + 1:] if len(h[1]) <= len(match[1])), len(source))
            return source[match.end():end].strip()
    return ''


def plain_html(markup):
    return html.unescape(bleach.clean(markup, tags=[], strip=True)).strip()


def progress_items(markup):
    """Keep source evidence intact while presenting tracker tables as readable cards."""
    items = []
    for table in re.findall(r'<table>(.*?)</table>', markup, re.S):
        headers = re.findall(r'<th[^>]*>(.*?)</th>', table, re.S)
        names = [plain_html(h) for h in headers]
        if '候选' not in names or '状态' not in names:
            continue
        for row in re.findall(r'<tr>(.*?)</tr>', table, re.S):
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)
            if len(cells) != len(names):
                continue
            fields = dict(zip(names, cells))
            content = fields['状态']
            if plain_html(fields.get('PR', '')) not in ('', '—', '-'):
                content += '<p>PR：' + fields['PR'] + '</p>'
            items.append({'title': plain_html(fields['候选']), 'html': content,
                          'summary': plain_html(fields['状态'])})
    if not items:
        # Progress sections in readmes/handoffs use lists instead of tracker tables.
        for item in re.findall(r'<li>(.*?)</li>', markup, re.S):
            strong = re.search(r'<strong>(.*?)</strong>', item, re.S)
            title = plain_html(strong[1]) if strong else plain_html(item).split('。')[0]
            items.append({'title': title, 'html': item, 'summary': plain_html(item)})
    return items


def contribution_projects(paths, docs, catalog, revision):
    # Only source selection lives here; progress comes from committed project records.
    sources = {
        'verl': ('tracker.md', '当前活跃'),
        'areal': ('handoff-m5.md', '阶段状态'),
        'vllm': ('readme.md', '新核验'),
        'moonshot': ('tracker.md', '当前活跃'),
    }
    preferred = ['verl', 'areal', 'vllm']
    readmes = sorted(p for p in paths if re.fullmatch(r'open-source/[^/]+/[Rr][Ee][Aa][Dd][Mm][Ee]\.md', p))
    readmes.sort(key=lambda p: (preferred.index(p.split('/')[1]) if p.split('/')[1] in preferred else len(preferred), p))
    projects, references = [], {}
    for readme in readmes:
        slug = readme.split('/')[1]
        def reference(path):
            if path not in docs:
                docs[path] = git('show', f'{revision}:{path}')
            title_match = re.search(r'^#\s+(.+)', docs[path], re.M)
            ref = {'id': doc_id(path), 'path': path,
                   'title': clean_text(title_match[1]) if title_match else Path(path).stem,
                   'url': github_url(path, revision)}
            references[path] = ref
            return ref
        overview = reference(readme)
        name = re.sub(r'\s*(开源贡献|贡献轨道).*$', '', overview['title'])
        filename, section = sources.get(slug, (Path(readme).name, ''))
        path = f'open-source/{slug}/{filename}'
        source = reference(path)
        excerpt = section_source(docs[path], section) if section else ''
        if section and not excerpt:
            raise RuntimeError(f'Missing contribution progress section: {path} / {section}')
        tracker_path = f'open-source/{slug}/tracker.md'
        tracker = reference(tracker_path) if tracker_path in paths else None
        materials = [d for d in catalog if d['path'].startswith(f'open-source/{slug}/')]
        projects.append({'slug': slug, 'name': name, 'overview': overview, 'source': source,
                         'tracker': tracker, 'section': section,
                         'updated': git('log', '-1', '--format=%cI', revision, '--', path).strip(),
                         'items': progress_items(render(excerpt, path, revision)) if excerpt else [],
                         'materials': materials})
    return projects, list(references.values())


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
    projects, project_refs = contribution_projects(paths, docs, catalog, revision)
    selected = {r['path'] for r in project_refs} | {f['path'] for f in featured} | {p['path'] for p in pending} | {latest}
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
    # Standalone research report (self-contained single page, not part of the workbench).
    for report in ['report_deepinfer.html']:
        if (WEB / report).exists():
            shutil.copyfile(WEB / report, OUT / report)
    index = (OUT / 'index.html').read_text()
    for name in ['app.js', 'styles.css', 'vendor/katex/katex.min.css']:
        index = index.replace(f'./{name}', asset_url(name))
    (OUT / 'index.html').write_text(index)
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
    data = {'revision': revision, 'built': datetime.now(TZ).isoformat(), 'sourceDate': git('log', '-1', '--format=%cI').strip(), 'repository': REPO, 'counts': counts, 'featured': featured, 'pending': pending, 'projects': projects, 'projectRefs': project_refs, 'diary': {'path': latest, 'id': doc_id(latest), 'date': Path(latest).stem, 'review': review, 'events': list(reversed(events))}, 'library': catalog}
    (OUT / 'data.json').write_text(json.dumps(data, ensure_ascii=False))
    print(f'Built {len(catalog)} document titles, {len(selected)} reading pages, {len(events)} diary entries; snapshot {revision[:7]}')


if __name__ == '__main__':
    main()
