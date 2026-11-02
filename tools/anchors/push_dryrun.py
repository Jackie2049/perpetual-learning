#!/usr/bin/env python3
"""推送前预演：这次推会**删掉**远端哪些文件？

为什么需要它：`tools/git-push-api/push_via_api.py` 在 base 不在本地时，
会把「远端有、HEAD 无」的文件判为删除（`changed[path] = "D"`）。
2026-11-03 因此误删过并行会话的 7 个实验文件——它们已下载到工作区，
但**未提交进 HEAD**，于是被当作不存在。

用法：
    python3 tools/anchors/push_dryrun.py            # 用远端当前 main 作 base
    python3 tools/anchors/push_dryrun.py <base_sha>

退出码：0 = 安全（不删任何文件）；1 = 会删除文件，**不要推**。

判据永远是 **HEAD**，不是工作区——工作区有文件不等于它会被推上去。
"""

import json
import subprocess
import sys
import urllib.error
import urllib.request

REPO = 'Jackie2049/perpetual-learning'
API = 'https://api.github.com/repos/' + REPO


def _get(url):
    try:
        return json.load(urllib.request.urlopen(url))
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print('GitHub API 速率限制（403）。未认证请求每小时 60 次。')
            print('等一会再试，或为 urllib 加 token（见 tools/git-push-api）。')
        elif e.code == 404:
            print(f'404: {url}')
        else:
            print(f'HTTP {e.code}: {url}')
        raise SystemExit(2)
    except urllib.error.URLError as e:
        print(f'网络不可达: {e.reason}')
        raise SystemExit(2)


def remote_main():
    return _get(API + '/git/ref/heads/main')['object']['sha']


def head_blobs():
    out = subprocess.run(['git', 'ls-tree', '-r', 'HEAD'],
                         capture_output=True, text=True, check=True).stdout
    return {line.split('\t', 1)[1] for line in out.splitlines()}


def remote_blobs(sha):
    d = _get(f'{API}/git/trees/{sha}?recursive=1')
    return {e['path'] for e in d.get('tree', []) if e['type'] == 'blob'}


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else remote_main()
    remote = remote_blobs(base)
    head = head_blobs()

    would_delete = sorted(p for p in remote if p not in head)
    would_add = sorted(p for p in head if p not in remote)

    print(f'base = {base[:10]}')
    print(f'远端文件 {len(remote)} / 本地 HEAD {len(head)}')
    print(f'将新增 {len(would_add)} 个')
    for p in would_add[:8]:
        print(f'   + {p}')
    if len(would_add) > 8:
        print(f'   … 另有 {len(would_add) - 8} 个')

    if would_delete:
        print(f'\n⚠️  将删除 {len(would_delete)} 个文件：')
        for p in would_delete:
            print(f'   - {p}')
        print('\n**不要推。** 先把这些文件补进 HEAD（git add + commit），'
              '或确认它们确实该删。')
        return 1

    print('\n安全：不会删除任何远端文件。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
