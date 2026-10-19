#!/usr/bin/env python3
"""Build committed main and publish its static snapshot to gh-pages without force-push."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import build

REMOTE = 'https://github.com/Jackie2049/perpetual-learning.git'
MARKER = '.perpetual-learning-pages'


def run(*args, cwd=build.ROOT, capture=True):
    return subprocess.check_output(list(args), cwd=cwd, text=True).strip() if capture else subprocess.run(args, cwd=cwd, check=True)


def main():
    origin = build.git('remote', 'get-url', 'origin').strip()
    if origin != REMOTE:
        raise RuntimeError('Origin differs from the configured project; publication stopped.')
    if build.git('status', '--porcelain', '--', 'web').strip():
        raise RuntimeError('Commit and push web source changes before publishing.')
    revision = build.git('rev-parse', 'HEAD').strip()
    published_main = run('git', 'ls-remote', origin, 'refs/heads/main').split()[0]
    if revision != published_main:
        raise RuntimeError('HEAD must match remote main so snapshot source links resolve.')
    build.main()
    remote_branch = run('git', 'ls-remote', origin, 'refs/heads/gh-pages')
    with tempfile.TemporaryDirectory(prefix='perpetual-pages-') as folder:
        temp = Path(folder)
        run('git', 'init', '-q', cwd=temp)
        run('git', 'remote', 'add', 'origin', origin, cwd=temp)
        if remote_branch:
            run('git', 'fetch', '--depth=1', 'origin', 'gh-pages', cwd=temp)
            run('git', 'checkout', '-qb', 'gh-pages', 'FETCH_HEAD', cwd=temp)
            if not (temp / MARKER).exists():
                raise RuntimeError('Existing gh-pages branch is not managed by this publisher; left unchanged.')
            run('git', 'rm', '-r', '--ignore-unmatch', '.', cwd=temp)
        else:
            run('git', 'checkout', '--orphan', 'gh-pages', cwd=temp)
        for key in ['user.name', 'user.email']:
            run('git', 'config', key, build.git('config', '--get', key).strip(), cwd=temp)
        shutil.copytree(build.OUT, temp, dirs_exist_ok=True)
        (temp / MARKER).write_text(json.dumps({'source': revision}) + '\n')
        run('git', 'add', '.', cwd=temp)
        run('git', 'commit', '-qm', f'Publish exploration snapshot {revision[:7]}', cwd=temp)
        run('git', 'push', 'origin', 'HEAD:gh-pages', cwd=temp, capture=False)
    print('Published snapshot; GitHub Pages deployment can now be checked.')


if __name__ == '__main__':
    main()
