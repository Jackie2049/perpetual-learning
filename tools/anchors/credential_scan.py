#!/usr/bin/env python3
"""凭据哨兵：扫工作区里的明文密码 / API key / 私钥。

起因（2026-11-03）：本仓库曾有**两个不同**的 SSH 密码明文躺在 6 个文件里，
最久的从 2026-06-05 起就在 git 历史中。发现过程是偶然的——加"孤儿文档"
检查时撞见。**不能靠运气下一次。**

## 用法

```bash
python3 tools/anchors/credential_scan.py            # 扫工作区
python3 tools/anchors/credential_scan.py --staged    # 只扫暂存区（可接 pre-commit）
python3 tools/anchors/credential_scan.py --selftest  # 负向对照：证明规则真能抓
```

退出码：0 = 干净；1 = 有命中。

## 设计取舍

- **只做高置信度模式**，宁漏不误报——误报会让人关掉哨兵，那比没有更糟。
- **不联网、不调用外部服务**。
- **不看 `diary/` 里"已清理"类记录的历史引用**：真的清理过就不该再出现，
  所以照扫；正文里用占位符即可（这条是给自己写的教训）。
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

# 高置信度模式：误报代价高，只留明确是凭据的
PATTERNS = [
    # sshpass -p 'literal'（变量展开 $VAR / "$VAR" 不算）
    (re.compile(r"""sshpass\s+-p\s+['"]([^'"$\s][^'"]{2,})['"]"""), 'sshpass 明文密码'),
    # password = "literal" / password: literal（排除占位符与变量）
    (re.compile(r"""password\s*[:=]\s*['"]([^'"$\s<>{}\[\]]{4,})['"]""", re.I), 'password 字面量'),
    # 常见 token 前缀
    (re.compile(r"""\b(sk-[A-Za-z0-9]{20,})\b"""), 'OpenAI 风格 key'),
    (re.compile(r"""\b(ghp_[A-Za-z0-9]{20,})\b"""), 'GitHub PAT'),
    (re.compile(r"""\b(AKIA[0-9A-Z]{16})\b"""), 'AWS Access Key'),
    # 私钥块
    (re.compile(r"""-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"""), '私钥'),
]

SKIP_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv',
             # 第三方源码 checkout：不是我们的代码，也不该由我们的哨兵管
             'vllm', 'vllm-latest', 'sglang', 'verl', 'AReaL', 'triton',
             'site-packages', 'dist-packages'}
SKIP_FILES = {'credential_scan.py'}          # 哨兵自身的测试样本会命中自己
SKIP_SUFFIX = {'.png', '.jpg', '.jpeg', '.gif', '.zip', '.gz', '.pack', '.idx',
               '.pdf', '.tar', '.whl', '.so', '.bin'}

# 明确允许的占位符（会出现在文档里）
ALLOWLIST = {'<原密码>', '<第二个密码>', '<密码>', 'your_password', 'xxx', '***',
             'REDACTED', 'CHANGEME', '$GPU_SSH_PASS', '${GPU_SSH_PASS}'}


def iter_files(root):
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if SKIP_DIRS & set(p.parts):
            continue
        if p.suffix.lower() in SKIP_SUFFIX:
            continue
        if p.name in SKIP_FILES:
            continue
        yield p


def scan_text(text):
    hits = []
    for pat, label in PATTERNS:
        for m in pat.finditer(text):
            val = m.group(1) if m.groups() else m.group(0)
            if val in ALLOWLIST or (val.startswith('<') and val.endswith('>')):
                continue
            if val.startswith('...') or val.startswith('…'):
                continue
            hits.append((label, val))
    return hits


def scan_worktree():
    out = []
    for p in iter_files(ROOT):
        try:
            text = p.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        for label, val in scan_text(text):
            out.append({'file': p.relative_to(ROOT).as_posix(), 'kind': label,
                        'sample': val[:4] + '…'})
    return out


def scan_staged():
    names = subprocess.run(['git', '-C', str(ROOT), 'diff', '--cached',
                            '--name-only', '--diff-filter=ACMR'],
                           capture_output=True, text=True).stdout.splitlines()
    out = []
    for name in names:
        if not name.strip():
            continue
        try:
            text = subprocess.run(['git', '-C', str(ROOT), 'show', ':' + name],
                                  capture_output=True, text=True).stdout
        except OSError:
            continue
        for label, val in scan_text(text):
            out.append({'file': name, 'kind': label, 'sample': val[:4] + '…'})
    return out


def selftest():
    """负向对照：用真实泄露形态测试规则，用占位符测试不误报。"""
    must_catch = [
        ("sshpass -p 'hunter2xyz' ssh root@h", 'sshpass'),
        ('password: "abc12345"', 'password'),
        ('sk-' + 'a' * 24, 'openai'),
        ('ghp_' + 'B' * 24, 'github'),
        ('AKIA' + 'C' * 16, 'aws'),
        ('-----BEGIN RSA PRIVATE KEY-----', 'pem'),
    ]
    must_pass = [
        'sshpass -p "$GPU_SSH_PASS" ssh root@h',
        '密码不入库，见本地凭据管理',
        'password: <原密码>',
    ]
    ok = True
    for text, what in must_catch:
        got = scan_text(text)
        hit = bool(got)
        ok &= hit
        print(f"  {'OK  ' if hit else 'FAIL'}  应抓到 {what:10s} -> {'抓到' if hit else '漏了'}")
    for text in must_pass:
        got = scan_text(text)
        clean = not got
        ok &= clean
        print(f"  {'OK  ' if clean else 'FAIL'}  不应报 {text[:34]:34s} -> {'干净' if clean else '误报'}")
    return 0 if ok else 1


def main():
    if '--selftest' in sys.argv:
        return selftest()
    hits = scan_staged() if '--staged' in sys.argv else scan_worktree()
    if not hits:
        print('OK  未发现凭据')
        return 0
    print(f'⚠️  发现 {len(hits)} 处疑似凭据：')
    for h in hits:
        print(f"   {h['file']}  [{h['kind']}]  {h['sample']}")
    print('\n**不要提交。** 改为从环境变量读，或从文件中移除。')
    return 1


if __name__ == '__main__':
    sys.exit(main())
