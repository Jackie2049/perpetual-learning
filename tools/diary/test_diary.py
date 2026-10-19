"""Integration tests use disposable repositories, never the working project index."""
import contextlib
from datetime import datetime, timedelta
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import diary

SOURCE = Path(__file__).resolve().parent


class DiaryTests(unittest.TestCase):
    def setUp(self):
        self.cwd = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        os.chdir(self.temp.name)
        self.env = patch.dict(os.environ, {'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_NOSYSTEM': '1'})
        self.env.start()
        self.git('init', '-q')
        self.git('config', 'user.name', 'Diary Test')
        self.git('config', 'user.email', 'diary@example.invalid')
        Path('diary').mkdir()
        Path('diary/readme.md').write_text('# 工作日志\n\n## 已有日志\n')
        Path('asset.md').write_text('baseline\n')
        Path('tools/diary/hooks').mkdir(parents=True)
        shutil.copy2(SOURCE / 'diary.py', 'tools/diary/diary.py')
        shutil.copy2(SOURCE / 'hooks/pre-commit', 'tools/diary/hooks/pre-commit')
        self.git('add', '.')
        self.git('commit', '-qm', 'initial fixture')
        self.now = datetime.now(diary.TZ)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)
        self.env.stop()
        os.chdir(self.cwd)
        self.temp.cleanup()

    def git(self, *args, ok=True):
        result = subprocess.run(['git', *args], capture_output=True, text=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def record(self, body='结果：核对完成；证据见 [成果](../asset.md)。'):
        diary.append('验证具体工作', body)
        return Path(f'diary/{datetime.now(diary.TZ):%Y-%m-%d}.md')

    def test_missing_record_blocks(self):
        Path('asset.md').write_text('changed')
        self.git('add', 'asset.md')
        with self.assertRaisesRegex(ValueError, '缺少对应日志'):
            diary.check()

    def test_unstaged_record_does_not_count(self):
        Path('asset.md').write_text('changed')
        self.record()
        self.git('add', 'asset.md')
        with self.assertRaisesRegex(ValueError, '缺少对应日志'):
            diary.check()

    def test_append_generates_time_index_and_log_only_commit_passes(self):
        p = self.record()
        self.assertIn(f'## {datetime.now(diary.TZ):%H:%M} —', p.read_text())
        self.assertIn(f']({p.name})', Path('diary/readme.md').read_text())
        self.git('add', 'diary')
        diary.check()

    def test_future_record_rejected(self):
        future = self.now + timedelta(days=1)
        name = f'diary/{future:%Y-%m-%d}.md'
        Path(name).write_text(f'# 日志\n\n## {future:%H:%M} — 工作\n\n实际正文\n')
        self.git('add', name)
        with self.assertRaisesRegex(ValueError, '未来时间'):
            diary.check(self.now)

    def test_unstaged_index_rejected(self):
        p = self.record()
        self.git('add', str(p))
        with self.assertRaisesRegex(ValueError, '索引'):
            diary.check()

    def test_untracked_link_rejected_then_staged_link_passes(self):
        Path('new.md').write_text('new result')
        self.record('结果：[新成果](../new.md)。')
        self.git('add', 'diary')
        with self.assertRaisesRegex(ValueError, '链接目标'):
            diary.check()
        self.git('add', 'new.md')
        diary.check()

    def test_previous_day_allowed_but_old_log_not_enough(self):
        yesterday = self.now - timedelta(days=1)
        p = Path(f'diary/{yesterday:%Y-%m-%d}.md')
        p.write_text(f'# 日志\n\n## 23:59 — 工作\n\n结果：跨零点收尾。\n')
        Path('diary/readme.md').write_text(f'## 已有日志\n\n- [{yesterday:%Y-%m-%d}]({p.name})\n')
        self.git('add', 'diary')
        diary.check(self.now)
        with self.assertRaisesRegex(ValueError, '缺少对应日志'):
            diary.check(self.now + timedelta(days=1))

    def test_existing_record_update_allowed_but_summary_or_title_only_not(self):
        p = self.record()
        self.git('add', 'diary')
        self.git('commit', '-qm', 'recorded work')
        original = p.read_text()
        p.write_text(original.replace('验证具体工作', '换个标题'))
        self.git('add', 'diary')
        with self.assertRaisesRegex(ValueError, '缺少对应日志'):
            diary.check()
        p.write_text(original.replace('\n\n## ', '\n\n## 当日回看\n\n更新摘要。\n\n## ', 1))
        self.git('add', 'diary')
        with self.assertRaisesRegex(ValueError, '缺少对应日志'):
            diary.check()
        p.write_text(original + '\n补充验证：第二种输入也通过，范围仍为本机 CPU。\n')
        self.git('add', 'diary')
        diary.check()

    def test_multiple_entries_parsed_independently(self):
        text = '## 00:01 — 一\n\n正文一\n\n## 00:02 — 二\n\n正文二\n'
        self.assertEqual(len(diary.entries(text)), 2)
        self.assertEqual(diary.entries(text)[('00:01', '一')], '正文一')

    def test_duplicate_append_rejected_without_mutation(self):
        p = self.record()
        before = p.read_text()
        with self.assertRaisesRegex(ValueError, '已存在'):
            self.record()
        self.assertEqual(p.read_text(), before)

    def test_install_preserves_existing_hooks(self):
        self.git('config', 'core.hooksPath', 'other-hooks')
        with self.assertRaisesRegex(ValueError, '未覆盖'):
            diary.install()
        self.assertEqual(self.git('config', '--get', 'core.hooksPath').stdout.strip(), 'other-hooks')
        self.git('config', '--unset', 'core.hooksPath')
        hook = Path('.git/hooks/pre-commit')
        hook.write_text('# existing hook\n')
        with self.assertRaisesRegex(ValueError, '未覆盖'):
            diary.install()
        self.assertEqual(hook.read_text(), '# existing hook\n')

    def test_real_git_commit_hook(self):
        diary.install()
        diary.install()  # idempotent
        Path('asset.md').write_text('changed')
        self.git('add', 'asset.md')
        result = self.git('commit', '-qm', 'must block', ok=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('缺少对应日志', result.stderr)
        self.record()
        self.git('add', 'diary')
        self.git('commit', '-qm', 'recorded work')


if __name__ == '__main__':
    unittest.main()
