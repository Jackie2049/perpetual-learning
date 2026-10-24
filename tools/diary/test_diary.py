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
        Path('tools/contributions').mkdir()
        shutil.copy2(SOURCE.parent / 'contributions/summary.py', 'tools/contributions/summary.py')
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

    def test_fenced_code_is_not_checked_as_links(self):
        self.record('复现命令：\n\n```python\ntests["check"](key)\n[示例](../not-a-result.md)\n```\n\n'
                    '证据：[成果](../asset.md)。')
        self.git('add', 'diary')
        diary.check()

    def test_link_after_fenced_code_is_still_checked(self):
        self.record('```python\nprint("ok")\n```\n\n结果：[遗漏成果](../missing.md)。')
        self.git('add', 'diary')
        with self.assertRaisesRegex(ValueError, '链接目标'):
            diary.check()

    def test_real_hook_accepts_fenced_python_call(self):
        diary.install()
        record = self.record('复现：\n\n```python\ntests["check"](key)\n```\n\n结果：[成果](../asset.md)。')
        self.git('add', 'diary')
        self.git('commit', '-qm', 'record code example')
        record.write_text(record.read_text() + '\n补充结果：[遗漏](../missing.md)。\n')
        self.git('add', 'diary')
        result = self.git('commit', '-qm', 'must reject real missing link', ok=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('链接目标', result.stderr)

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

    def committed_records(self):
        day = self.now.strftime('%Y-%m-%d')
        path = Path(f'diary/{day}.md')
        path.write_text('# 日志\n\n## 00:00 — 第一项\n\n结果一。\n\n'
                        '## 00:00 — 第二项\n\n结果二。\n')
        Path('diary/readme.md').write_text(f'## 已有日志\n\n- [{day}]({day}.md)\n')
        self.git('add', 'diary')
        self.git('commit', '-qm', 'recorded history')
        return path

    def test_new_record_does_not_excuse_lost_history(self):
        path = self.committed_records()
        path.write_text('# 日志\n\n## 00:00 — 第二项\n\n结果二。\n')
        self.record('本项结果：新增一项，但不应覆盖历史。')
        self.git('add', 'diary')
        with self.assertRaisesRegex(ValueError, '移除已有日志.*第一项'):
            diary.check()

    def test_deleting_a_day_is_rejected_even_with_another_recent_record(self):
        path = self.committed_records()
        yesterday = self.now - timedelta(days=1)
        destination = Path(f'diary/{yesterday:%Y-%m-%d}.md')
        self.git('mv', str(path), str(destination))
        self.git('commit', '-qm', 'older history fixture')
        self.git('rm', str(destination))
        self.record()
        self.git('add', 'diary')
        with self.assertRaisesRegex(ValueError, '移除已有日志'):
            diary.check()

    def test_preserved_body_can_be_reheaded_alongside_new_work(self):
        path = self.committed_records()
        path.write_text(path.read_text().replace('00:00 — 第一项', '00:00 — 更正标题'))
        self.record('结果：改正标题并记录新工作。')
        self.git('add', 'diary')
        diary.check()

    def test_renaming_a_day_does_not_hide_source_history_removal(self):
        path = self.committed_records()
        destination = Path(f'diary/{self.now - timedelta(days=1):%Y-%m-%d}.md')
        self.git('mv', str(path), str(destination))
        index = Path('diary/readme.md')
        index.write_text(index.read_text() + f'\n- [历史]({destination.name})\n')
        self.record('结果：日期文件迁移也不能绕过源记录保全检查。')
        self.git('add', 'diary')
        with self.assertRaisesRegex(ValueError, '移除已有日志'):
            diary.check()

    def test_one_body_cannot_account_for_two_removed_entries(self):
        path = self.committed_records()
        path.write_text(path.read_text().replace('结果二。', '结果一。'))
        self.git('add', 'diary')
        self.git('commit', '-qm', 'same body fixture')
        path.write_text('# 日志\n\n## 00:00 — 合并标题\n\n结果一。\n')
        self.record('结果：合并时不可静默丢失同正文的另一条。')
        self.git('add', 'diary')
        with self.assertRaisesRegex(ValueError, '移除已有日志'):
            diary.check()

    def test_body_cannot_be_emptied_while_heading_survives(self):
        path = self.committed_records()
        path.write_text(path.read_text().replace('结果一。', ''))
        self.record('结果：新增正文不能抵消旧正文清空。')
        self.git('add', 'diary')
        with self.assertRaisesRegex(ValueError, '清空已有日志'):
            diary.check()

    def test_unstaged_recovery_does_not_hide_staged_history_loss(self):
        path = self.committed_records()
        original = path.read_text()
        path.write_text('# 日志\n\n## 00:00 — 第二项\n\n结果二。\n')
        self.record('结果：检查必须只看暂存区。')
        self.git('add', 'diary')
        path.write_text(original + '\n工作区已恢复，但未暂存。\n')
        with self.assertRaisesRegex(ValueError, '移除已有日志'):
            diary.check()

    def test_real_hook_rejects_history_replacement_with_new_record(self):
        path = self.committed_records()
        diary.install()
        path.write_text('# 日志\n')
        self.record('结果：替换历史再加新条目，hook也应拒绝。')
        self.git('add', 'diary')
        result = self.git('commit', '-qm', 'must preserve history', ok=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('移除已有日志', result.stderr)

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


class FenceTests(unittest.TestCase):
    def test_fence_kinds_indentation_and_longer_closers(self):
        for opening, closing in [('```python', '```'), ('~~~sh', '~~~~'), ('   ````py', '   `````')]:
            with self.subTest(opening=opening):
                text = f'before\n{opening}\n[example](missing)\n{closing}\nafter\n'
                prose = diary.without_fenced_code(text)
                self.assertIn('before', prose)
                self.assertIn('after', prose)
                self.assertNotIn('missing', prose)

    def test_short_or_other_kind_fence_does_not_close(self):
        text = '````py\n```\n~~~\n````not-a-close\n[example](missing)\n````\n[real](asset.md)\n'
        prose = diary.without_fenced_code(text)
        self.assertNotIn('missing', prose)
        self.assertIn('[real](asset.md)', prose)

    def test_unclosed_fence_hides_only_remaining_code(self):
        text = '[real](asset.md)\n```py\n[example](missing)\n'
        prose = diary.without_fenced_code(text)
        self.assertIn('[real](asset.md)', prose)
        self.assertNotIn('missing', prose)

    def test_non_fence_and_backtick_info_remain_prose(self):
        for text in ['`` text [real](missing)\n', '```py`info\n[real](missing)\n']:
            with self.subTest(text=text):
                self.assertEqual(diary.without_fenced_code(text), text)


if __name__ == '__main__':
    unittest.main()
