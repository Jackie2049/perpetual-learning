import unittest
import subprocess
import build


class RenderingTests(unittest.TestCase):
    def test_repo_relative_links_keep_snapshot(self):
        result = build.render('[paper](../papers/gmpo.md#goal)', 'notebook/frameworks/test.md', 'abc123')
        self.assertIn('https://github.com/Jackie2049/perpetual-learning/blob/abc123/notebook/papers/gmpo.md#goal', result)

    def test_active_html_is_removed(self):
        result = build.render('<script>alert(1)</script><img src=x onerror=alert(1)><a href="javascript:alert(1)" onclick="alert(1)">bad</a>', 'test.md', 'abc123')
        self.assertNotIn('<script', result)
        self.assertNotIn('<img', result)
        self.assertNotIn('onclick', result)
        self.assertNotIn('javascript:', result)

    def test_markdown_javascript_link_is_removed(self):
        result = build.render('[bad](javascript:alert%281%29)', 'test.md', 'abc123')
        self.assertNotIn('href=', result)

    def test_code_and_tables_survive(self):
        result = build.render('```python\nx = "<script>"\n```\n\n| 影响 | 成本 |\n|---|---|\n| 4 | 2 |', 'test.md', 'abc123')
        self.assertIn('<pre>', result)
        self.assertIn('&lt;script&gt;', result)
        self.assertIn('<table>', result)
        self.assertIn('<td>4</td>', result)

    def test_math_bars_do_not_split_table_cells(self):
        result = build.render('| Method | Weight |\n|---|---|\n| GMPO | **ratio^{1/|o_i|}** |', 'test.md', 'abc123')
        self.assertEqual(result.count('<td>'), 2)
        self.assertIn('ratio^{1/|o_i|}', result)

    def test_issue_number_keeps_hash(self):
        self.assertEqual(build.clean_text('**vLLM #61160**'), 'vLLM #61160')

    def test_inline_and_display_math_survive_markdown_and_cleaning(self):
        result = build.render(r'Inline $\frac{a_i}{b_i}$.' + '\n\n$$\n' + r'\sum_{i=1}^{G} x_i' + '\n$$', 'test.md', 'abc123')
        self.assertIn('<span class="arithmatex">', result)
        self.assertIn('<div class="arithmatex">', result)
        self.assertIn(r'\frac{a_i}{b_i}', result)
        self.assertIn(r'\sum_{i=1}^{G} x_i', result)
        self.assertNotIn('<em>', result)

    def test_math_inside_code_stays_code(self):
        result = build.render('```python\n' + r'formula = "$\frac{a}{b}$"' + '\n```', 'test.md', 'abc123')
        self.assertIn('<pre>', result)
        self.assertNotIn('class="arithmatex"', result)

    def test_table_inline_math_keeps_two_cells(self):
        source = '| Method | Weight |\n|---|---|\n| GMPO | $' + r'\rho^{1/\lvert o_i\rvert}' + '$ |'
        result = build.render(source, 'test.md', 'abc123')
        self.assertEqual(result.count('<td>'), 2)
        self.assertIn('class="arithmatex"', result)
        self.assertIn(r'\lvert o_i\rvert', result)

    def test_gmpo_is_published_as_rendered_math(self):
        source = (build.ROOT / 'notebook/papers/gmpo.md').read_text()
        result = build.render(source, 'notebook/papers/gmpo.md', 'abc123')
        self.assertEqual(result.count('class="katex"'), 4)
        self.assertEqual(result.count('class="katex-display"'), 2)
        self.assertNotIn('class="katex-error"', result)
        self.assertIn('<math ', result)
        self.assertIn('class="katex-html"', result)

    def test_invalid_math_stops_the_build(self):
        with self.assertRaises(subprocess.CalledProcessError):
            build.render(r'$\thisCommandDoesNotExist{x}$', 'test.md', 'abc123')

    def test_escaped_delimiters_and_code(self):
        result = build.render(r'Inline \(x_i\) and `$raw$`.' + '\n\n' + r'\[x^2\]', 'test.md', 'abc123')
        self.assertEqual(result.count('class="katex"'), 2)
        self.assertIn('<code>$raw$</code>', result)

    def test_math_cannot_inject_html(self):
        result = build.render(r'$\text{<script>alert(1)</script>}$', 'test.md', 'abc123')
        self.assertNotIn('<script', result)
        self.assertIn('&lt;script&gt;', result)

    def test_external_and_anchor_links_survive(self):
        result = build.render('[external](https://example.org/a) [section](#section)', 'test.md', 'abc123')
        self.assertIn('href="https://example.org/a"', result)
        self.assertIn('href="#section"', result)


class ContributionTests(unittest.TestCase):
    def test_section_keeps_nested_content_but_excludes_history(self):
        source = '# Project\n## 当前活跃\nCurrent\n### Detail\nEvidence\n## 已过期\nOld'
        self.assertEqual(build.section_source(source, '当前活跃'), 'Current\n### Detail\nEvidence')
        self.assertEqual(build.section_source(source, 'missing'), '')

    def test_tracker_cards_keep_status_pr_and_source_links(self):
        source = '| 候选 | PR | 状态 |\n|---|---|---|\n| Fix #12 | [#34](https://github.com/example/project/pull/34) | **已合并**，见 [验证](proof.md) |'
        items = build.progress_items(build.render(source, 'open-source/test/tracker.md', 'abc123'))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['title'], 'Fix #12')
        self.assertIn('已合并', items[0]['summary'])
        self.assertIn('/pull/34', items[0]['html'])
        self.assertIn('/blob/abc123/open-source/test/proof.md', items[0]['html'])

    def test_handoff_keeps_negative_conclusion_and_verification_boundary(self):
        source = '- **还不是已证实缺陷。** 尚无 PR。\n- CPU 通过，GPU 未验证。'
        items = build.progress_items(build.render(source, 'open-source/test/handoff.md', 'abc123'))
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]['title'], '还不是已证实缺陷。')
        self.assertIn('GPU 未验证', items[1]['summary'])



class SummaryContractTests(unittest.TestCase):
    def setUp(self):
        self.source = (build.ROOT / 'open-source/verl/candidate-8237-offload-fraction.md').read_text()

    def test_header_preserves_hdo_evidence_and_limitations(self):
        fields = build.parse_summary(self.source)
        self.assertIn('/pull/8286', fields['进展'])
        self.assertIn('CPU', fields['大白话讲明白'])
        self.assertIn('未声称吞吐提升', fields['how'])
        self.assertEqual(fields['类型'], '漏洞修复')

    def test_reject_missing_field_duplicate_block_and_old_labels(self):
        for broken in (
            self.source.replace('**【进展】**', '**【近况】**', 1),
            self.source + '\n<!-- contribution-summary:start -->\n',
            self.source.replace('**What：**', '**What｜研究什么：**', 1),
            self.source.replace('9 / 10 —', '42 —', 1),
        ):
            with self.subTest(broken=broken[:80]), self.assertRaises(ValueError):
                build.parse_summary(broken)

    def test_body_cannot_impersonate_header(self):
        with self.assertRaises(ValueError):
            build.parse_summary('# Heading\n\n## Body\n' + self.source.partition('\n')[2])

    def test_staged_check_does_not_read_unstaged_repair(self):
        import tempfile
        import contextlib
        import io
        from pathlib import Path
        from unittest.mock import patch
        import summary
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(['git', 'init', '-q', folder], check=True)
            target = root / 'open-source/new.md'
            target.parent.mkdir()
            target.write_text('# Missing header\n')
            subprocess.run(['git', '-C', folder, 'add', '.'], check=True)
            target.write_text(self.source)
            with patch.object(summary, 'ROOT', root), patch('sys.argv', ['summary.py', '--staged']), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    summary.main()
                subprocess.run(['git', '-C', folder, 'add', '.'], check=True)
                summary.main()

if __name__ == '__main__':
    unittest.main()
