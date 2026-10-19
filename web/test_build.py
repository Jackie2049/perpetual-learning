import unittest
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

    def test_external_and_anchor_links_survive(self):
        result = build.render('[external](https://example.org/a) [section](#section)', 'test.md', 'abc123')
        self.assertIn('href="https://example.org/a"', result)
        self.assertIn('href="#section"', result)


if __name__ == '__main__':
    unittest.main()
