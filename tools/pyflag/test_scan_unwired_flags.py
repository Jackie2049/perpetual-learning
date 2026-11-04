#!/usr/bin/env python3
"""扫描器的单元测试（2026-11-02 M1）。

纪律：每一条判定都要有能推翻它的对照。本文件的测试用例全部是**构造的小仓库**，
不依赖网络与外部源码；另有两条"回归"用例指向真实快照（存在时才跑）。

四个必须钉住的坑（都是实际踩到的）：
  ① 构造器的调用点写的是**类名**，不是 `__init__`；
  ② Ray 的 `.remote(...)` 把方法名包了一层；
  ③ 函数体内给同名名字重新赋值后，`if x:` 不再是读形参；
  ④ 形参自己的默认值（`f(use_x=False)`）不是"生产者"。
"""

import ast
import pathlib
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import scan_unwired_flags as m  # noqa: E402


def make_repo(files):
    """在临时目录里造一个小仓库，返回路径。files: {相对路径: 源码}。"""
    d = pathlib.Path(tempfile.mkdtemp())
    for rel, src in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src), encoding='utf-8')
    return d


class TestTruePositive(unittest.TestCase):
    def test_consumer_without_producer_is_inert(self):
        """真正的失效：形参被分支使用，但没有任何调用点传它。"""
        d = make_repo({
            'pkg/engine.py': '''
                def apply_patch(model, use_fancy=False):
                    if use_fancy:
                        model.patch()
                    return model
            ''',
            'pkg/worker.py': '''
                from pkg.engine import apply_patch
                def build(model):
                    return apply_patch(model)
            ''',
        })
        r = m.scan(d, 'use_fancy')
        self.assertTrue(r['silently_inert'], r)
        self.assertEqual(len(r['deciding_consumers']), 1)

    def test_producer_keyword_is_not_inert(self):
        """对照：一旦有人传了，就不再是失效。"""
        d = make_repo({
            'pkg/engine.py': '''
                def apply_patch(model, use_fancy=False):
                    if use_fancy:
                        model.patch()
                    return model
            ''',
            'pkg/worker.py': '''
                from pkg.engine import apply_patch
                def build(model):
                    return apply_patch(model, use_fancy=True)
            ''',
        })
        r = m.scan(d, 'use_fancy')
        self.assertFalse(r['silently_inert'], r)
        self.assertGreaterEqual(r['producers_total'], 1)


class TestPitfallClassConstructor(unittest.TestCase):
    def test_call_site_uses_class_name(self):
        """坑 ①：`Actor(..., use_x=...)` 的消费者是 `Actor.__init__`。"""
        d = make_repo({
            'pkg/model.py': '''
                class Actor:
                    def __init__(self, name, use_liger=False):
                        if use_liger:
                            self.fast = True
            ''',
            'pkg/train.py': '''
                from pkg.model import Actor
                a = Actor("m", use_liger=True)
            ''',
        })
        r = m.scan(d, 'use_liger')
        self.assertFalse(r['silently_inert'], r)

    def test_unwired_constructor_is_inert(self):
        d = make_repo({
            'pkg/model.py': '''
                class Actor:
                    def __init__(self, name, use_liger=False):
                        if use_liger:
                            self.fast = True
            ''',
            'pkg/train.py': '''
                from pkg.model import Actor
                a = Actor("m")
            ''',
        })
        r = m.scan(d, 'use_liger')
        self.assertTrue(r['silently_inert'], r)


class TestPitfallRayWrapper(unittest.TestCase):
    def test_remote_wrapper_is_unwrapped(self):
        """坑 ②：`eng.fn.remote(..., use_ray=x)` 仍应算作给 `fn` 传了值。"""
        d = make_repo({
            'pkg/engine.py': '''
                class Engine:
                    def init_group(self, addr, use_ray):
                        if use_ray:
                            return "ray"
                        return "nccl"
            ''',
            'pkg/actor.py': '''
                def launch(engines, flag):
                    for e in engines:
                        e.init_group.remote("addr", use_ray=flag)
            ''',
        })
        r = m.scan(d, 'use_ray')
        self.assertFalse(r['silently_inert'], r)

    def test_plain_method_call_still_detected(self):
        d = make_repo({
            'pkg/engine.py': '''
                class Engine:
                    def init_group(self, addr, use_ray):
                        if use_ray:
                            return "ray"
                        return "nccl"
            ''',
            'pkg/actor.py': '''
                def launch(eng):
                    eng.init_group("addr", use_ray=True)
            ''',
        })
        r = m.scan(d, 'use_ray')
        self.assertFalse(r['silently_inert'], r)


class TestPitfallLocalRebinding(unittest.TestCase):
    def test_rebound_name_is_not_a_branch(self):
        """坑 ③：函数体内 `use_ray = getattr(...)` 后，`if use_ray:` 读的是局部变量。"""
        d = make_repo({
            'pkg/patch.py': '''
                def fix(use_ray):
                    use_ray = getattr(object(), "x", False)
                    if use_ray:
                        return 1
                    return 0
            ''',
            'pkg/caller.py': '''
                from pkg.patch import fix
                def go():
                    return fix(True)
            ''',
        })
        r = m.scan(d, 'use_ray')
        # 形参被重新绑定 ⇒ 不构成"按该形参分支"，故不应判为可决定的消费者
        self.assertEqual(len(r['deciding_consumers']), 0, r)
        self.assertFalse(r['silently_inert'])


class TestPitfallDefaultIsNotProducer(unittest.TestCase):
    def test_signature_default_not_counted(self):
        """坑 ④：形参默认值 `use_x=False` 出现在源码行里，但不是生产者。"""
        d = make_repo({
            'pkg/engine.py': '''
                def run(model, use_x=False):
                    if use_x:
                        model.turbo()
            ''',
        })
        r = m.scan(d, 'use_x')
        self.assertEqual(r['producers_total'], 0)
        self.assertTrue(r['silently_inert'])


class TestShadowedArms(unittest.TestCase):
    """第十一类：同一 if/elif 链上，前面的 arm 被调用点钉死为 True。"""

    SRC = textwrap.dedent("""
        def kernel(stage, use_sfa_byte_offset=False, use_split_scale=False, other=0):
            if use_sfa_byte_offset:
                base = 1
            elif use_split_scale:
                base = 2
            else:
                base = 3
            return base

        def caller():
            return kernel(1, use_sfa_byte_offset=True)
    """)

    def test_pinned_sibling_shadows_later_arm(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / 'k.py').write_text(self.SRC)
            hits = m.shadowed_arms(root / 'k.py', ast.parse(self.SRC))
        self.assertEqual([(h['flag'], h['shadowed_by']) for h in hits],
                         [('use_split_scale', 'use_sfa_byte_offset')], hits)

    def test_false_at_any_call_site_keeps_arm_alive(self):
        src = self.SRC + textwrap.dedent("""
        def caller2():
            return kernel(1, use_sfa_byte_offset=False)
        """)
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / 'k.py').write_text(src)
            hits = m.shadowed_arms(root / 'k.py', ast.parse(src))
        self.assertEqual(hits, [])

    def test_computed_value_keeps_arm_alive(self):
        src = self.SRC + textwrap.dedent("""
        def caller3(bias):
            has = bias is not None
            return kernel(1, use_sfa_byte_offset=has)
        """)
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / 'k.py').write_text(src)
            hits = m.shadowed_arms(root / 'k.py', ast.parse(src))
        self.assertEqual(hits, [], "a bare name may be False at runtime")

    def test_forwarding_call_is_not_a_decision(self):
        src = self.SRC + textwrap.dedent("""
        def mid(use_sfa_byte_offset=False):
            return kernel(1, use_sfa_byte_offset=use_sfa_byte_offset)

        def caller():
            return mid(use_sfa_byte_offset=True)
        """)
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / 'k.py').write_text(src)
            hits = m.shadowed_arms(root / 'k.py', ast.parse(src))
        # mid() forwards, so the only deciding call pins nothing at kernel();
        # the arm is reported only because caller() pins it one hop up.
        self.assertTrue(all(h['flag'] == 'use_split_scale' for h in hits), hits)


class TestRealSnapshotsIfPresent(unittest.TestCase):
    """真实快照在我机器上时才跑；缺失就跳过，不让测试依赖网络。"""

    VERL = pathlib.Path('/tmp/mine/verl-project-verl-6879e1ab07')
    ORLHF = pathlib.Path('/tmp/mine/OpenRLHF-OpenRLHF')

    @unittest.skipUnless(VERL.exists(), '需要 verl 快照')
    def test_verl_prefix_grouper_is_inert(self):
        r = m.scan(self.VERL, 'use_prefix_grouper')
        self.assertTrue(r['silently_inert'], r)

    @unittest.skipUnless(ORLHF.exists(), '需要 OpenRLHF 快照')
    def test_openrlhf_has_no_false_positive(self):
        for flag in ('use_liger_kernel', 'use_ray'):
            r = m.scan(self.ORLHF, flag)
            self.assertFalse(r['silently_inert'], (flag, r))


if __name__ == '__main__':
    unittest.main()
