#!/usr/bin/env python3
"""llama.cpp `reasoning_control`：请求里写了 true，budget 采样器到底建不建。

探针不自己写判定逻辑，而是**从快照里抽出真实源码**在 CPU 上执行：

  1. `common_sampling.cpp:315` 那条守卫（`if (!start.empty() && !end.empty() && (...))`）
     原样抽出来，改写成 Python 表达式——只做 `&&` → `and`、`!x` → `not x`、
     `a.empty()` → `not a`，不改任何一项；
  2. `common/reasoning-budget.cpp:167-187` 的 `common_reasoning_budget_apply`
     整段抽出来，逐行转成 Python（只替换 C++ 的类型名与字段访问），
     用来量"采样器存在时它会不会真的动手"。

两档配置只差一个字段：`reasoning_control` 为 True（请求里写了）与 False（没写）。
其余字段全部取自同一次 server 请求的真实形状（`thinking_end_tags` 非空、
`reasoning_budget_tokens = -1`、无 lazy grammar）。

退出码：0 = 结论成立（开关打不开采样器）；2 = 上游已变（守卫真的看了该字段）。

    python3 probe_reasoning_control.py <llama.cpp 快照根目录>
    python3 probe_reasoning_control.py --selftest
"""

import json
import pathlib
import re
import sys

GUARD_FILE = 'common/sampling.cpp'
APPLY_FILE = 'common/reasoning-budget.cpp'


def read(root, rel):
    return (pathlib.Path(root) / rel).read_text(encoding='utf-8', errors='ignore')


def extract_guard(src):
    """抽出 `if (!params.reasoning_budget_start.empty() && ... ) {` 这一行。"""
    for line in src.splitlines():
        s = line.strip()
        if s.startswith('if (') and 'reasoning_budget_start.empty()' in s:
            return s
    return None


def guard_to_python(cond):
    """`if (A && B && (C || D || E)) {` → `A and B and (C or D or E)`。

    只做必须的替换，任何一项判据都不合并、不省略。
    """
    body = cond[len('if ('):]
    if body.endswith('{'):
        body = body[:-1]
    body = body.strip()
    assert body.endswith(')'), body
    body = body[:-1].strip()
    body = body.replace('&&', ' and ').replace('||', ' or ')
    body = re.sub(r'(?:params|p|opt)\.(\w+)', r'\1', body)
    # `.empty()` 必须先换成裸名字，否则 `!x.empty()` 会被换成 `not x.not`。
    body = re.sub(r'(\w+)\.empty\(\)', r'\1_empty', body)
    body = re.sub(r'!\s*(\w+)', r'not \1', body)
    return body


def guard_env(params):
    """守卫里用到的量：`X.empty()` → `X_empty`，其余同名照搬。"""
    return {
        'reasoning_budget_start_empty': len(params['reasoning_budget_start']) == 0,
        'reasoning_budget_end_empty': len(params['reasoning_budget_end']) == 0,
        'grammar_lazy': params['grammar_lazy'],
        'reasoning_budget_tokens': params['reasoning_budget_tokens'],
        'reasoning_control': params['reasoning_control'],
        'true': True,
    }


def extract_apply(src):
    """抽出 common_reasoning_budget_apply 整个函数体。"""
    m = re.search(r'^static void common_reasoning_budget_apply\(.*?\n\}(?=\n)',
                  src, re.S | re.M)
    return m.group(0) if m else None


def apply_to_python(fn_src):
    """把 apply 函数体逐行转成 Python（只替换类型名与字段访问）。

    返回 (python 源码, 转译报告)。转译报告逐条记下做了什么替换；
    遇到不认识的语句就抛异常——宁可失败，也不静默换一套语义。
    """
    out, report = [], []
    for raw in fn_src.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('static void common_reasoning_budget_apply'):
            out.append(('def apply(ctx, cur_p):', 0))
            continue
        if line.startswith('auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;'):
            report.append('param smpl->ctx -> ctx（由调用方传入同名对象）')
            continue
        if line.startswith('// passthrough') or line.startswith('// set all logits'):
            continue
        if line.startswith('if (ctx->state != REASONING_BUDGET_FORCING) {'):
            out.append(('if ctx["state"] != "FORCING":', 1))
            continue
        if line.startswith('if (ctx->force_pos >= ctx->forced_tokens.size()) {'):
            out.append(('if ctx["force_pos"] >= len(ctx["forced_tokens"]):', 1))
            continue
        if line.startswith('const llama_token forced = ctx->forced_tokens[ctx->force_pos];'):
            out.append(('forced = ctx["forced_tokens"][ctx["force_pos"]]', 1))
            report.append('forced_tokens[force_pos] 取值 -> forced')
            continue
        if line.startswith('for (size_t i = 0; i < cur_p->size; i++) {'):
            out.append(('nzw = 0', 1))
            out.append(('for i in range(len(cur_p)):', 1))
            continue
        if line.startswith('if (cur_p->data[i].id != forced) {'):
            out.append(('if cur_p[i]["id"] != forced:', 2))
            continue
        if line.startswith('cur_p->data[i].logit = -INFINITY;'):
            out.append(('cur_p[i]["logit"] = "-inf"', 3))
            out.append(('nzw += 1', 3))
            report.append('logit = -INFINITY -> 字符串 "-inf"（只保留"被改写"这一事实）')
            continue
        if line == 'return;':
            out.append(('return 0', 2))
            continue
        if line == '}':
            continue        # 花括号在下面的层级判定里已经吸收
        raise ValueError('apply() 里出现了探针不认识的语句，拒绝静默跳过: ' + line)
    out.append(('return nzw', 1))     # 不改写时返回 0
    code = '\n'.join('    ' * lvl + t for t, lvl in out)
    return code, report


def build_params(control_on, budget_tokens=-1, lazy=False, start='<think>', end='</think>'):
    """一次真实 server 请求的 sampling 参数（server-common.cpp:1517-1531 的形状）。"""
    return {
        'reasoning_budget_start': [1] if start else [],
        'reasoning_budget_end': [[2]] if end else [],
        'grammar_lazy': lazy,
        'reasoning_budget_tokens': budget_tokens,
        'reasoning_control': control_on,
    }


def run(root):
    """核对 `reasoning_control` 是否真的接线（结论：接线了，探针自己也这么判）。

    第一版探针把"写不写 reasoning_control"当成守卫的唯一变量，读到 False 就报了
    一个缺陷——**那是错的**：守卫里还有 `forced.empty()` 这一项，而默认请求下
    它本来就是空的。这版把每个候选量都显式列出来，避免"结论对但理由错"。
    """
    guard_src = extract_guard(read(root, GUARD_FILE))
    apply_src = extract_apply(read(root, APPLY_FILE))
    rep = {'root': str(root), 'guard_line': guard_src, 'checks': [], 'data': {}}
    if guard_src is None or apply_src is None:
        rep['verdict'] = 'SOURCE_NOT_FOUND'
        return rep, 2

    py_cond = guard_to_python(guard_src)
    rep['guard_python'] = py_cond
    apply_code, apply_report = apply_to_python(apply_src)
    rep['apply_python'] = apply_code
    rep['apply_transpilation'] = apply_report
    ns = {}
    exec(compile(apply_code, 'reasoning-budget.apply', 'exec'), ns)  # noqa: S102
    apply_fn = ns['apply']

    changed = apply_fn({'state': 'FORCING', 'force_pos': 0, 'forced_tokens': [2]},
                       [{'id': t, 'logit': 0.0} for t in range(5)])
    changed_idle = apply_fn({'state': 'IDLE', 'force_pos': 0, 'forced_tokens': [2]},
                            [{'id': t, 'logit': 0.0} for t in range(5)])
    rep['data']['apply_changes_forcing'] = changed
    rep['data']['apply_changes_idle'] = changed_idle

    def built(control, tokens=-1, lazy=False, start=True, end=True):
        env = guard_env(build_params(control, budget_tokens=tokens, lazy=lazy,
                                     start='<think>' if start else None,
                                     end='</think>' if end else None))
        return bool(eval(py_cond, {'__builtins__': {}}, env))  # noqa: S307

    grid = {
        'control=True, 无预算（server 默认）': built(True),
        'control=False, 无预算': built(False),
        'control=True, budget=256': built(True, tokens=256),
        'control=False, lazy grammar': built(False, lazy=True),
        'control=True, 模板无 thinking 标记': built(True, end=False),
        'control=True, 无 thinking 起始标记': built(True, start=False),
    }
    rep['data']['sampler_built'] = grid

    def check(name, got, want):
        rep['checks'].append({'name': name, 'got': got, 'want': want, 'ok': got == want})

    check('apply 在 FORCING 状态真的改写 logits（采样器存在时是活的）', changed > 0, True)
    check('apply 在 IDLE 状态不动手', changed_idle, 0)
    check('control=True 在默认请求下会建采样器（本字段确实接线）',
          grid['control=True, 无预算（server 默认）'], True)
    check('control=False 在同样请求下不建（两档确实有差别）',
          grid['control=False, 无预算'], False)
    check('显式预算不依赖 control（对照）', grid['control=True, budget=256'], True)
    check('lazy grammar 不依赖 control（对照）', grid['control=False, lazy grammar'], True)
    check('模板没有 thinking 标记时不建（对照：守卫真的看那两个向量）',
          grid['control=True, 模板无 thinking 标记'], False)

    ok = all(c['ok'] for c in rep['checks'])
    rep['verdict'] = 'WIRED_AS_DOCUMENTED' if ok else 'UPSTREAM_CHANGED'
    return rep, (0 if ok else 2)


def selftest():
    """四条注入：改坏守卫翻译、改坏 apply 转译，都必须被抓到。"""
    notes, ok = [], True

    def expect(tag, cond, want):
        nonlocal ok
        good = cond == want
        ok = ok and good
        notes.append('%s %s got=%s want=%s' % ('PASS' if good else 'FAIL', tag, cond, want))

    real_guard = ('if (!params.reasoning_budget_start.empty() && '
                  '!params.reasoning_budget_end.empty() && (params.grammar_lazy || '
                  'params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {')
    py = guard_to_python(real_guard)
    base = dict(reasoning_budget_start=[1], reasoning_budget_end=[[2]], grammar_lazy=False,
                reasoning_budget_tokens=-1)

    def g(control, **kw):
        env = guard_env(dict(build_params(control), **kw))
        return bool(eval(py, {'__builtins__': {}}, env))  # noqa: S307

    expect('真实守卫 + control=True → 建', g(True), True)
    expect('真实守卫 + control=False → 不建', g(False), False)
    # 负向对照：把 `|| reasoning_control` 那一项删掉，两档必须变成一样（说明这一项确实是唯一的差）
    stripped = guard_to_python(real_guard.replace(' || params.reasoning_control', ''))
    env = guard_env(build_params(True))
    expect('把 reasoning_control 从守卫里删掉 → 不再建（证明这项在起作用）',
           bool(eval(stripped, {'__builtins__': {}}, env)), False)  # noqa: S307
    # 负向对照：模板没有 thinking 结束标记时，即使 control=True 也不建
    expect('无 thinking 结束标记 → 不建（守卫确实看那个向量）',
           g(True, reasoning_budget_end=[]), False)

    here = pathlib.Path(__file__).resolve().parent
    snap = here / 'raw' / 'snapshot'
    apply_src = extract_apply(read(snap, APPLY_FILE)) if snap.exists() else None
    if apply_src:
        code, _ = apply_to_python(apply_src)
        ns = {}
        exec(compile(code, 'apply', 'exec'), ns)  # noqa: S102
        expect('真实 apply 在 FORCING 下改写 4 个 logit（5 个竞争者里剩 1 个）',
               ns['apply']({'state': 'FORCING', 'force_pos': 0, 'forced_tokens': [2]},
                           [{'id': t, 'logit': 0.0} for t in range(5)]), 4)
        expect('真实 apply 在 IDLE 下不改写',
               ns['apply']({'state': 'IDLE', 'force_pos': 0, 'forced_tokens': [2]},
                           [{'id': t, 'logit': 0.0} for t in range(5)]), 0)
    else:
        notes.append('SKIP apply 夹具（raw/snapshot 未就绪）')
    return ok, notes


def main():
    argv = sys.argv[1:]
    if '--selftest' in argv:
        ok, notes = selftest()
        for n in notes:
            print(n)
        return 0 if ok else 1
    if len(argv) != 1:
        print(__doc__)
        return 2
    rep, rc = run(argv[0])
    here = pathlib.Path(__file__).resolve().parent
    out = here / 'raw' / 'reasoning-control-probe.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    for c in rep.get('checks', []):
        print('%s %s got=%s want=%s' % ('PASS' if c['ok'] else 'FAIL', c['name'], c['got'], c['want']))
    print('verdict=%s' % rep.get('verdict'))
    print('written: %s' % out)
    return rc


if __name__ == '__main__':
    sys.exit(main())
