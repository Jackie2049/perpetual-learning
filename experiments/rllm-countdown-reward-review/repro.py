"""Independent check of issue #945 and PR #946 in rllm-org/rllm.

Two questions, both answered by running the *upstream* code, not a rewrite of it:

  (1) On main `4e828c8`, does `compute_score` in countdown_reward.py really give
      full reward to answers that break the task rule (only + - * / on the given
      integers)? And does an exponent chain really keep `eval` busy?
  (2) With PR #946's one-hunk patch applied, do any *legal* answers lose reward?

The module under test imports `rllm` and `rllm.rewards.reward_types`, which need
a full rllm install; they are stubbed with the minimal surface the file touches so
the reward logic itself runs verbatim. The snapshot is a copy of the upstream file
at `4e828c8`, so this runs without network access to github.

Usage:
    python repro.py            # both checks, exit 0 if every stated result holds
    python repro.py --json     # machine-readable result on stdout
"""

import ast
import json
import operator
import random
import signal
import sys
import types

SNAPSHOT = "countdown_reward_main_4e828c8.py"

# Exactly the text of PR #946's hunk, so a mismatch raises instead of silently
# "verifying" an unchanged file. See the PR's diff for this hunk.
PR_946_BEFORE = (
    '    allowed_pattern = r"^[\\d+\\-*/().\\s]+$"\n'
    "        if not re.match(allowed_pattern, equation_str):\n"
    '            raise ValueError("Invalid characters in equation.")'
)
PR_946_AFTER = (
    '    allowed_pattern = r"^[\\d+\\-*/()\\s]+$"\n'
    '        if not re.match(allowed_pattern, equation_str) or "**" in equation_str or "//" in equation_str:\n'
    '            raise ValueError("Invalid characters in equation.")'
)


def load_compute_score(source_text):
    """exec the real module with stubbed rllm imports; return its compute_score."""
    mod = types.ModuleType("rllm")
    mod.Action = object
    sys.modules["rllm"] = mod
    sys.modules["rllm.rewards"] = types.ModuleType("rllm.rewards")
    reward_types = types.ModuleType("rllm.rewards.reward_types")

    class RewardOutput:
        def __init__(self, reward=0.0, is_correct=False, metadata=None):
            self.reward = reward
            self.is_correct = is_correct
            self.metadata = metadata

    reward_types.RewardOutput = RewardOutput
    sys.modules["rllm.rewards.reward_types"] = reward_types

    namespace = {}
    exec(compile(source_text, "<countdown_reward>", "exec"), namespace)
    return namespace["compute_score"]


def answer(equation):
    return f"<answer>{equation}</answer>"


def call_with_deadline(compute_score, solution, ground_truth, seconds):
    """Run one scoring call under SIGALRM; return (outcome, value, elapsed)."""
    import time

    def on_alarm(signum, frame):
        raise TimeoutError("eval did not return")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(seconds)
    start = time.time()
    try:
        value = compute_score(solution, ground_truth)
        return "returned", value, time.time() - start
    except TimeoutError:
        return "timeout", None, time.time() - start
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


# --------------------------------------------------------------------------- (1)

# Issue #945's four examples, plus a legal control. Expected values are constants
# written out here, not read back from the code under test.
ISSUE_CASES = [
    ("1.5 * 2", {"target": 3, "numbers": [1, 5, 2]}, 1.0, "decimal point joins 1 and 5 into one token"),
    (".5 * 8", {"target": 4, "numbers": [5, 8]}, 1.0, "leading decimal"),
    ("3 ** 2 * 5 - 7", {"target": 38, "numbers": [3, 2, 5, 7]}, 1.0, "** accepted"),
    ("7 // 2", {"target": 3, "numbers": [7, 2]}, 1.0, "// accepted"),
    ("(3*2)+(5-7)", {"target": 4, "numbers": [3, 2, 5, 7]}, 1.0, "legal control: must also score 1.0"),
]

# Cost of an exponent chain, measured rather than asserted. Each entry gets its
# own deadline so a pathological case cannot hang the script.
CHAIN_CASES = [
    ("44**19**2", {"target": 1, "numbers": [44, 19, 2]}, 5),
    ("44**19**5", {"target": 1, "numbers": [44, 19, 5]}, 30),
]

# The issue quotes `44**19**35` as never returning inside 10 s. That is not
# reproduced on purpose: the value has about 10**45 digits, the run cannot finish,
# and killing it leaves nothing to report beyond the statement. The two cases above
# are the ones whose cost is measurable on this machine (and both stay under 3 s,
# so the "one sampled answer can stall a reward worker" reading is bounded by the
# exponent height the model actually writes).


# --------------------------------------------------------------------------- (2)

BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def evaluate(node):
    if isinstance(node, ast.Expression):
        return evaluate(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return BINOPS[type(node.op)](evaluate(node.left), evaluate(node.right))
    raise ValueError(f"unexpected node {type(node).__name__}")


def legal_expressions(numbers):
    """Every fully parenthesised expression over `numbers` (each used once, order fixed)."""
    if len(numbers) == 1:
        yield str(numbers[0])
        return
    for split in range(1, len(numbers)):
        for left in legal_expressions(numbers[:split]):
            for right in legal_expressions(numbers[split:]):
                for op in "+-*/":
                    yield f"({left} {op} {right})"


def patch_rule_rejects(equation):
    """Does PR #946's rule reject this answer text?"""
    return "." in equation or "**" in equation or "//" in equation


def search_regressions(compute_main, compute_pr, draws=300, seed=0, limit=5):
    """Compare main against the patched rule on every legal answer it grants.

    A "regression" is an answer that scores 1.0 on main and less than 1.0 after
    the patch. Scope is every legal answer main gives full reward to, not only the
    ones the patch plainly rejects: a bug in the patched rule could also lose
    reward on text it is *supposed* to keep (e.g. an over-greedy match), and that
    is exactly the class of regression this search has to be able to see.
    """
    random.seed(seed)
    legal_full_reward = 0
    regressions = []
    combos = []
    for _ in range(draws):
        size = random.choice([3, 4])  # Countdown-Tasks-3to4: three or four numbers
        combos.append(tuple(random.randint(1, 99) for _ in range(size)))

    for numbers in combos:
        for equation in legal_expressions(list(numbers)):
            try:
                value = evaluate(ast.parse(equation, mode="eval"))
            except ZeroDivisionError:
                continue
            if abs(value - round(value)) > 1e-9 or abs(value) > 1e6:
                continue  # only integer-valued answers correspond to a real target
            ground_truth = {"target": int(round(value)), "numbers": list(numbers)}
            # `<answer>...</answer>` is mandatory: compute_score reads the equation out
            # of the tag and scores 0 without it, which would fake "both sides agree".
            solution = answer(equation)
            before = compute_main(solution, ground_truth)
            if before != 1.0:
                continue  # nothing to regress from
            legal_full_reward += 1
            after = compute_pr(solution, ground_truth)
            if after != 1.0:
                regressions.append(
                    {
                        "equation": equation,
                        "target": ground_truth["target"],
                        "numbers": ground_truth["numbers"],
                        "rejected_by_patch_rule_on_sight": patch_rule_rejects(equation),
                        "before": before,
                        "after": after,
                    }
                )
        if len(regressions) >= limit:
            break
    return {"legal_answers_with_full_reward": legal_full_reward, "regressions": regressions}


# ---------------------------------------------------------------------------

def main():
    with open(SNAPSHOT, encoding="utf-8") as handle:
        source = handle.read()

    if PR_946_BEFORE not in source:
        raise SystemExit("snapshot does not contain PR #946's context lines; refresh the snapshot")
    compute_main = load_compute_score(source)
    compute_pr = load_compute_score(source.replace(PR_946_BEFORE, PR_946_AFTER))

    issue_results = []
    for equation, ground_truth, expected, note in ISSUE_CASES:
        score = compute_main(answer(equation), ground_truth)
        issue_results.append(
            {
                "equation": equation,
                "target": ground_truth["target"],
                "numbers": ground_truth["numbers"],
                "score_on_main": score,
                "expected": expected,
                "note": note,
            }
        )

    chain_results = []
    for equation, ground_truth, deadline in CHAIN_CASES:
        outcome, score, elapsed = call_with_deadline(compute_main, answer(equation), ground_truth, deadline)
        chain_results.append(
            {"equation": equation, "outcome": outcome, "score": score, "seconds": round(elapsed, 3), "deadline": deadline}
        )

    search = search_regressions(compute_main, compute_pr)

    ok = all(result["score_on_main"] == result["expected"] for result in issue_results)
    ok = ok and not search["regressions"]
    report = {
        "snapshot": SNAPSHOT,
        "issue_945_reproduced": issue_results,
        "exponent_chain_cost": chain_results,
        "pr_946_regression_search": search,
        "all_claims_hold": ok,
    }

    if "--json" in sys.argv:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("(1) main 4e828c8: compute_score on issue #945's examples")
        for result in issue_results:
            verdict = "as claimed" if result["score_on_main"] == result["expected"] else "MISMATCH"
            print(
                f"    {result['equation']!r:18} numbers={result['numbers']} target={result['target']}"
                f" -> {result['score_on_main']}  ({verdict}: {result['note']})"
            )
        print("\n    cost of an exponent chain (per-case deadline, so nothing hangs):")
        for result in chain_results:
            if result["outcome"] == "timeout":
                print(f"    {result['equation']!r:14} -> still running after {result['deadline']}s")
            else:
                print(f"    {result['equation']!r:14} -> score={result['score']} in {result['seconds']}s")
        print("\n(2) PR #946: answers that score 1.0 on main and lose it after the patch")
        print(
            f"    compared on {search['legal_answers_with_full_reward']} legal answers that score 1.0 on main"
        )
        for regression in search["regressions"]:
            scope = "rejected on sight" if regression["rejected_by_patch_rule_on_sight"] else "should have been kept"
            print(
                f"    REGRESSION [{scope}] {regression['equation']}  "
                f"{regression['before']} -> {regression['after']}"
            )
        if not search["regressions"]:
            print("    none found")
        print(f"\nall claims hold: {ok}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
