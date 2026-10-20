#!/usr/bin/env python3
"""
#8281 verification: agent-loop shared prompt cap vs data.truncation policy.

Extracts the verbatim logic of verl's ``AgentLoopBase._cap_text_prompt_length``
(verl/experimental/agent_loop/agent_loop.py:467) and reproduces the issue's
repro: an overlong prompt is left-truncated to the same suffix under every
``data.truncation`` policy, even ``error``.

Pre-fix  : four policies all return the SAME suffix ``[6,7,8,9]`` and log a
           left-truncation warning, ignoring the documented contract
           (docs/examples/config.rst:76-81).
Post-fix : the cap honors the policy - ``error`` raises, ``left`` keeps the
           suffix, ``right`` keeps the prefix, ``middle`` keeps head+tail.

Reference implementation (multiturn_sft_dataset.py:380-393) is mirrored for
the error/left/right semantics; ``middle`` follows the documented
"keeps head and tail" wording.
"""

import re

# --- verbatim current implementation (agent_loop.py:467-476, logs elided) ---
def cap_current(prompt_ids, prompt_length):
    if len(prompt_ids) > prompt_length:
        return prompt_ids[-prompt_length:]
    return prompt_ids


# --- proposed fix: honor data.truncation (default 'error' matches rl_dataset.py:118) ---
def cap_fixed(prompt_ids, prompt_length, truncation="error"):
    if len(prompt_ids) <= prompt_length:
        return prompt_ids
    if truncation == "left":
        return prompt_ids[-prompt_length:]
    if truncation == "right":
        return prompt_ids[:prompt_length]
    if truncation == "middle":
        head = (prompt_length + 1) // 2
        return prompt_ids[:head] + prompt_ids[-(prompt_length - head):]
    # 'error' (default) - fail loudly, mirroring the multimodal ValueError path
    raise ValueError(
        f"Prompt of {len(prompt_ids)} tokens exceeds rollout.prompt_length={prompt_length}; "
        f"configured data.truncation='error'. Increase rollout.prompt_length or set a "
        f"non-'error' data.truncation."
    )


def main():
    prompt_ids = list(range(10))  # 10 tokens, same as the issue repro
    prompt_length = 4

    print("=== Pre-fix: unconditional left-truncation under every policy ===\n")
    results = {}
    for policy in ("error", "left", "right", "middle"):
        out = cap_current(prompt_ids, prompt_length)
        results[policy] = out
        print(f"  {policy:<7} -> {out}")
    all_same = len({tuple(v) for v in results.values()}) == 1
    print(f"\n  all four policies identical suffix: {all_same}  (BUG reproduced)")

    assert all_same, "expected all policies to left-truncate to the same suffix"

    print("\n=== Post-fix: policy honored ===\n")

    # left / right / middle truncate deterministically
    exp_left = prompt_ids[-4:]  # [6,7,8,9]
    exp_right = prompt_ids[:4]  # [0,1,2,3]
    exp_middle = prompt_ids[:2] + prompt_ids[-2:]  # head 2 + tail 2 = [0,1,8,9]

    got_left = cap_fixed(prompt_ids, prompt_length, "left")
    got_right = cap_fixed(prompt_ids, prompt_length, "right")
    got_middle = cap_fixed(prompt_ids, prompt_length, "middle")

    print(f"  left   -> {got_left}   expected {exp_left}")
    print(f"  right  -> {got_right}   expected {exp_right}")
    print(f"  middle -> {got_middle}   expected {exp_middle}")

    # 'error' must raise instead of silently truncating
    try:
        cap_fixed(prompt_ids, prompt_length, "error")
        error_raises = False
    except ValueError as e:
        error_raises = True
        msg = str(e)
    print(f"  error  -> raises ValueError: {error_raises}")
    print(f"           message: {msg!r}")

    # short prompt (not exceeding cap) is returned unchanged under all policies
    short = list(range(3))
    unchanged = all(cap_fixed(short, prompt_length, p) == short for p in ("error", "left", "right", "middle"))
    print(f"\n  short prompt unchanged under all policies: {unchanged}")

    # middle split sanity: output length == cap, head + tail, no overlap for len 10 cap 4
    assert len(exp_middle) == 4
    assert exp_middle == [0, 1, 8, 9]
    assert exp_left == [6, 7, 8, 9]
    assert exp_right == [0, 1, 2, 3]
    assert error_raises
    assert unchanged
    assert got_left == exp_left and got_right == exp_right and got_middle == exp_middle

    # middle on an even/odd cap keeps correct total length
    for cap, pid in ((5, list(range(10))), (6, list(range(10)))):
        m = cap_fixed(pid, cap, "middle")
        assert len(m) == cap, f"middle cap={cap} gave len {len(m)}"

    print("\n  middle length invariant holds for cap 5 and 6")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
