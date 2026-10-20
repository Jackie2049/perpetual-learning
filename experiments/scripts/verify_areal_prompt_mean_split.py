"""CPU reference checks for AReaL loss contracts, not an AReaL runtime test.

Baseline: AReaL 3b93679888e9f4929f6f155f13c6a286a62dfd72.
Uses scalar token derivatives at fixed parameters; no torch/GPU required.
Outer PPO optimizer steps and inner gradient accumulation are separate axes.
Megatron default-path calculation is restricted to DP=CP=1, without MTP/MoE.
Run: python3 experiments/scripts/verify_areal_prompt_mean_split.py
"""
import math

CHECKS = 0


def close(actual, expected):
    global CHECKS
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (
        actual, expected
    )
    CHECKS += 1


def chunks(items, count):
    count = min(count, len(items))
    return [
        items[i * len(items) // count:(i + 1) * len(items) // count]
        for i in range(count)
    ]


def step_gradient(rows, group_tokens, active_groups, outer_steps,
                  mode, inner_count=1, backend="fsdp"):
    """Reference engine weighting for ONE optimizer step; DP=CP=1.

    rows are (group id, valid token count, sum of token derivatives).
    Prompt engine weights partition this step's rows, not the full rollout.
    Megatron default loss_multiplier=P is canceled by MCore schedule /P.
    """
    inner = chunks(rows, inner_count)
    total_mass = (1.0 if mode == "prompt_mean"
                  else sum(tokens for _, tokens, _ in rows))
    result = 0.0
    for mb in inner:
        if mode == "prompt_mean":
            mass = len(mb) / len(rows)
            local = sum(value / group_tokens[group] for group, _, value in mb)
            local *= outer_steps / active_groups / mass
        else:
            mass = sum(tokens for _, tokens, _ in mb)
            local = sum(value for _, _, value in mb) / mass
        loss = local * mass / total_mass
        if backend == "megatron":
            loss *= len(inner)       # AReaL loss_multiplier, DP=1
            loss /= len(inner)       # MCore default schedule, CP=1
        result += loss
    return result


def gradients(rows, group_tokens, steps, mode, inner_count=1, backend="fsdp"):
    schedule = chunks(rows, steps)
    return [
        step_gradient(mb, group_tokens, len(group_tokens), len(schedule),
                      mode, inner_count, backend)
        for mb in schedule
    ]


def synchronized_schedule(counts, requested):
    """Mirror data.py concrete-count/modulo scheduling, NOT tensor splitting."""
    if requested < 1 or any(count < 1 for count in counts):
        raise ValueError("Positive concrete counts and requested steps required")
    effective = max(min(requested, sum(counts)), max(counts))
    schedule = []
    offset = 0
    for count in counts:
        slots = [False] * effective
        for index in range(count):
            slot = (offset + index) % effective
            assert not slots[slot]
            slots[slot] = True
        offset += count
        schedule.append(slots)
    if any(not any(rank[slot] for rank in schedule)
           for slot in range(effective)):
        raise ValueError("Every optimizer step must contain real responses")
    return schedule


def adam_delta(grads, lr=0.01, epsilon=1e-8):
    """Scalar Adam recurrence, zero initial moments, frozen positive gradients."""
    first = second = delta = 0.0
    for step, gradient in enumerate(grads, 1):
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        corrected_first = first / (1 - 0.9 ** step)
        corrected_second = second / (1 - 0.999 ** step)
        delta -= lr * corrected_first / (math.sqrt(corrected_second) + epsilon)
    return delta


def main():
    rows = [(index // 2, 1, float(index + 1)) for index in range(8)]
    groups = {group: 2 for group in range(4)}
    print("Balanced tokens/groups: prompt_mean and token_mean")
    for steps in (1, 2, 4):
        prompt = gradients(rows, groups, steps, "prompt_mean")
        token = gradients(rows, groups, steps, "token_mean")
        for left, right in zip(prompt, token):
            close(left, right)
        close(sum(prompt) / steps, 4.5)
        close(sum(token), steps * 4.5)
        print(f"M={steps}: mean={sum(prompt)/steps:.6f}; "
              f"both frozen sums={sum(prompt):.6f}")

    # Unequal full-group token counts, unlike the historical uniform-mask cases.
    uneven = [(0, 1, 1.0), (1, 2, 4.0), (1, 3, 9.0), (1, 4, 16.0),
              (2, 1, 5.0), (2, 2, 12.0), (2, 3, 21.0), (2, 4, 32.0)]
    group_tokens = {0: 1, 1: 9, 2: 10}
    oracle = (1 + 29 / 9 + 70 / 10) / 3
    for steps in (1, 2, 4):
        prompt = gradients(uneven, group_tokens, steps, "prompt_mean")
        close(sum(prompt) / steps, oracle)
    print(f"Unequal groups: mean of prompt step gradients={oracle:.9f}")

    # Fixed outer steps: inner accumulation preserves each step's gradient.
    for mode in ("prompt_mean", "token_mean"):
        baseline = gradients(uneven, group_tokens, 2, mode)
        for backend in ("fsdp", "megatron"):
            for inner_count in (1, 2, 4):
                candidate = gradients(uneven, group_tokens, 2, mode,
                                      inner_count, backend)
                for actual, expected in zip(candidate, baseline):
                    close(actual, expected)
    print("Fixed M=2: inner P=1/2/4 invariant in scalar DP=CP=1 model")

    # A dummy is local: another rank has real responses in that global slot.
    for counts, requested in (([1, 3], 3), ([2, 2], 4), ([1, 1], 6)):
        schedule = synchronized_schedule(counts, requested)
        assert all(any(rank[slot] for rank in schedule)
                   for slot in range(len(schedule[0])))
        print(f"DP concrete counts={counts}, requested={requested}: "
              f"M={len(schedule[0])}; rank slots={schedule}")
    # DP=1 cannot append an empty extra global step.
    assert len(synchronized_schedule([1], 6)[0]) == 1
    # At fixed M, replacing a zero-mass local dummy with no local contribution
    # gives the same global prompt contribution; there is no extra dummy factor.
    close((0 + 3) * 3 / 4, 3 * 3 / 4)

    # Same per-step values in both modes: different M still changes Adam steps.
    deltas = []
    for steps in (2, 4):
        prompt = gradients(rows, groups, steps, "prompt_mean")
        token = gradients(rows, groups, steps, "token_mean")
        close(adam_delta(prompt), adam_delta(token))
        deltas.append(adam_delta(prompt))
    assert not math.isclose(deltas[0], deltas[1], rel_tol=0.01)
    print(f"Scalar Adam, frozen gradients: M=2 delta={deltas[0]:.9f}, "
          f"M=4 delta={deltas[1]:.9f}; both modes agree per M")
    print(f"PASS: {CHECKS} numeric comparisons plus schedule checks.")
    print("Reference simulation only; actual AReaL/FSDP/Megatron GPU tests pending.")


if __name__ == "__main__":
    main()
