"""The noise floor and selection rule of the detection tuning pass.

Both were declared in docs/EXPERIMENTS.md ("Detection tuning pass, declared before
the first arm ran") before any arm ran; this module applies them mechanically.

- Noise floor: the largest three-seed standard deviation of the score over the arms,
  rounded up to a multiple of 0.005, never below 0.005.
- Selection: start from the baseline's settings and change a factor to another level
  only if the highest-mean arm has that level and the arm that differs from the
  baseline in that factor alone beats the baseline's mean by more than the floor.
  If the best arm itself is within the floor of the baseline, no single change can
  beat it either, so the baseline is kept. Finally, if the arm this gives has a mean
  more than the floor below the highest-mean arm's, the highest-mean arm is taken
  instead (an interaction, e.g. only the combination beating the baseline, would
  otherwise keep an arm clearly worse than the best). Differences are compared at
  9 decimals, so a difference equal to the floor does not beat it because of
  floating point.
"""
from __future__ import annotations

import math


def noise_floor(stds, step: float = 0.005) -> float:
    return round(max(step, math.ceil(round(max(stds) / step, 9)) * step), 9)


def select_arm(means: dict[str, float], arms: dict[str, dict], baseline: str,
               floor: float) -> str:
    """`arms` maps arm name to its overrides (the factors); returns the selected arm."""
    best = max(means, key=means.get)
    chosen = dict(arms[baseline])
    for k, v in arms[best].items():
        if v == arms[baseline][k]:
            continue
        alone = next(a for a, o in arms.items() if o == {**arms[baseline], k: v})
        if round(means[alone] - means[baseline], 9) > floor:
            chosen[k] = v
    selected = next(a for a, o in arms.items() if o == chosen)
    if round(means[best] - means[selected], 9) > floor:
        return best
    return selected
