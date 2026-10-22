"""The tuning pass's noise floor and selection rule, as declared in docs/EXPERIMENTS.md
("Detection tuning pass, declared before the first arm ran"), on hand-made means."""
from __future__ import annotations

import pytest

from solar_inspect.detection.tuning import noise_floor, select_arm

ARMS = {
    "yolov8n-hsv0.4": {"model": "yolov8n.pt", "hsv_v": 0.4},
    "yolov8n-hsv0.0": {"model": "yolov8n.pt", "hsv_v": 0.0},
    "yolov8s-hsv0.4": {"model": "yolov8s.pt", "hsv_v": 0.4},
    "yolov8s-hsv0.0": {"model": "yolov8s.pt", "hsv_v": 0.0},
}
BASE = "yolov8n-hsv0.4"


def means(n4, n0, s4, s0):
    return dict(zip(ARMS, (n4, n0, s4, s0)))


@pytest.mark.parametrize("stds, floor", [
    ([0.0031, 0.0042, 0.0012, 0.0], 0.005),   # largest 0.0042 -> next multiple 0.005
    ([0.0051, 0.001], 0.010),                 # 0.0051 -> 0.010
    ([0.010, 0.002], 0.010),                  # already a multiple: stays
    ([0.0, 0.0], 0.005),                      # never below 0.005
])
def test_noise_floor_is_the_largest_std_rounded_up_to_0_005(stds, floor):
    assert noise_floor(stds) == pytest.approx(floor)


def test_a_factor_is_adopted_only_if_its_single_change_beats_the_baseline():
    # best is yolov8s-hsv0.0 (0.842). model alone: 0.840 - 0.820 = 0.020 > 0.005, adopted.
    # hsv_v alone: 0.823 - 0.820 = 0.003, not adopted. -> yolov8s at hsv_v 0.4
    assert select_arm(means(0.820, 0.823, 0.840, 0.842), ARMS, BASE, 0.005) == "yolov8s-hsv0.4"


def test_best_arm_inside_the_floor_keeps_the_baseline():
    # best 0.8249 - 0.820 = 0.0049 <= 0.005
    assert select_arm(means(0.820, 0.822, 0.824, 0.8249), ARMS, BASE, 0.005) == BASE


def test_a_difference_exactly_at_the_floor_does_not_beat_it():
    # 0.825 - 0.820 is 0.005 (0.0050000000000000044 in floating point)
    assert select_arm(means(0.820, 0.810, 0.825, 0.810), ARMS, BASE, 0.005) == BASE


def test_a_level_the_best_arm_does_not_have_is_never_adopted():
    # both single changes beat the baseline, but the best arm is yolov8s-hsv0.4,
    # so hsv_v stays at 0.4 even though yolov8n-hsv0.0 beat the baseline
    assert select_arm(means(0.820, 0.830, 0.850, 0.835), ARMS, BASE, 0.005) == "yolov8s-hsv0.4"


def test_both_changes_adopted_when_both_single_changes_beat_the_baseline():
    assert select_arm(means(0.820, 0.830, 0.840, 0.850), ARMS, BASE, 0.005) == "yolov8s-hsv0.0"


def test_only_the_combination_beating_the_baseline_selects_the_combination():
    # neither single change beats the baseline by more than the floor, so the factor
    # loop keeps the baseline; the best arm beats it by 0.020 > 0.005, so the final
    # clause takes the best arm
    assert select_arm(means(0.820, 0.823, 0.823, 0.840), ARMS, BASE, 0.005) == "yolov8s-hsv0.0"


def test_an_adopted_single_change_gives_way_to_a_best_arm_beyond_the_floor():
    # model alone: 0.840 - 0.820 = 0.020, adopted; hsv_v alone: 0.003, not adopted.
    # The loop gives yolov8s-hsv0.4 (0.840); the best arm (0.850) beats it by 0.010 > 0.005
    assert select_arm(means(0.820, 0.823, 0.840, 0.850), ARMS, BASE, 0.005) == "yolov8s-hsv0.0"


def test_the_final_clause_also_needs_more_than_the_floor():
    # loop gives yolov8s-hsv0.4 (0.840); best 0.845 is exactly 0.005 above it: not taken
    assert select_arm(means(0.820, 0.823, 0.840, 0.845), ARMS, BASE, 0.005) == "yolov8s-hsv0.4"
