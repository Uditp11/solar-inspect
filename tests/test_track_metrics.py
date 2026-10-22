"""Tracking metrics on hand-made sequences, tested before the module exists.

A sequence is a list of `Frame`s: visible ground truth (ids, xyxy boxes), ignore boxes,
and the tracker's hypotheses (reported ids, xyxy boxes). Every expected value below is
worked out by hand in the comment beside it.
"""
from __future__ import annotations

import numpy as np
import pytest

from solar_inspect.tracking.metrics import (Frame, clear_mot_matches, count_error,
                                            id_switches, ignore_filter, score)

A = [0, 0, 10, 10]
B = [20, 0, 30, 10]
NONE = np.zeros((0, 4))


def frame(gt=(), ign=(), hyp=()):
    """gt and hyp are lists of (id, box); ign a list of boxes."""
    def split(pairs):
        ids = np.array([i for i, _ in pairs], dtype=np.int64)
        boxes = np.array([b for _, b in pairs], dtype=float).reshape(-1, 4)
        return ids, boxes
    g_ids, g_boxes = split(gt)
    h_ids, h_boxes = split(hyp)
    return Frame(g_ids, g_boxes, np.array(ign, dtype=float).reshape(-1, 4), h_ids, h_boxes)


def test_perfect_tracker():
    # Two panels, three frames, every hypothesis exactly on its panel with one id each:
    # 6 matches, no FP, FN or switch -> MOTA 1, IDF1 1, IDSW 0, K = G = 2.
    seq = [frame(gt=[(1, A), (2, B)], hyp=[(7, A), (9, B)]) for _ in range(3)]
    s = score({"s": seq})["s"]
    assert (s["mota"], s["idf1"]) == (1.0, 1.0)
    assert (s["idsw"], s["idsw_mm"], s["fp"], s["fn"], s["dropped"]) == (0, 0, 0, 0, 0)
    assert (s["K"], s["G"], s["K_minus_G"], s["rel_count_error"]) == (2, 2, 0, 0.0)


def test_one_switch():
    # One panel for 4 frames; track 5 covers frames 0-1, track 6 frames 2-3.
    # Matches 5, 5, 6, 6 -> one switch. MOTA = 1 - (FN 0 + FP 0 + IDSW 1) / 4 = 0.75.
    # IDF1: the panel's trajectory is paired with one track (2 frames): IDTP 2, IDFN 2,
    # IDFP 2 -> 2 * 2 / (2 * 2 + 2 + 2) = 0.5. K = 2 ids, G = 1 -> K - G = 1, ratio 1.
    seq = [frame(gt=[(1, A)], hyp=[(5 if t < 2 else 6, A)]) for t in range(4)]
    s = score({"s": seq})["s"]
    assert (s["idsw"], s["idsw_mm"]) == (1, 1)
    assert s["mota"] == pytest.approx(0.75)
    assert s["idf1"] == pytest.approx(0.5)
    assert (s["K"], s["G"], s["K_minus_G"], s["rel_count_error"]) == (2, 1, 1, 1.0)


def test_id_switches_counts_against_the_last_matched_id():
    # Panel 1 matched to 5, 5, 6, (unmatched), 5: 5 -> 6 and 6 -> 5, 2 switches.
    # Panel 2 matched to 8, unmatched for 3 frames, 8 again: a gap is not a switch.
    matches = [[(1, 5), (2, 8)], [(1, 5)], [(1, 6)], [], [(1, 5), (2, 8)]]
    assert id_switches(matches) == 2


def test_last_correspondence_is_kept_over_a_better_overlap():
    # CLEAR MOT: frame 1 offers track 6 at IoU 1 and the panel's last track 5 at IoU
    # 0.6 (10 x 6 inside 10 x 10). The last correspondence is kept while it reaches 0.5,
    # so no switch, and track 6 is a false positive. A plain per-frame assignment would
    # take track 6 and count a switch.
    seq = [frame(gt=[(1, A)], hyp=[(5, A)]),
           frame(gt=[(1, A)], hyp=[(5, [0, 0, 10, 6]), (6, A)])]
    assert clear_mot_matches(seq) == [[(1, 5)], [(1, 5)]]
    s = score({"s": seq})["s"]
    assert (s["idsw"], s["idsw_mm"], s["fp"], s["fn"]) == (0, 0, 1, 0)


def test_hypothesis_on_an_ignore_box_is_dropped_not_counted():
    # Visible A, ignore box B. Track 7 on A, track 9 exactly on B (IoU 0 with A), track
    # 11 on [20, 0, 24, 10] (IoU 0.4 with B, 0 with A).
    # Track 9 is dropped (1 hypothesis); track 11 does not reach 0.5 with B, so it stays
    # and is a false positive. B is never a miss. K counts the ids that survive: 7, 11.
    seq = [frame(gt=[(1, A)], ign=[B], hyp=[(7, A), (9, B), (11, [20, 0, 24, 10])])]
    assert ignore_filter(seq[0].gt_boxes, seq[0].ign_boxes, seq[0].hyp_boxes).tolist() \
        == [True, False, True]
    s = score({"s": seq})["s"]
    assert (s["dropped"], s["fp"], s["fn"], s["mota"]) == (1, 1, 0, 0.0)
    assert (s["K"], s["G"]) == (2, 1)


def test_hypothesis_matching_a_visible_box_is_kept_despite_an_ignore_box():
    # Visible A, ignore box [0, 0, 10, 8] over the same panel. Track 7 exactly on A;
    # track 9 on [0, 0, 10, 9]: IoU 0.9 with A and 8/9 with the ignore box. It reaches
    # 0.5 with a visible box, so it is kept. A is taken by track 7, so track 9 is a
    # false positive (a duplicate). Nothing is dropped; K = 2.
    seq = [frame(gt=[(1, A)], ign=[[0, 0, 10, 8]], hyp=[(7, A), (9, [0, 0, 10, 9])])]
    assert ignore_filter(seq[0].gt_boxes, seq[0].ign_boxes, seq[0].hyp_boxes).tolist() \
        == [True, True]
    s = score({"s": seq})["s"]
    assert (s["dropped"], s["fp"], s["K"]) == (0, 1, 2)


def test_count_error_and_overall():
    # Sequence a: G = 1 (panel 1), K = 3 (tracks 1, 2, 3 on it in turn).
    # Sequence b: G = 2, K = 1 (one track on panel 1, panel 2 never covered).
    a = [frame(gt=[(1, A)], hyp=[(t + 1, A)]) for t in range(3)]
    b = [frame(gt=[(1, A), (2, B)], hyp=[(1, A)])]
    assert count_error(a) == {"K": 3, "G": 1, "K_minus_G": 2, "rel_count_error": 2.0}
    assert count_error(b) == {"K": 1, "G": 2, "K_minus_G": -1, "rel_count_error": -0.5}
    # Overall sums the counts: 5 GT boxes, 2 switches (a), 1 miss (b), 0 FP ->
    # MOTA = 1 - 3 / 5 = 0.4; K = 4, G = 3.
    out = score({"a": a, "b": b})
    assert out["overall"]["mota"] == pytest.approx(0.4)
    assert (out["overall"]["idsw"], out["overall"]["idsw_mm"]) == (2, 2)
    assert (out["overall"]["K"], out["overall"]["G"], out["overall"]["K_minus_G"]) == (4, 3, 1)


def test_empty_frames():
    # A frame with no ground truth and no hypothesis, and one with only a hypothesis.
    seq = [frame(), frame(hyp=[(3, A)])]
    s = score({"s": seq})["s"]
    assert (s["fp"], s["fn"], s["idsw"], s["K"], s["G"]) == (1, 0, 0, 1, 0)
