"""SORT, the greedy-IoU baseline and the no-tracker floor, tested before they exist.

Every tracker returns, per frame, one id per input detection: the id of the track
that detection belongs to, or -1 while that track is not yet reported. Track
management is SORT's (Bewley et al. 2016) for both trackers:

- a detection the association leaves unmatched starts a track; ids count from 1;
- a track's hit streak counts consecutive matches (its birth is not a match), and a
  track is reported once the streak reaches min_hits, or at any time during the
  first min_hits frames;
- a track not matched for more than max_age frames is deleted.

Association matrices below are (detections, tracks).
"""
from __future__ import annotations

import numpy as np
import pytest

from solar_inspect.tracking.sort import (GreedyIoU, NoTracker, Sort, greedy_match,
                                         hungarian_match)

TRACKERS = [Sort, GreedyIoU]


def run(tracker, frames):
    return [tracker.update(np.asarray(f, dtype=float).reshape(-1, 4)).tolist()
            for f in frames]


# --- association --------------------------------------------------------------------

def test_rectangular_more_detections_than_tracks():
    # Hungarian maximises total IoU over one-to-one pairs:
    #   d0-t1 + d1-t0 = 0.7 + 0.6 = 1.3   (best)
    #   d0-t1 + d2-t0 = 0.7 + 0.5 = 1.2
    #   d1-t0 + d2-t1 = 0.6 + 0.4 = 1.0, and the rest lower.
    # Greedy takes 0.7 (d0-t1), then the best left in t0's column, 0.6 (d1-t0).
    m = np.array([[0.1, 0.7],
                  [0.6, 0.0],
                  [0.5, 0.4]])
    for match in (hungarian_match, greedy_match):
        pairs, um_det, um_trk = match(m, 0.3)
        assert sorted(pairs) == [(0, 1), (1, 0)]
        assert (um_det, um_trk) == ([2], [])


def test_rectangular_more_tracks_than_detections():
    m = np.array([[0.1, 0.7],
                  [0.6, 0.0],
                  [0.5, 0.4]]).T         # same numbers, (2 detections, 3 tracks)
    for match in (hungarian_match, greedy_match):
        pairs, um_det, um_trk = match(m, 0.3)
        assert sorted(pairs) == [(0, 1), (1, 0)]
        assert (um_det, um_trk) == ([], [2])


def test_a_pair_below_the_gate_is_not_matched_though_the_assignment_pairs_it():
    # 1 x 1: any assignment must pair d0 with t0. IoU 0.25 < 0.3, so the gate, applied
    # after the assignment, drops it: the detection is unmatched and so is the track.
    pairs, um_det, um_trk = hungarian_match(np.array([[0.25]]), 0.3)
    assert (pairs, um_det, um_trk) == ([], [0], [0])
    # At exactly the gate a pair is kept (SORT drops only IoU < gate).
    assert hungarian_match(np.array([[0.3]]), 0.3)[0] == [(0, 0)]
    assert greedy_match(np.array([[0.25]]), 0.3) == ([], [0], [0])
    assert greedy_match(np.array([[0.3]]), 0.3)[0] == [(0, 0)]


def test_hungarian_and_greedy_differ_where_they_should():
    #            t0   t1
    #   d0      0.9  0.8
    #   d1      0.8  0.1
    # Hungarian: d0-t1 + d1-t0 = 1.6 beats d0-t0 + d1-t1 = 1.0.
    # Greedy: takes 0.9 (d0-t0) first, leaving d1-t1 = 0.1, below the gate.
    m = np.array([[0.9, 0.8], [0.8, 0.1]])
    assert sorted(hungarian_match(m, 0.3)[0]) == [(0, 1), (1, 0)]
    assert greedy_match(m, 0.3) == ([(0, 0)], [1], [1])


def test_empty_matrices():
    for match in (hungarian_match, greedy_match):
        assert match(np.zeros((0, 2)), 0.3) == ([], [], [0, 1])
        assert match(np.zeros((3, 0)), 0.3) == ([], [0, 1, 2], [])


# --- whole trackers on toy sequences -------------------------------------------------

def crossing():
    """Two 20 x 20 boxes moving 10 px per frame towards each other along x, 5 px apart
    in y, crossing at frame 5. Detections are listed [A, B] in every frame.

        frame   A                  B
        0       [0, 0, 20, 20]     [100, 5, 120, 25]
        5       [50, 0, 70, 20]    [50, 5, 70, 25]      A vs B: 20 x 15 = 300 / 500 -> 0.6
        10      [100, 0, 120, 20]  [0, 5, 20, 25]

    A 10 px step on a 20 px box is IoU 200 / 600 = 0.333 frame to frame, just above
    the gate, so a tracker that does not follow both boxes loses them.
    """
    return [[[10 * t, 0, 10 * t + 20, 20], [100 - 10 * t, 5, 120 - 10 * t, 25]]
            for t in range(11)]


@pytest.mark.parametrize("cls", TRACKERS)
def test_two_crossing_tracks_keep_their_ids(cls):
    # min_hits 3: frames 0-2 are reported under the start-up rule, and from frame 3
    # each track has 3 consecutive matches, so every frame reports [1, 2].
    assert run(cls(), crossing()) == [[1, 2]] * 11


BOX = [0, 0, 20, 20]


def with_gap(k: int, before: int = 4, after: int = 3):
    """A stationary box, present for `before` frames, absent for k, present `after`."""
    return [[BOX]] * before + [[]] * k + [[BOX]] * after


@pytest.mark.parametrize("cls", TRACKERS)
def test_a_gap_of_max_age_frames_keeps_the_id(cls):
    # max_age 2, min_hits 1, gap k = 2 (frames 4, 5).
    # frame 4: not matched, 1 frame unmatched; frame 5: 2, not > 2, kept;
    # frame 6: matched again -> id 1, streak restarts at 1 = min_hits -> reported.
    out = run(cls(max_age=2, min_hits=1), with_gap(2))
    assert out == [[1], [1], [1], [1], [], [], [1], [1], [1]]


@pytest.mark.parametrize("cls", TRACKERS)
def test_a_gap_longer_than_max_age_gives_a_new_id(cls):
    # max_age 2, min_hits 1, gap k = 3 (frames 4, 5, 6): at frame 6 the track has gone
    # 3 > 2 frames unmatched and is deleted. Frame 7 starts track 2: born, streak 0,
    # past the first min_hits frames, so not yet reported (-1); frame 8: streak 1.
    out = run(cls(max_age=2, min_hits=1), with_gap(3))
    assert out == [[1], [1], [1], [1], [], [], [], [-1], [2], [2]]


@pytest.mark.parametrize("cls", TRACKERS)
def test_the_gate_starts_a_new_track_through_the_whole_tracker(cls):
    # Stationary [0, 0, 20, 20] for 3 frames, then the detection jumps to
    # [12, 0, 32, 20]: IoU 8 x 20 = 160 / (400 + 400 - 160) = 0.25 < 0.3 -> new id 2.
    # A jump to [10, 0, 30, 20] is IoU 200 / 600 = 0.333 -> same id 1.
    # min_hits 0, so every track is reported from birth.
    far = [[BOX]] * 3 + [[[12, 0, 32, 20]]]
    near = [[BOX]] * 3 + [[[10, 0, 30, 20]]]
    assert run(cls(min_hits=0), far) == [[1], [1], [1], [2]]
    assert run(cls(min_hits=0), near) == [[1], [1], [1], [1]]


@pytest.mark.parametrize("cls", TRACKERS)
def test_min_hits_holds_a_new_track_back(cls):
    # Defaults: max_age 1, min_hits 3. A box appears at frame 4 (frame_count 5 > 3):
    # born (streak 0), then streaks 1, 2, 3 -> reported from its 4th detection.
    frames = [[]] * 4 + [[BOX]] * 5
    assert run(cls(), frames) == [[]] * 4 + [[-1], [-1], [-1], [1], [1]]


def test_the_kalman_filter_carries_a_moving_box_through_a_gap_greedy_cannot():
    # A 20 px box moving 7 px per frame, missing at frame 3. Frame to frame the IoU is
    # 13 x 20 = 260 / 540 = 0.48, so both trackers follow it at first. At frame 4 greedy
    # compares the detection with the box from frame 2, 14 px back: IoU 6 x 20 = 120 /
    # 680 = 0.18 < 0.3 -> new id 2 (and track 1, 2 > max_age frames unmatched, is
    # deleted). SORT has fitted a velocity of about 7 px per frame and predicts the box
    # about 14 px on, IoU near 1 -> same id. min_hits 0, so every track is reported.
    frames = [[[7 * t, 0, 7 * t + 20, 20]] for t in range(6)]
    frames[3] = []                                                    # one-frame gap
    sort = run(Sort(max_age=1, min_hits=0), frames)
    greedy = run(GreedyIoU(max_age=1, min_hits=0), frames)
    assert sort == [[1], [1], [1], [], [1], [1]]
    assert greedy == [[1], [1], [1], [], [2], [2]]


def test_no_tracker_gives_every_detection_a_new_id():
    assert run(NoTracker(), [[BOX, BOX], [], [BOX], [BOX, BOX, BOX]]) == \
        [[1, 2], [], [3], [4, 5, 6]]
