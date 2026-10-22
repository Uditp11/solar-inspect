"""SORT and the two baselines it is compared with, on per-frame detection boxes.

- `Sort`: Bewley et al. 2016. Each track carries the constant-velocity Kalman filter
  of `kalman.py`; per frame every track is predicted, the cost is 1 - IoU between
  detections and predicted boxes, and `scipy.optimize.linear_sum_assignment` solves
  the (rectangular) assignment.
- `GreedyIoU`: no motion model. A track's prediction is its last matched box, and
  pairs are taken greedily by highest IoU. The baseline that says whether the Kalman
  filter earns its place.
- `NoTracker`: every detection is a new identity. The count-error floor.

The IoU gate is applied **after** assignment: pairs below `iou_gate` are dropped and
both sides become unmatched. Folding the gate into the cost (say, infinite cost below
it) would make a real but poor pair look the same as the padding of a rectangular
matrix.

One difference from SORT's reference code: it skips the assignment when every
detection and every track has at most one partner above the gate, and takes those
pairs directly. That is not reproduced. The two can differ: with IoU d0-t0 0.35,
d0-t1 0.29, d1-t0 0.29, d1-t1 0, the assignment prefers d0-t1 + d1-t0 (0.58) over
d0-t0 (0.35), and the gate then drops both, where the shortcut would keep d0-t0.

Track management, the same for both trackers and as in SORT: an unmatched detection
starts a track (ids count from 1); a track's hit streak counts consecutive matches,
its birth not included; a track is reported once its streak reaches `min_hits`, or
at any time in the first `min_hits` frames; a track unmatched for more than `max_age`
frames is deleted. `update()` returns, per detection, the id of its track, or -1
while that track is not reported: the value a `Finding.track_id` would carry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from solar_inspect.detection.boxes import iou
from solar_inspect.tracking import kalman

Match = tuple[list[tuple[int, int]], list[int], list[int]]


def _unmatched(n_det: int, n_trk: int, pairs: list[tuple[int, int]]) -> Match:
    d_used = {d for d, _ in pairs}
    t_used = {t for _, t in pairs}
    return (pairs, [d for d in range(n_det) if d not in d_used],
            [t for t in range(n_trk) if t not in t_used])


def hungarian_match(iou_m: np.ndarray, gate: float) -> Match:
    """Minimum-cost one-to-one assignment on 1 - IoU (detections x tracks), then the
    gate: pairs with IoU < gate are dropped. Returns (pairs, unmatched detections,
    unmatched tracks)."""
    n_det, n_trk = iou_m.shape
    pairs = []
    if n_det and n_trk:
        rows, cols = linear_sum_assignment(1.0 - iou_m)
        pairs = [(int(d), int(t)) for d, t in zip(rows, cols) if iou_m[d, t] >= gate]
    return _unmatched(n_det, n_trk, pairs)


def greedy_match(iou_m: np.ndarray, gate: float) -> Match:
    """Repeatedly take the highest remaining IoU >= gate whose detection and track are
    both free. Ties go to the lower (detection, track) index, row-major."""
    n_det, n_trk = iou_m.shape
    pairs, d_used, t_used = [], set(), set()
    for k in np.argsort(-iou_m, axis=None, kind="stable"):
        d, t = divmod(int(k), n_trk)
        if iou_m[d, t] < gate:
            break
        if d not in d_used and t not in t_used:
            pairs.append((d, t))
            d_used.add(d)
            t_used.add(t)
    return _unmatched(n_det, n_trk, pairs)


@dataclass
class Track:
    id: int
    x: np.ndarray              # Kalman mean (Sort) or last matched box (GreedyIoU)
    P: np.ndarray | None = None
    hit_streak: int = 0
    time_since_update: int = 0


class Sort:
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_gate: float = 0.3,
                 Q: np.ndarray | None = None, R: np.ndarray | None = None):
        self.max_age, self.min_hits, self.iou_gate = max_age, min_hits, iou_gate
        self.Q = kalman.default_Q() if Q is None else Q
        self.R = kalman.default_R() if R is None else R
        self.tracks: list[Track] = []
        self.frame_count = 0
        self._next_id = 1

    # The three places the trackers differ.
    def _predict(self, t: Track) -> np.ndarray:
        t.x, t.P = kalman.predict(t.x, t.P, self.Q)
        return kalman.z_to_box(t.x)

    def _correct(self, t: Track, box: np.ndarray) -> None:
        t.x, t.P = kalman.update(t.x, t.P, kalman.box_to_z(box), self.R)

    def _new(self, box: np.ndarray) -> Track:
        x, P = kalman.initial_state(kalman.box_to_z(box))
        return Track(self._take_id(), x, P)

    _match = staticmethod(hungarian_match)

    def _take_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def update(self, dets: np.ndarray) -> np.ndarray:
        """One frame. `dets` (N, 4) xyxy; returns (N,) track ids, -1 = not reported."""
        dets = np.asarray(dets, dtype=float).reshape(-1, 4)
        self.frame_count += 1
        for t in self.tracks:
            if t.time_since_update > 0:      # missed last frame: the streak is broken
                t.hit_streak = 0
            t.time_since_update += 1
        pred = np.array([self._predict(t) for t in self.tracks]).reshape(-1, 4)
        pairs, um_det, _ = self._match(iou(dets, pred), self.iou_gate)

        owner: dict[int, Track] = {}
        for d, k in pairs:
            t = self.tracks[k]
            self._correct(t, dets[d])
            t.time_since_update = 0
            t.hit_streak += 1
            owner[d] = t
        for d in um_det:
            t = self._new(dets[d])
            self.tracks.append(t)
            owner[d] = t

        ids = np.full(len(dets), -1, dtype=np.int64)
        for d, t in owner.items():
            if t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                ids[d] = t.id
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return ids


class GreedyIoU(Sort):
    """SORT's track management with no motion model and greedy matching."""

    def _predict(self, t: Track) -> np.ndarray:
        return t.x

    def _correct(self, t: Track, box: np.ndarray) -> None:
        t.x = box.copy()

    def _new(self, box: np.ndarray) -> Track:
        return Track(self._take_id(), box.copy())

    _match = staticmethod(greedy_match)


class NoTracker:
    """Every detection is a new identity."""

    def __init__(self):
        self._next_id = 1

    def update(self, dets: np.ndarray) -> np.ndarray:
        n = len(np.asarray(dets).reshape(-1, 4))
        ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._next_id += n
        return ids
