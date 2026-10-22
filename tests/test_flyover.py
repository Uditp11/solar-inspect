"""The D5 flyover generator, tested on hand-made geometry and a synthetic image.

No D2 frame is read here. Coordinates are continuous (a box [x1, y1, x2, y2] has
area (x2 - x1)(y2 - y1)), the convention of `detection.boxes`.
"""
from __future__ import annotations

import numpy as np

from solar_inspect.tracking.flyover import (FlyoverParams, box_corners, hull, make_sequence,
                                            similarity, visibility, warp_points)

BOXES = np.array([[10, 20, 50, 45], [100, 100, 130, 160], [200, 50, 260, 80]], dtype=float)


def rect(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)


def test_hull_of_a_box_warped_by_the_identity_is_the_box():
    quads = warp_points(np.eye(3), box_corners(BOXES))
    np.testing.assert_allclose(hull(quads, (320, 256)), BOXES, rtol=0, atol=1e-12)


def test_a_pure_translation_moves_every_box_by_the_translation():
    # centre (160, 128) of a 320 x 256 window, no yaw, scale 1: p_w = p_s - c + (160, 128).
    # With c = (150, 118) every point moves by (+10, +10).
    H = similarity(centre=(150.0, 118.0), theta=0.0, scale=1.0, window=(320, 256))
    np.testing.assert_allclose(H, [[1, 0, 10], [0, 1, 10], [0, 0, 1]], rtol=0, atol=1e-12)
    moved = hull(warp_points(H, box_corners(BOXES)), (320, 256))
    np.testing.assert_allclose(moved, BOXES + 10, rtol=0, atol=1e-12)


def test_a_box_is_clipped_to_the_window():
    # [300, 240, 340, 270] in a 320 x 256 window -> [300, 240, 320, 256].
    quad = box_corners(np.array([[300, 240, 340, 270]], dtype=float))
    np.testing.assert_allclose(hull(quad, (320, 256)), [[300, 240, 320, 256]], atol=1e-12)


def test_rotation_by_90_degrees_about_the_window_centre():
    # Window 100 x 100, centre (50, 50), source centre (50, 50), theta = 90 deg, scale 1:
    # p_w = (50, 50) + R(-90) (p_s - (50, 50)), R(-90) (x, y) = (y, -x).
    # Box [60, 40, 80, 50]: offsets x 10..30, y -10..0 -> window x = 50 + y in 40..50,
    # y = 50 - x in 20..40. Hull [40, 20, 50, 40].
    H = similarity(centre=(50.0, 50.0), theta=np.pi / 2, scale=1.0, window=(100, 100))
    got = hull(warp_points(H, box_corners(np.array([[60, 40, 80, 50]], dtype=float))), (100, 100))
    np.testing.assert_allclose(got, [[40, 20, 50, 40]], rtol=0, atol=1e-9)


def test_the_occluder_hides_exactly_the_boxes_it_covers():
    # Window 100 x 100, occluder the strip x in [0, 40]. Each box is 20 x 20 = 400.
    #   A [10, 10, 30, 30]    inside, all 400 under the strip      -> visible 0     hidden
    #   B [35, 10, 55, 30]    inside, x 35..40 under: 100 of 400   -> visible 0.75  shown
    #   C [30, 40, 50, 60]    inside, x 30..40 under: 200 of 400   -> visible 0.50  shown (>= 0.5)
    #   D [60, 10, 80, 30]    inside, not under                    -> visible 1     shown
    #   E [90, 70, 110, 90]   half outside the window: 200 inside  -> inside 0.5    shown
    #   F [95, 70, 115, 90]   a quarter inside                     -> inside 0.25   not shown
    boxes = np.array([[10, 10, 30, 30], [35, 10, 55, 30], [30, 40, 50, 60],
                      [60, 10, 80, 30], [90, 70, 110, 90], [95, 70, 115, 90]], dtype=float)
    inside, visible = visibility(box_corners(boxes), (100, 100), rect(0, 0, 40, 100))
    np.testing.assert_allclose(inside, [1, 1, 1, 1, 0.5, 0.25], atol=1e-6)
    np.testing.assert_allclose(visible, [0, 0.75, 0.5, 1, 0.5, 0.25], atol=1e-6)
    shown = visible >= 0.5
    assert shown.tolist() == [False, True, True, True, True, False]
    # Occluded = would be shown but for the occluder: A only.
    assert ((inside >= 0.5) & ~shown).tolist() == [True, False, False, False, False, False]
    # No occluder: visibility is the inside fraction.
    np.testing.assert_allclose(visibility(box_corners(boxes), (100, 100), None)[1], inside)


def synthetic_source(seed: int = 0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (512, 640, 3), dtype=np.uint8)
    lt = rng.uniform(0, 600, (40, 2))
    return img, np.column_stack([lt, lt + rng.uniform(20, 40, (40, 2))])


def test_the_same_seed_gives_the_same_sequence_byte_for_byte():
    img, boxes = synthetic_source()
    p = FlyoverParams(frames=12)
    a, b, c = (make_sequence(img, boxes, p, s) for s in (3, 3, 4))
    for k in ("frames", "H", "gt_frame", "gt_id", "gt_box", "occ_frame", "occ_id"):
        assert a[k].tobytes() == b[k].tobytes(), k
    assert a["frames"].tobytes() != c["frames"].tobytes()


def test_a_sequence_stays_inside_the_source_and_inside_its_ranges():
    img, boxes = synthetic_source()
    p = FlyoverParams(frames=60, pan_speed=12.0)       # fast, so the path hits the bounds
    s = make_sequence(img, boxes, p, 7)
    assert s["frames"].shape == (60, 256, 320, 3)
    win = np.array([[0, 0], [320, 0], [320, 256], [0, 256]], dtype=float)
    for H in s["H"]:                                    # the window's footprint in the source
        foot = warp_points(np.linalg.inv(H), win)
        assert foot.min() >= 0 and foot[:, 0].max() <= 640 and foot[:, 1].max() <= 512
    lo, hi = p.scale_range
    assert lo - 1e-12 <= s["scale"].min() and s["scale"].max() <= hi + 1e-12
    assert {s["scale"][0], s["scale"][-1]} == {lo, hi}  # a ramp from one end to the other
    assert s["bounces"] > 0


def test_the_velocity_is_not_constant_between_bounces():
    # Slow and short, so most paths never reach a bound: every second difference of the
    # centre of such a path is an acceleration draw, and none is zero.
    img, boxes = synthetic_source()
    checked = 0
    for seed in range(10):
        s = make_sequence(img, boxes, FlyoverParams(frames=10, pan_speed=1.0), seed)
        if s["bounces"] == 0:
            assert np.abs(np.diff(s["centre"], 2, axis=0)).min() > 1e-6
            checked += 1
    assert checked >= 5
