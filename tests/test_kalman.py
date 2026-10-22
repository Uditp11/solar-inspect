"""SORT's constant-velocity box Kalman filter, tested before it exists.

State x = [u, v, s, r, u', v', s']: box centre, area, aspect ratio w/h, and the
velocities of the first three. `r` has no velocity. Measurement z = [u, v, s, r].

The trajectory used below, worked by hand (one frame per step, dt = 1):

    t   u          v         s            r
    0   100        50        400          1.6
    1   103        48        420          1.6
    2   106        46        440          1.6
    3   109        44        460          1.6
    velocity       u' = 3, v' = -2, s' = 20, and r constant.
"""
from __future__ import annotations

import numpy as np

from solar_inspect.tracking.kalman import (box_to_z, default_P0, default_Q, initial_state,
                                           predict, update, z_to_box)

U0, V0, S0, R0 = 100.0, 50.0, 400.0, 1.6
VEL = np.array([3.0, -2.0, 20.0])


def z_at(t: int) -> np.ndarray:
    return np.array([U0 + 3 * t, V0 - 2 * t, S0 + 20 * t, R0])


def test_kalman_recovers_exact_state_on_noiseless_linear_track():
    """With Q = 0, noiseless measurements and a linear trajectory, after two updates
    the filter recovers position and velocity exactly.

    The track starts SORT's way, from the first box: x = [z0, 0, 0, 0] with SORT's
    initial P, whose velocity block (10,000) is large and whose position block is 10.
    "Noiseless" is said to the filter as R -> 0 (1e-12 I): R exactly 0 would make the
    innovation covariance singular on `r`, which has no velocity, so after one exact
    update its variance is 0 and Q = 0 keeps it there.

    Why two updates are enough. Per coordinate the unknowns are position and velocity.
    Predict maps them through F, so the predicted position carries the velocity
    variance, and its covariance with velocity equals the velocity variance. The
    first update (z1) cannot set velocity exactly: the filter still believes z0 had
    variance 10, so it splits the innovation z1 - z0 between "z0 was off" and "the box
    moved", in the ratio 10 : 10,000. It does set position exactly (R -> 0), and in
    doing so leaves a velocity variance of about 10 that is fully correlated with the
    next predicted position. The second update (z2) then has an exact previous position
    and an innovation that is exactly the velocity error, with gain 1 on velocity: the
    estimate becomes z2 - z1, the true velocity. Two noiseless observations of a
    straight line fix the line.
    """
    Q = np.zeros((7, 7))
    R = 1e-12 * np.eye(4)
    x, P = initial_state(z_at(0))

    x, P = update(*predict(x, P, Q), z_at(1), R)
    # One update is not enough: velocity is 10,000 / 10,010 of the truth, not all of it.
    assert np.abs(x[4:] - VEL).max() > 1e-3

    x, P = update(*predict(x, P, Q), z_at(2), R)
    np.testing.assert_allclose(x[:4], z_at(2), rtol=0, atol=1e-9)
    np.testing.assert_allclose(x[4:], VEL, rtol=0, atol=1e-9)

    # And the next prediction lands on the next box of the line.
    x, _ = predict(x, P, Q)
    np.testing.assert_allclose(x[:4], z_at(3), rtol=0, atol=1e-9)


def test_predict_moves_the_mean_by_velocity_times_dt():
    # x = [10, 20, 100, 2 | 1, -1, 5], dt = 2:
    # u 10 + 2*1 = 12, v 20 + 2*(-1) = 18, s 100 + 2*5 = 110, r unchanged 2,
    # velocities unchanged.
    x = np.array([10.0, 20.0, 100.0, 2.0, 1.0, -1.0, 5.0])
    x2, _ = predict(x, np.eye(7), default_Q(), dt=2.0)
    np.testing.assert_allclose(x2, [12, 18, 110, 2, 1, -1, 5], rtol=0, atol=1e-12)


def test_predict_grows_the_covariance_by_F_P_Ft_plus_Q():
    # P = I, dt = 2. Each (position, velocity) pair has F = [[1, 2], [0, 1]], so
    #   F I F^T = [[1 + 4, 2], [2, 1]] = [[5, 2], [2, 1]].
    # Adding SORT's Q (1 on positions, 0.01 on u' v', 1e-4 on s'):
    #   u, v: [[6, 2], [2, 1.01]]     s: [[6, 2], [2, 1.0001]]
    # r has no velocity: 1 + 1 = 2, and no cross term with anything.
    expect = np.zeros((7, 7))
    for p, vel, qv in ((0, 4, 0.01), (1, 5, 0.01), (2, 6, 1e-4)):
        expect[p, p], expect[p, vel] = 6.0, 2.0
        expect[vel, p], expect[vel, vel] = 2.0, 1.0 + qv
    expect[3, 3] = 2.0
    _, P2 = predict(np.zeros(7), np.eye(7), default_Q(), dt=2.0)
    np.testing.assert_allclose(P2, expect, rtol=0, atol=1e-12)

    # A correlated 2 x 2 block, dt = 1, Q = 0: P_u = [[4, 1], [1, 2]],
    #   F P_u     = [[4 + 1, 1 + 2], [1, 2]] = [[5, 3], [1, 2]]
    #   F P_u F^T = [[5 + 3, 3], [1 + 2, 2]] = [[8, 3], [3, 2]].
    P = np.eye(7)
    P[np.ix_([0, 4], [0, 4])] = [[4.0, 1.0], [1.0, 2.0]]
    _, P1 = predict(np.zeros(7), P, np.zeros((7, 7)))
    np.testing.assert_allclose(P1[np.ix_([0, 4], [0, 4])], [[8, 3], [3, 2]], rtol=0, atol=1e-12)


def prior() -> tuple[np.ndarray, np.ndarray]:
    """A predicted belief to update: SORT's P0 pushed through one predict."""
    x = np.array([10.0, 20.0, 100.0, 2.0, 1.0, -1.0, 5.0])
    return predict(x, default_P0(), default_Q())


def test_update_with_R_to_zero_snaps_the_measured_components_to_z():
    # A measurement trusted completely: the posterior's u, v, s, r are z itself.
    x, P = prior()
    z = np.array([13.0, 17.0, 120.0, 2.5])
    xu, Pu = update(x, P, z, 1e-12 * np.eye(4))
    np.testing.assert_allclose(xu[:4], z, rtol=0, atol=1e-6)
    # ...and nothing is left uncertain about them.
    np.testing.assert_allclose(Pu[:4, :4], 0.0, rtol=0, atol=1e-6)


def test_update_with_R_to_infinity_leaves_the_prediction_unchanged():
    # A measurement trusted not at all: gain ~ P / R ~ 1e4 / 1e12, times an innovation
    # of at most 20, moves the mean by ~2e-7.
    x, P = prior()
    z = np.array([13.0, 17.0, 120.0, 2.5])
    xu, Pu = update(x, P, z, 1e12 * np.eye(4))
    np.testing.assert_allclose(xu, x, rtol=0, atol=1e-6)
    np.testing.assert_allclose(Pu, P, rtol=1e-6, atol=1e-6)


def test_a_shrinking_box_is_never_predicted_past_zero_area():
    # s = 100, s' = -30, coasting with no updates (Q = 0):
    #   100 -> 70 -> 40 -> 10, and next 10 - 30 = -20 <= 0, so the area velocity is
    #   zeroed first (SORT) and s stays 10 from then on.
    x = np.array([50.0, 50.0, 100.0, 1.0, 0.0, 0.0, -30.0])
    P, Q = np.eye(7), np.zeros((7, 7))
    areas = []
    for _ in range(5):
        x, P = predict(x, P, Q)
        areas.append(x[2])
    assert areas == [70.0, 40.0, 10.0, 10.0, 10.0]
    assert x[6] == 0.0

    # Not clamped while the prediction stays positive: 100 - 50 = 50, s' kept.
    x, _ = predict(np.array([50.0, 50.0, 100.0, 1.0, 0.0, 0.0, -50.0]), np.eye(7), Q)
    assert (x[2], x[6]) == (50.0, -50.0)


def test_box_to_z_and_back():
    # [10, 20, 50, 45]: w 40, h 25 -> u 30, v 32.5, s 40 * 25 = 1000, r 40 / 25 = 1.6
    box = np.array([10.0, 20.0, 50.0, 45.0])
    z = box_to_z(box)
    np.testing.assert_allclose(z, [30.0, 32.5, 1000.0, 1.6], rtol=0, atol=1e-12)
    np.testing.assert_allclose(z_to_box(z), box, rtol=0, atol=1e-9)
    # A full state converts too (its velocities are ignored), and so does a batch.
    np.testing.assert_allclose(z_to_box(np.r_[z, 1.0, 2.0, 3.0]), box, rtol=0, atol=1e-9)
    rng = np.random.default_rng(0)
    lt = rng.uniform(0, 600, (50, 2))
    boxes = np.column_stack([lt, lt + rng.uniform(1, 80, (50, 2))])
    np.testing.assert_allclose(z_to_box(box_to_z(boxes)), boxes, rtol=0, atol=1e-9)
