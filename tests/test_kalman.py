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

from solar_inspect.tracking.kalman import initial_state, predict, update

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
