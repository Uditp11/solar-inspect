"""SORT's constant-velocity Kalman filter for one box, NumPy only.

State x = [u, v, s, r, u', v', s']: box centre (u, v), area s, aspect ratio r = w/h,
and the velocities of u, v and s. **r has no velocity**: a panel's aspect ratio is
modelled as constant, so it is a state that only measurements move. Measurement
z = [u, v, s, r].

The filter is Bayesian inference under Gaussian assumptions: (x, P) is the mean and
covariance of a Gaussian belief about the box. Predict pushes that belief through the
linear motion model F and widens it by the process noise Q; update conditions it on
a measurement with noise R, which for a linear-Gaussian model is the Kalman gain.

The noise scalings are SORT's (Bewley et al. 2016), in SORT's units (px, px^2), and
are not retuned for D2. dt is one frame.
"""
from __future__ import annotations

import numpy as np

DIM_X, DIM_Z = 7, 4

# x_{t+1} = F x_t: each of u, v, s gains its velocity times dt; r and the velocities
# carry over.
F = np.eye(DIM_X)
F[0, 4] = F[1, 5] = F[2, 6] = 1.0

# z = H x: the measurement is the first four state components.
H = np.eye(DIM_Z, DIM_X)

# Initial covariance P0 (SORT: P = I, P[4:,4:] *= 1000, then P *= 10).
P0_POS = 10.0      # the first box is a measurement, so u, v, s, r start roughly known
P0_VEL = 10_000.0  # velocities are unobserved at birth: wide, so early updates set them

# Process noise Q (SORT: Q = I, Q[-1,-1] *= 0.01, Q[4:,4:] *= 0.01).
Q_POS = 1.0          # u, v, s, r drift by about a pixel per frame beyond the motion model
Q_VEL = 0.01         # velocities change slowly from frame to frame
Q_AREA_VEL = 1e-4    # area velocity compounds into s every frame, so it may drift least

# Measurement noise R (SORT: R = I, R[2:,2:] *= 10).
R_CENTRE = 1.0       # a detector's box centre is its most reliable part
R_SHAPE = 10.0       # its area and aspect ratio wobble more than its centre


def box_to_z(box: np.ndarray) -> np.ndarray:
    """xyxy box(es) (..., 4) -> measurement [u, v, s, r]."""
    b = np.asarray(box, dtype=float)
    w, h = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
    return np.stack([b[..., 0] + w / 2, b[..., 1] + h / 2, w * h, w / h], axis=-1)


def z_to_box(x: np.ndarray) -> np.ndarray:
    """Measurement or state (..., >= 4) -> xyxy box(es). w = sqrt(s r), h = s / w."""
    x = np.asarray(x, dtype=float)
    u, v, s, r = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    w = np.sqrt(s * r)
    h = s / w
    return np.stack([u - w / 2, v - h / 2, u + w / 2, v + h / 2], axis=-1)


def default_P0() -> np.ndarray:
    return np.diag([P0_POS] * 4 + [P0_VEL] * 3)


def default_Q() -> np.ndarray:
    return np.diag([Q_POS] * 4 + [Q_VEL, Q_VEL, Q_AREA_VEL])


def default_R() -> np.ndarray:
    return np.diag([R_CENTRE, R_CENTRE, R_SHAPE, R_SHAPE])


def initial_state(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A new track's belief, SORT's way: position from the first box, velocity 0."""
    x = np.zeros(DIM_X)
    x[:DIM_Z] = np.asarray(z, dtype=float)
    return x, default_P0()


def transition(dt: float = 1.0) -> np.ndarray:
    """F for a step of dt frames."""
    Fd = F.copy()
    Fd[0, 4] = Fd[1, 5] = Fd[2, 6] = dt
    return Fd


def predict(x: np.ndarray, P: np.ndarray, Q: np.ndarray,
            dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Prior for the next step: x' = F x, P' = F P F^T + Q. Q is per step, as in SORT.

    Area clamp, SORT's: if the area velocity would carry s to zero or below, it is
    zeroed first, so a shrinking box coasts at its last positive area instead of
    turning into a negative area and a NaN width in z_to_box.
    """
    x = np.asarray(x, dtype=float)
    if x[2] + dt * x[6] <= 0:
        x = x.copy()
        x[6] = 0.0
    Fd = transition(dt)
    return Fd @ x, Fd @ P @ Fd.T + Q


def update(x: np.ndarray, P: np.ndarray, z: np.ndarray,
           R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Posterior after measuring z. Joseph form for P, as filterpy (which SORT uses)
    computes it: it stays symmetric and positive semi-definite under rounding."""
    y = np.asarray(z, dtype=float) - H @ x          # innovation
    S = H @ P @ H.T + R                             # its covariance
    K = np.linalg.solve(S, H @ P).T                 # P H^T S^-1, as S and P are symmetric
    I_KH = np.eye(DIM_X) - K @ H
    return x + K @ y, I_KH @ P @ I_KH.T + K @ R @ K.T
