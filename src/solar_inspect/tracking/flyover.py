"""D5: a synthetic flyover rendered from one D2 frame, with ground-truth identities.

D2's frames are non-overlapping, so a real flyover cannot be reconstructed from them.
This builds the next best thing, and it is synthetic: a viewing window moves over one
source frame, and every labelled panel in that frame keeps its identity from window
to window.

Camera path, seeded. The window centre moves with random-walk velocity (white
acceleration, `accel_std` per axis per frame), starting at `pan_speed` in a random
direction, and reflects off the bounds that keep the whole window inside the source
frame. Yaw ramps at `yaw_rate_deg` per frame from 0, in a random direction; scale
ramps linearly from one end of `scale_range` to the other. Each frame's map from
source to window is the similarity

    p_w = c_w + s R(-theta) (p_s - c_t),       c_w = window centre,

written as a 3 x 3 homography H_t; the image is warped with cv2.warpPerspective.
Because the motion is not constant-velocity, the trackers' constant-velocity Kalman
filter is deliberately mis-specified.

Ground truth. A label box's four corners are warped with H_t; its box in the window
is the axis-aligned hull of the warped quad, clipped to the window, which adds the
shape noise of a rotating, scaling view. A box is **visible** in a frame when at
least `visible_fraction` of its warped area is inside the window and not under the
occluder; it is **occluded** when enough of it is inside the window but the occluder
hides it.

Occluder. An opaque band fixed in the source plane (a cloud shadow, say), filled with
the source frame's median DN, `occluder_speed * occlusion_frames` px wide, drifting
across the source at `occluder_speed` px per frame in a random direction and crossing
the window centre's mid-sequence position at mid-sequence. Panels are fixed in the
source plane, so the band passes over each at exactly `occluder_speed`; a panel no
wider than twice the band, measured across it, is more than half covered for about
`occlusion_frames` frames. The lengths actually produced are measured per sequence.

Coordinates are continuous, as in `detection.boxes`: pixel (i, j) covers
[i, i + 1) x [j, j + 1). OpenCV puts pixel centres on integers, so the warp and the
fill shift by half a pixel.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FlyoverParams:
    window: tuple[int, int] = (320, 256)            # w, h in px
    frames: int = 60
    pan_speed: float = 4.0                          # source px per frame, initial
    accel_std: float = 0.25                         # source px per frame^2, per axis
    yaw_rate_deg: float = 0.15                      # per frame
    scale_range: tuple[float, float] = (0.85, 1.15)
    occlusion_frames: float = 10.0                  # 0 = no occluder
    occluder_speed: float = 8.0                     # source px per frame
    visible_fraction: float = 0.5


def similarity(centre, theta: float, scale: float, window) -> np.ndarray:
    """H mapping source to window: p_w = c_w + s R(-theta) (p_s - centre)."""
    c, s = np.cos(theta), np.sin(theta)
    A = scale * np.array([[c, s], [-s, c]])          # s R(-theta)
    cw = np.array(window, dtype=float) / 2
    H = np.eye(3)
    H[:2, :2] = A
    H[:2, 2] = cw - A @ np.asarray(centre, dtype=float)
    return H


def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    q = pts @ H[:2, :2].T + H[:2, 2]
    w = pts @ H[2, :2] + H[2, 2]
    return q / w[..., None]


def box_corners(boxes: np.ndarray) -> np.ndarray:
    """(N, 4) xyxy -> (N, 4, 2) corners TL, TR, BR, BL."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    x1, y1, x2, y2 = b.T
    return np.stack([np.c_[x1, y1], np.c_[x2, y1], np.c_[x2, y2], np.c_[x1, y2]], axis=1)


def hull(quads: np.ndarray, window) -> np.ndarray:
    """Axis-aligned hull of each quad (N, 4, 2), clipped to the window -> (N, 4)."""
    w, h = window
    lo, hi = quads.min(axis=1), quads.max(axis=1)
    return np.c_[lo[:, 0].clip(0, w), lo[:, 1].clip(0, h), hi[:, 0].clip(0, w), hi[:, 1].clip(0, h)]


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1))) / 2


def visibility(quads: np.ndarray, window, occluder: np.ndarray | None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Per quad: the fraction of its area inside the window, and the fraction inside
    the window and not under the occluder (a convex quad in window coordinates)."""
    w, h = window
    win = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    occ = None if occluder is None else np.asarray(occluder, dtype=np.float32)
    inside, visible = np.zeros(len(quads)), np.zeros(len(quads))
    for i, q in enumerate(quads):
        total = _area(q)
        a_in, poly = cv2.intersectConvexConvex(q.astype(np.float32), win)
        hidden = 0.0
        if occ is not None and poly is not None and a_in > 0:
            hidden = cv2.intersectConvexConvex(poly.reshape(-1, 2), occ)[0]
        inside[i], visible[i] = a_in / total, (a_in - hidden) / total
    return inside, visible


def camera_path(src_wh, p: FlyoverParams, rng: np.random.Generator):
    """Per-frame window centre (T, 2), yaw (T,), scale (T,), and the bounce count."""
    T, (W, Hs), (ww, wh) = p.frames, src_wh, p.window
    theta = rng.choice([-1.0, 1.0]) * np.deg2rad(p.yaw_rate_deg) * np.arange(T)
    lo, hi = p.scale_range
    scale = np.linspace(lo, hi, T) if rng.random() < 0.5 else np.linspace(hi, lo, T)
    # Half-extents of the window's footprint in the source at the largest yaw and the
    # smallest scale: centres inside these bounds keep the whole window in the frame.
    tm = np.abs(theta).max()
    hx = (ww / 2 * np.cos(tm) + wh / 2 * np.sin(tm)) / scale.min()
    hy = (wh / 2 * np.cos(tm) + ww / 2 * np.sin(tm)) / scale.min()
    bounds = np.array([[hx, W - hx], [hy, Hs - hy]])
    if (bounds[:, 0] > bounds[:, 1]).any():
        raise ValueError(f"a {ww} x {wh} window does not fit a {W} x {Hs} source")
    c = rng.uniform(bounds[:, 0], bounds[:, 1])
    phi = rng.uniform(0, 2 * np.pi)
    v = p.pan_speed * np.array([np.cos(phi), np.sin(phi)])
    centres, bounces = np.zeros((T, 2)), 0
    for t in range(T):
        centres[t] = c
        c = c + v
        for k in range(2):
            a, b = bounds[k]
            if c[k] < a or c[k] > b:                 # reflect off the bound
                c[k] = 2 * (a if c[k] < a else b) - c[k]
                v[k] = -v[k]
                bounces += 1
        c = c.clip(bounds[:, 0], bounds[:, 1])
        v = v + rng.normal(0.0, p.accel_std, 2)
    return centres, theta, scale, bounces


def band(src_wh, p: FlyoverParams, centres: np.ndarray, rng: np.random.Generator):
    """The occluder's corners in the source plane per frame, (T, 4, 2), or None."""
    width = p.occluder_speed * p.occlusion_frames
    psi = rng.uniform(0, 2 * np.pi)
    if width <= 0:
        return None
    n, tau = np.array([np.cos(psi), np.sin(psi)]), np.array([-np.sin(psi), np.cos(psi)])
    L = float(np.hypot(*src_wh))                     # longer than any footprint
    mid = (p.frames - 1) / 2
    q = n @ centres[int(round(mid))] + p.occluder_speed * (np.arange(p.frames) - mid)
    a, b = (q - width / 2)[:, None] * n, (q + width / 2)[:, None] * n
    return np.stack([a + L * tau, b + L * tau, b - L * tau, a - L * tau], axis=1)


_HALF = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])   # continuous -> OpenCV pixel


def render(src: np.ndarray, H: np.ndarray, window, occluder: np.ndarray | None,
           fill: int) -> np.ndarray:
    img = cv2.warpPerspective(src, _HALF @ H @ np.linalg.inv(_HALF), tuple(window),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if occluder is not None:
        pts = np.round((occluder - 0.5) * 16).astype(np.int32)    # 4 fractional bits
        cv2.fillConvexPoly(img, pts, (fill,) * img.shape[2] if img.ndim == 3 else fill,
                           lineType=cv2.LINE_8, shift=4)
    return img


def make_sequence(src: np.ndarray, boxes: np.ndarray, p: FlyoverParams, seed: int) -> dict:
    """Render one sequence from `src` (H, W[, 3]) uint8 and its xyxy label boxes.
    Ground-truth id k + 1 is label box k, in every frame."""
    rng = np.random.default_rng(seed)
    src_wh = (src.shape[1], src.shape[0])
    centres, theta, scale, bounces = camera_path(src_wh, p, rng)
    occ_src = band(src_wh, p, centres, rng)
    fill = int(round(float(np.median(src[..., 0] if src.ndim == 3 else src))))
    corners = box_corners(boxes)

    frames, Hs, occ_w = [], [], []
    gt_frame, gt_id, gt_box, occ_frame, occ_id = [], [], [], [], []
    for t in range(p.frames):
        H = similarity(centres[t], theta[t], scale[t], p.window)
        occ = None if occ_src is None else warp_points(H, occ_src[t])
        quads = warp_points(H, corners)
        inside, visible = visibility(quads, p.window, occ)
        shown = visible >= p.visible_fraction
        hidden = (inside >= p.visible_fraction) & ~shown
        k = np.flatnonzero(shown)
        gt_frame += [t] * len(k)
        gt_id += (k + 1).tolist()
        gt_box.append(hull(quads[k], p.window))
        occ_frame += [t] * int(hidden.sum())
        occ_id += (np.flatnonzero(hidden) + 1).tolist()
        frames.append(render(src, H, p.window, occ, fill))
        Hs.append(H)
        occ_w.append(np.zeros((4, 2)) if occ is None else occ)

    return {
        "frames": np.stack(frames), "H": np.stack(Hs), "centre": centres, "theta": theta,
        "scale": scale, "occluder": np.stack(occ_w), "bounces": bounces, "fill": fill,
        "gt_frame": np.array(gt_frame, dtype=np.int64), "gt_id": np.array(gt_id, dtype=np.int64),
        "gt_box": np.concatenate(gt_box).reshape(-1, 4),
        "occ_frame": np.array(occ_frame, dtype=np.int64), "occ_id": np.array(occ_id, dtype=np.int64),
    }
