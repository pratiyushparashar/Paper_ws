"""
2D LiDAR scan matching (change 1). Pure Python + numpy/scipy.

match(ref, cur_pts, init) estimates the pose of the CURRENT scan in the REFERENCE scan's frame
(x, y, θ: where the sensor moved to) by point-to-line ICP:

    residual_i = n_i · (R(θ) p_i + t - q_i)       q_i, n_i: nearest reference point and its
                                                  line normal; p_i: current point
    Gauss-Newton on (t_x, t_y, θ), Huber weights, correspondences re-found every iteration
    with a shrinking distance gate; points outside the reference field of view are skipped
    (they were never observable from the reference pose, see core/lidar.py).

Covariance = σ_r² (JᵀWJ)⁻¹ · inflation. Point-to-line residuals carry no information along a
wall, so in a corridor JᵀWJ is nearly singular in the along-wall direction and the covariance
becomes huge there: the EKF then ignores the scan along the corridor and still uses it across
it and in heading. `degenerate` reports this (smallest-to-largest eigenvalue ratio of the
translation block).

Degeneracy handling (solution remapping, Zhang, Kaess & Singh, ICRA 2016): the information
matrix is eigen-decomposed (θ scaled by 1 m); directions whose eigenvalue is below
`degen_ratio` × the largest are not updated by ICP (they keep the initial guess, i.e. the EKF
prediction) and get variance `degen_var` in the covariance. Without this, noise in the estimated
wall normals leaks a little along-wall "information": in the unit-test corridor the matcher
reported ±1 cm along the axis while being 5 cm wrong — in a lane that would tell the EKF the
robot is not moving, the same failure as locked wheels on ice.

Quality gates (result.ok): enough matched points, converged, RMS residual small. No gate on the
EKF innovation: a correct scan that disagrees with wrongly-trusted wheels (Test C ice slide) must
not be thrown away because the filter currently believes the wheels.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from .lidar import scan_points


@dataclass
class ScanMatchConfig:
    max_iter: int = 20
    gate0: float = 0.30          # m, initial correspondence gate
    gate_min: float = 0.05       # m, final gate
    huber: float = 0.01          # m
    min_points: int = 60
    max_rms: float = 0.01        # m, accepted point-to-line RMS
    sigma_floor: float = 0.002   # m, residual std floor for the covariance
    cov_inflation: float = 25.0  # residuals are correlated; dev Tests C/D z-score rms 1.5-1.7 at 10
    conv_trans: float = 1e-4     # m
    conv_rot: float = 1e-4       # rad
    fov_margin: float = 0.05     # m inside max range still observable
    max_gap: float = 0.1         # m, for line normals
    normal_k: int = 3            # neighbours each side for the line fit of a normal
    degen_ratio: float = 0.02    # eigenvalue ratio below which a direction is unobservable
    degen_var: float = 100.0     # variance (m², rad²) reported for unobservable directions
    theta_scale: float = 1.0     # m, scales θ against translation in the eigen-analysis


@dataclass
class Reference:
    pts: np.ndarray
    normals: np.ndarray
    tree: cKDTree
    fov: tuple                    # (angle_min, angle_max, range_max)


@dataclass
class MatchResult:
    pose: np.ndarray              # (x, y, θ) of the current scan in the reference frame
    cov: np.ndarray               # 3x3
    n: int                        # matched points
    rms: float
    iters: int
    degenerate: float             # translation eigenvalue ratio (small = corridor-like)
    ok: bool


def fit_normals(pts, k=3, max_gap=0.1):
    """Unit normal per point from a total-least-squares line through its ±k beam-order
    neighbours (all consecutive gaps < max_gap); nan where that is not possible."""
    n = np.full_like(pts, np.nan)
    m = len(pts)
    if m < 2 * k + 1:
        return n
    gaps = np.r_[np.linalg.norm(np.diff(pts, axis=0), axis=1), np.inf]
    for i in range(k, m - k):
        if np.any(gaps[i - k:i + k] >= max_gap):
            continue
        q = pts[i - k:i + k + 1]
        c = q - q.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        n[i] = vt[1]
    return n


def make_reference(ranges, angle_min, angle_inc, rmin, rmax, max_gap=0.1, k=3):
    pts, r, _ = scan_points(ranges, angle_min, angle_inc, rmin, rmax)
    if len(pts) < 2 * k + 1:
        return None
    return Reference(pts, fit_normals(pts, k, max_gap), cKDTree(pts),
                     (angle_min, angle_min + angle_inc * (len(r) - 1), rmax))


def _subspace(H, cfg):
    """(projector onto observable directions, eigenvalues, eigenvectors) in scaled coords."""
    S = np.diag([1.0, 1.0, cfg.theta_scale])
    Hs = np.linalg.inv(S) @ H @ np.linalg.inv(S)
    ev, V = np.linalg.eigh(Hs)
    good = ev >= cfg.degen_ratio * max(ev[-1], 1e-12)
    Vg = V[:, good]
    return S, Vg @ Vg.T, ev, V, good


def _rot(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def match(ref: Reference, cur_pts, init=(0.0, 0.0, 0.0), cfg: ScanMatchConfig = None):
    cfg = cfg or ScanMatchConfig()
    x = np.array(init, float)
    has_n = ~np.isnan(ref.normals[:, 0])
    amin, amax, rmax = ref.fov
    it, res_w, J_w, n_used, rms = 0, None, None, 0, math.inf
    for it in range(1, cfg.max_iter + 1):
        gate = max(cfg.gate_min, cfg.gate0 * (0.7 ** (it - 1)))
        R = _rot(x[2])
        q = cur_pts @ R.T + x[:2]
        rr = np.hypot(q[:, 0], q[:, 1])
        bb = np.arctan2(q[:, 1], q[:, 0])
        seen = (rr < rmax - cfg.fov_margin) & (bb > amin + 0.02) & (bb < amax - 0.02)
        d, j = ref.tree.query(q)
        use = seen & (d < gate) & has_n[j]
        n_used = int(use.sum())
        if n_used < cfg.min_points:
            break
        qi, pi, nj = q[use], cur_pts[use], ref.normals[j[use]]
        e = np.sum((qi - ref.pts[j[use]]) * nj, axis=1)
        dR = np.array([[-math.sin(x[2]), -math.cos(x[2])], [math.cos(x[2]), -math.sin(x[2])]])
        J = np.c_[nj[:, 0], nj[:, 1], np.sum(nj * (pi @ dR.T), axis=1)]
        a = np.abs(e)
        w = np.where(a <= cfg.huber, 1.0, cfg.huber / np.maximum(a, 1e-12))
        H = J.T @ (J * w[:, None])
        g = J.T @ (w * e)
        S, Pg, _, _, _ = _subspace(H, cfg)
        try:
            dx = -np.linalg.solve(H + 1e-9 * np.eye(3), g)
        except np.linalg.LinAlgError:
            break
        dx = S @ (Pg @ (np.linalg.inv(S) @ dx))          # no update along unobservable directions
        x = x + dx
        x[2] = (x[2] + math.pi) % (2 * math.pi) - math.pi
        rms = float(math.sqrt(np.mean(e ** 2)))
        res_w, J_w = (e, w), (J, w)
        if np.hypot(dx[0], dx[1]) < cfg.conv_trans and abs(dx[2]) < cfg.conv_rot and gate <= cfg.gate_min + 1e-12:
            break
    if J_w is None:
        return MatchResult(x, np.eye(3) * 1e6, n_used, math.inf, it, 0.0, False)
    J, w = J_w
    H = J.T @ (J * w[:, None])
    sig2 = max(rms, cfg.sigma_floor) ** 2
    S, _, evs, V, good = _subspace(H, cfg)
    var_s = np.where(good, sig2 * cfg.cov_inflation / np.maximum(evs, 1e-300), cfg.degen_var)
    cov = S @ (V @ np.diag(var_s) @ V.T) @ S
    ev = np.linalg.eigvalsh(H[:2, :2])
    degenerate = float(ev[0] / max(ev[1], 1e-12))
    ok = n_used >= cfg.min_points and rms <= cfg.max_rms and np.all(np.isfinite(cov))
    return MatchResult(x, cov, n_used, rms, it, degenerate, bool(ok))
