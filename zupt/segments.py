"""
ZUPT-bounded segment check (build step 5c). Pure Python.

A SEGMENT is the motion between two confirmed zero-velocity intervals (normal stops or stuck
commits). Velocity is known to be exactly zero at both ends, which turns the accelerometer into a
usable distance sensor for that segment:

    1. accelerometer bias at the start = mean horizontal specific force over the still interval
       before the segment; at the end = the same over the first `post_window` s of the still
       interval after it; linearly interpolated in between (handles drift and different tilt
       at the two stops, e.g. parking on a slope).
    2. forward velocity  v(t) = ∫ (a_x + ω² l - b(t)) dt      (ω² l: IMU l ahead of the axle)
    3. misclosure m = v(t_end), which must be 0. Removed by a linear ramp (assumes the velocity
       error accumulated uniformly); |m| is kept as a quality measure.
    4. d_imu = ∫ v dt,  d_wheel = ∫ v_odom dt  (forward distances, signed)
    5. σ_imu = sqrt(σ0² + (k_m |m| T)² + σ_n² dt T³ / 12)   (provisional, see config)
       z = (d_wheel - d_imu) / sqrt(σ_imu² + σ_wheel²)
       slip if |z| >= z_slip AND |d_wheel - d_imu| >= min_slip
    6. per-sample slip mask: |v_wheel - v_imu| >= v_slip inside a slipping segment. These are
       the self-supervised P_slip labels (no ground truth needed) and the samples the
       retroactive re-filter distrusts.
    7. wheel scale: weighted least squares d_imu = s * d_wheel over non-slip segments with
       |d_wheel| >= min_scale_dist.

The check is causal at segment level: a segment is judged at the following stop (+post_window).
"""
from dataclasses import dataclass, field, fields
import math

import numpy as np


@dataclass
class SegmentConfig:
    lever_l: float           # m
    post_window: float       # s of still data after the closing stop used for the end bias
    min_pre_samples: int     # still samples needed at an end to estimate the bias there
    sigma0: float            # m, base uncertainty of d_imu
    k_misclosure: float      # σ term per (m/s of misclosure * s of duration)
    sigma_wheel_rel: float   # relative wheel distance uncertainty (scale error)
    sigma_wheel_abs: float   # m
    z_slip: float
    min_slip: float          # m
    v_slip: float            # m/s, per-sample mask threshold
    min_scale_dist: float    # m
    scale_max_sigma: float   # apply a scale estimate only if its std is at most this
    scale_min_segments: int  # ... and it uses at least this many segments

    def __post_init__(self):
        missing = [f.name for f in fields(self) if getattr(self, f.name) is None]
        if missing:
            raise ValueError(f"Placeholder segment parameters (null): {missing}")


def load_segment_config(path):
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = raw.get("segments", raw)
    known = {f.name for f in fields(SegmentConfig)}
    unknown = set(raw) - known - {"status"}
    if unknown:
        raise ValueError(f"Unknown segment parameters: {sorted(unknown)}")
    return SegmentConfig(**{k: raw.get(k) for k in known})


@dataclass
class SegmentResult:
    t0: float
    t1: float
    d_wheel: float
    d_imu: float
    sigma: float
    z: float
    misclosure: float
    slip: bool
    t: np.ndarray = field(repr=False)          # IMU sample times inside the segment
    v_imu: np.ndarray = field(repr=False)      # corrected IMU forward velocity
    v_wheel: np.ndarray = field(repr=False)    # wheel forward velocity on the same times
    slip_mask: np.ndarray = field(repr=False)  # per-sample: wheels disagree with the IMU

    @property
    def T(self):
        return self.t1 - self.t0


def analyse_segment(cfg, t, ax, gz, still_pre, still_post, odom_t, odom_v):
    """t, ax, gz: IMU samples of the MOVING part [first moving sample .. last moving sample].
    still_pre / still_post: arrays of horizontal accel (centripetal-corrected) during the still
    intervals before and after. Returns SegmentResult or None if it cannot be judged."""
    pre_ok = len(still_pre) >= cfg.min_pre_samples
    post_ok = len(still_post) >= cfg.min_pre_samples
    if len(t) < 3 or not (pre_ok or post_ok):
        return None
    a = np.asarray(ax) + np.asarray(gz) ** 2 * cfg.lever_l
    # a very short stop gives no usable bias; then the other end's estimate is used for both
    b0 = float(np.mean(still_pre)) if pre_ok else float(np.mean(still_post))
    b1 = float(np.mean(still_post)) if post_ok else b0
    T = t[-1] - t[0]
    frac = (t - t[0]) / T if T > 0 else np.zeros_like(t)
    b = b0 + (b1 - b0) * frac
    dt = np.r_[np.diff(t), np.median(np.diff(t))]
    v = np.cumsum((a - b) * dt)
    m = float(v[-1])
    v = v - m * frac
    d_imu = float(np.sum(v * dt))
    vw = np.interp(t, odom_t, odom_v)
    d_wheel = float(np.sum(vw * dt))
    dd = np.diff(a)
    sig_n = 1.4826 * float(np.median(np.abs(dd - np.median(dd)))) / math.sqrt(2) if len(dd) else 0.0
    sig_imu = math.sqrt(cfg.sigma0 ** 2 + (cfg.k_misclosure * abs(m) * T) ** 2
                        + sig_n ** 2 * float(np.median(dt)) * T ** 3 / 12.0)
    sig_w = cfg.sigma_wheel_rel * abs(d_wheel) + cfg.sigma_wheel_abs
    z = (d_wheel - d_imu) / math.sqrt(sig_imu ** 2 + sig_w ** 2)
    slip = abs(z) >= cfg.z_slip and abs(d_wheel - d_imu) >= cfg.min_slip
    mask = (np.abs(vw - v) >= cfg.v_slip) if slip else np.zeros(len(t), bool)
    return SegmentResult(float(t[0]), float(t[-1]), d_wheel, d_imu, sig_imu, z, m, slip,
                         t, v, vw, mask)


def split_segments(t, stationary):
    """Index ranges (s, e) of the moving parts between stationary intervals, plus the index
    ranges of the still intervals before and after each. Segments touching the session start
    or end (no closing stop) are not returned."""
    st = np.asarray(stationary, bool)
    out = []
    k = 0
    n = len(st)
    while k < n and not st[k]:
        k += 1
    while k < n:
        p0 = k
        while k < n and st[k]:
            k += 1
        p1 = k                                    # still interval [p0, p1)
        if k >= n:
            break
        s = k
        while k < n and not st[k]:
            k += 1
        if k >= n:
            break
        e = k                                     # moving [s, e), next still starts at e
        q = e
        while q < n and st[q]:
            q += 1
        out.append(((p0, p1), (s, e), (e, q)))
    return out


def run_segments(cfg, t, ax_h, gz, stationary, odom_t, odom_v, imu_rate=50.0):
    """Whole-session helper: returns [SegmentResult]. ax_h = gravity-compensated horizontal
    forward specific force per IMU sample."""
    t = np.asarray(t); ax_h = np.asarray(ax_h); gz = np.asarray(gz)
    a_c = ax_h + gz ** 2 * cfg.lever_l
    npost = max(1, int(round(cfg.post_window * imu_rate)))
    res = []
    for (p0, p1), (s, e), (q0, q1) in split_segments(t, stationary):
        r = analyse_segment(cfg, t[s:e], ax_h[s:e], gz[s:e], a_c[p0:p1],
                            a_c[q0:min(q1, q0 + npost)], odom_t, odom_v)
        if r is not None:
            res.append(r)
    return res


def scale_usable(cfg, s, s_sigma, n):
    """Apply a wheel-scale correction only if it is clearly better than doing nothing: a 2 %
    radius error is not fixed by an estimate with a 3 % (or 47 %) standard deviation, it is
    made worse (step 5c dev run: Test D 0.159 -> 0.791 m)."""
    return n >= cfg.scale_min_segments and s == s and s_sigma <= cfg.scale_max_sigma


def wheel_scale(cfg, results):
    """Weighted LS of d_imu = s * d_wheel over non-slip segments. Returns (s, sigma_s, n)."""
    num = den = 0.0
    n = 0
    for r in results:
        if r.slip or abs(r.d_wheel) < cfg.min_scale_dist:
            continue
        w = 1.0 / r.sigma ** 2
        num += w * r.d_wheel * r.d_imu
        den += w * r.d_wheel ** 2
        n += 1
    if n == 0:
        return math.nan, math.nan, 0
    return num / den, 1.0 / math.sqrt(den), n
