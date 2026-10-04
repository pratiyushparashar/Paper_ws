"""
LiDAR motion evidence for the stationarity detector (Tier 3 + stuck detection). Pure Python.

Measured on the terrain lanes (Tests C/D): inside a lane the scan is mostly two parallel side
walls, so driving ALONG the lane leaves ~90% of the beams unchanged; only the wall ribs move.
Any MEDIAN-based statistic is therefore blind to motion there. Everything below uses FRACTIONS
of beams/points instead, and point-to-LINE distances (a point that slid along a wall is still on
the wall and must not count as a mismatch).

Per scan:
  changed      fraction of beams (valid in this and the previous scan) whose range changed by
               more than change_tol. Tier 3 (blocks only): a changed scan = a moving robot.

  hypothesis test against a REFERENCE scan (stuck check):
      H0 "not moved":           current points vs reference points
      H1 "moved as wheels say": current points vs reference points, current scan placed at the
                                displacement the wheel odometry claims since the reference
      out0, out1 = fraction of current points that do not lie on any reference surface
                   (point-to-line distance > nn_tol, or no reference point within max_gap).
                   Points that land outside the reference scan's field of view (beyond its
                   max range or angle span) are UNKNOWN, not mismatches: they were never
                   observable from the reference pose. Without this, wheel-claimed motion
                   along a featureless corridor pushes points past the 2.5 m range limit and
                   fakes ~9% contradiction (unit test test_featureless_corridor_refuses).
  The reference is kept while out0 stays small and reset when the scene changes, so ref_age is
  "how long the scan has been unchanged".

Why a hypothesis test: under a nonzero command the IMU cannot tell standing still from driving at
constant velocity, so LiDAR is the only velocity-sensitive check. In a degenerate place
(featureless corridor along its axis) an unchanged scan does NOT prove stillness, but there H1
fits as well as H0 (points slide along the walls), so out1 - out0 stays small and the stuck check
refuses. Stuck is declared only when the scan positively contradicts the wheels.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class LidarEvidence:
    t: float
    changed: float             # fraction of beams changed vs previous scan (nan for first scan)
    out0: float                # H0 outlier fraction vs reference
    out1: float                # H1 outlier fraction vs reference (nan if no wheel claim)
    ref_age: float             # s since the reference scan
    claim: float               # m, wheel-claimed translation since the reference
    claim_yaw: float           # rad, wheel-claimed rotation since the reference


def scan_points(ranges, angle_min, angle_inc, rmin, rmax):
    r = np.asarray(ranges, float)
    a = angle_min + angle_inc * np.arange(len(r))
    ok = np.isfinite(r) & (r > rmin) & (r < rmax)
    return np.c_[r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])], r, ok


def line_normals(pts, max_gap):
    """Unit normal at each point from its beam-order neighbours; nan where the neighbours are
    too far apart (edge / isolated point)."""
    n = np.full_like(pts, np.nan)
    if len(pts) < 3:
        return n
    tang = pts[2:] - pts[:-2]
    gap = np.maximum(np.linalg.norm(pts[2:] - pts[1:-1], axis=1),
                     np.linalg.norm(pts[1:-1] - pts[:-2], axis=1))
    L = np.linalg.norm(tang, axis=1)
    good = (gap < max_gap) & (L > 1e-9)
    nn = np.c_[-tang[:, 1], tang[:, 0]] / np.where(L > 1e-9, L, 1.0)[:, None]
    n[1:-1][good] = nn[good]
    return n


class ScanMotionCheck:
    """Scene-change anchor + short look-back reference.

    anchor     the scan at which the scene last changed; ref_age = time since then. Every new
               scan is compared to it (H0); above ref_reset the scene changed => new anchor.
    reference  for the hypothesis test: the oldest scan of the current static stretch that is
               at most `lookback` s old. Keeping the reference recent bounds the wheel claim
               (e.g. 0.45 m at 0.3 m/s), so the H1-shifted points stay inside the field of view
               however long the robot stays stuck. (With the anchor as reference, a 23 s stuck
               episode pushed every point out of view and the check went blind.)
    """

    def __init__(self, change_tol=0.01, nn_tol=0.01, max_gap=0.1, ref_reset=0.05,
                 min_points=30, lookback=1.5):
        self.change_tol = change_tol
        self.nn_tol = nn_tol
        self.max_gap = max_gap
        self.ref_reset = ref_reset           # anchor H0 outliers above this => scene changed
        self.min_points = min_points
        self.lookback = lookback             # s
        self.fov_margin = 0.05               # m inside max range still counts as observable
        self._fov = None                     # (angle_min, angle_max, range_max)
        self._prev_r = self._prev_ok = None
        self._anchor = None                  # (pts, normals, tree)
        self._static_t = None
        self._hist = []                      # [(t, ref, W)] scans of the current static stretch
        self._W = np.zeros(3)                # wheel-claimed sensor pose, odometry frame

    def add_odom(self, v, w, dt, lever_l=0.0):
        """Integrate the wheel-claimed motion of the sensor point (lever_l ahead of the axle)."""
        x, y, th = self._W
        u, s = v, w * lever_l
        self._W = np.array([x + (u * math.cos(th) - s * math.sin(th)) * dt,
                            y + (u * math.sin(th) + s * math.cos(th)) * dt, th + w * dt])

    def _make_ref(self, pts):
        return (pts, line_normals(pts, self.max_gap), cKDTree(pts))

    def _outliers(self, ref, pts, pose=None):
        if pose is not None:
            c, s = math.cos(pose[2]), math.sin(pose[2])
            pts = pts @ np.array([[c, s], [-s, c]]) + pose[:2]     # current frame -> ref frame
            rr = np.hypot(pts[:, 0], pts[:, 1])
            bb = np.arctan2(pts[:, 1], pts[:, 0])
            amin, amax, rmax = self._fov
            seen = (rr < rmax - self.fov_margin) & (bb > amin + 0.02) & (bb < amax - 0.02)
            if seen.sum() < self.min_points:
                return math.nan
            pts = pts[seen]
        rp, rn, tree = ref
        d, j = tree.query(pts)
        pl = np.abs(np.sum((pts - rp[j]) * rn[j], axis=1))
        dist = np.where(np.isnan(pl), d, pl)                      # no normal: point distance
        return float(((dist > self.nn_tol) | (d > self.max_gap)).mean())

    def step(self, t, ranges, angle_min, angle_inc, rmin, rmax):
        pts, r, ok = scan_points(ranges, angle_min, angle_inc, rmin, rmax)
        self._fov = (angle_min, angle_min + angle_inc * (len(r) - 1), rmax)
        both = ok & self._prev_ok if self._prev_ok is not None else None
        changed = (float(np.mean(np.abs(r[both] - self._prev_r[both]) > self.change_tol))
                   if both is not None and both.sum() >= self.min_points else math.nan)
        self._prev_r, self._prev_ok = r, ok
        if len(pts) < self.min_points:                     # nothing to compare against
            self._anchor, self._static_t, self._hist = None, t, []
            return LidarEvidence(t, changed, math.nan, math.nan, 0.0, 0.0, 0.0)
        cur = self._make_ref(pts)
        if self._anchor is None:
            self._anchor, self._static_t, self._hist = cur, t, []
        out_anchor = self._outliers(self._anchor, pts)
        if out_anchor > self.ref_reset:                    # scene changed
            self._anchor, self._static_t, self._hist = cur, t, []
            out_anchor = 0.0
        self._hist = [h for h in self._hist if t - h[0] <= self.lookback + 1e-9]
        if self._hist:
            t_r, ref, W_r = self._hist[0]
            c, s = math.cos(W_r[2]), math.sin(W_r[2])
            dx, dy = self._W[0] - W_r[0], self._W[1] - W_r[1]
            rel = np.array([c * dx + s * dy, -s * dx + c * dy, self._W[2] - W_r[2]])
            claim, yaw = float(math.hypot(rel[0], rel[1])), float(rel[2])
            out0 = self._outliers(ref, pts)
            out1 = self._outliers(ref, pts, rel) if (claim > 0 or yaw != 0) else math.nan
        else:
            claim = yaw = 0.0
            out0, out1 = out_anchor, math.nan
        self._hist.append((t, cur, self._W.copy()))
        return LidarEvidence(t, changed, out0, out1, t - self._static_t, claim, yaw)
