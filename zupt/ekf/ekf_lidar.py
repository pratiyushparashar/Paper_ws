"""
EKF9 + LiDAR scan matching via stochastic cloning (change 1).

State = EKF9's 9 states + a CLONE of the pose (x_c, y_c, θ_c) taken at the previous scan:

    X = [x, y, θ, u, s, ω, b_g, b_ax, b_ay, x_c, y_c, θ_c]

At every scan k:
  1. the matcher aligns scan k to scan k-1, initialised with the EKF-predicted relative pose
     (so unobservable directions keep the prediction, see core/scan_match.py);
  2. measurement z = relative pose of the sensor between the two scans, in the clone frame:
        h = [ R(θ_c)ᵀ (p - p_c) ;  θ - θ_c ],   R = scan-match covariance
  3. the clone is replaced by the current pose (scan-to-scan).
The clone keeps the correlation between the old and the new pose in P, which is what makes a
relative measurement consistent (Roumeliotis & Burdick, ICRA 2002). The LiDAR sits at the same
x, y as the IMU point (robot.xacro lidar_joint 0 0 h), so no lever arm.
"""
import math

import numpy as np

from zupt.core.scan_match import ScanMatchConfig, match
from .ekf9 import Ekf9, Ekf9Params, ITH, IX, IY, N as N9, wrap

IXC, IYC, ITHC = N9, N9 + 1, N9 + 2
N12 = N9 + 3


class EkfLidar(Ekf9):
    def __init__(self, p: Ekf9Params, x0=0.0, y0=0.0, theta0=0.0, sm_cfg: ScanMatchConfig = None):
        super().__init__(p, x0, y0, theta0)
        self.n = N12
        x = np.zeros(N12); x[:N9] = self.x
        P = np.zeros((N12, N12)); P[:N9, :N9] = self.P
        self.x, self.P = x, P
        self.sm_cfg = sm_cfg or ScanMatchConfig()
        self.ref = None                   # reference scan (core.scan_match.Reference)
        self.last_match = None

    def _clone(self):
        self.x[[IXC, IYC, ITHC]] = self.x[[IX, IY, ITH]]
        idx, src = [IXC, IYC, ITHC], [IX, IY, ITH]
        self.P[idx, :] = self.P[src, :]
        self.P[:, idx] = self.P[:, src]
        self.P[np.ix_(idx, idx)] = self.P[np.ix_(src, src)]

    def predicted_relative(self):
        dx, dy = self.x[IX] - self.x[IXC], self.x[IY] - self.x[IYC]
        c, s = math.cos(self.x[ITHC]), math.sin(self.x[ITHC])
        return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(self.x[ITH] - self.x[ITHC])])

    def update_relative_pose(self, z, R):
        h = self.predicted_relative()
        dx, dy = self.x[IX] - self.x[IXC], self.x[IY] - self.x[IYC]
        c, s = math.cos(self.x[ITHC]), math.sin(self.x[ITHC])
        H = np.zeros((3, self.n))
        H[0, IX], H[0, IY], H[0, IXC], H[0, IYC], H[0, ITHC] = c, s, -c, -s, -s * dx + c * dy
        H[1, IX], H[1, IY], H[1, IXC], H[1, IYC], H[1, ITHC] = -s, c, s, -c, -c * dx - s * dy
        H[2, ITH], H[2, ITHC] = 1.0, -1.0
        z = np.asarray(z, float).copy()
        z[2] = h[2] + wrap(z[2] - h[2])
        return self._update("lidar", z, h, H, R)

    def scan(self, cur_pts, ref_next):
        """Process one scan: cur_pts = points of this scan (sensor frame), ref_next = the same
        scan prepared as the next reference. Call right after predict() to the scan time.
        Returns the MatchResult (None for the first scan)."""
        res = None
        if self.ref is not None and len(cur_pts):
            res = match(self.ref, cur_pts, init=self.predicted_relative(), cfg=self.sm_cfg)
            if res.ok:
                self.update_relative_pose(res.pose, res.cov)
        self.last_match = res
        self.ref = ref_next
        self._clone()
        return res
