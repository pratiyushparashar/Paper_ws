"""
6-state EKF for a differential-drive robot (spec rev 2.1 §2, skeleton change 2).

    state X = [x, y, theta, v, omega, b_g]
      x, y   position (m)          theta  heading (rad)
      v      forward speed (m/s)   omega  true yaw rate (rad/s)
      b_g    gyro z bias (rad/s)

Predict: constant-velocity kinematics; v, omega, b_g random walks.
Updates (each a standard EKF measurement update with its own R):
    odometry  z = [v_odom, w_odom]   h = [v, omega]          R = R_odom,0 / (r_odom + eps)
    gyro      z = gyro_z             h = omega + b_g         R = R_g,0    / (r_imu  + eps)
    ZUPT      z = 0                  h = v                   R = R_zupt,0 * R_scale(c)
    ZARU      z = 0                  h = omega               R = R_zaru,0 * R_scale(c)

The Gazebo IMU orientation is never used (decision D4): heading comes only from omega.
Q(t) is inflated on v and omega by the slip probability (skeleton change 5); P_slip defaults
to 0 until the LSTM provides it.
"""
from dataclasses import dataclass
import math

import numpy as np

IX, IY, ITH, IV, IW, IB = range(6)
EPS = 1e-3


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class EkfParams:
    # process noise densities (per second)
    q_xy: float
    q_theta: float
    q_v: float
    q_w: float
    q_b: float
    k_v_slip: float          # Q_v  *= (1 + k_v_slip * P_slip)
    k_w_slip: float          # Q_w  *= (1 + k_w_slip * P_slip)
    # measurement noise (variances)
    r_odom_v: float
    r_odom_w: float
    r_gyro: float
    r_zupt: float
    r_zaru: float
    # initial uncertainty (std)
    p0_xy: float
    p0_theta: float
    p0_v: float
    p0_w: float
    p0_b: float


def load_ekf_params(path):
    import yaml
    from dataclasses import fields
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = raw.get("ekf", raw)
    known = {f.name for f in fields(EkfParams)}
    unknown = set(raw) - known - {"status"}
    if unknown:
        raise ValueError(f"Unknown EKF parameters: {sorted(unknown)}")
    missing = [k for k in known if raw.get(k) is None]
    if missing:
        raise ValueError(f"Placeholder EKF parameters (null): {sorted(missing)}")
    return EkfParams(**{k: float(raw[k]) for k in known})


class Ekf6:
    def __init__(self, p: EkfParams, x0=0.0, y0=0.0, theta0=0.0):
        self.p = p
        self.x = np.array([x0, y0, theta0, 0.0, 0.0, 0.0])
        self.P = np.diag([p.p0_xy ** 2, p.p0_xy ** 2, p.p0_theta ** 2,
                          p.p0_v ** 2, p.p0_w ** 2, p.p0_b ** 2])
        self.t = None
        self.last_innov = {}

    # ---------------------------------------------------------------- predict
    def predict(self, t, p_slip=0.0):
        if self.t is None:
            self.t = t
            return
        dt = t - self.t
        if dt <= 0:
            return
        self.t = t
        x, y, th, v, w, b = self.x
        c, s = math.cos(th), math.sin(th)
        self.x = np.array([x + v * c * dt, y + v * s * dt, wrap(th + w * dt), v, w, b])
        F = np.eye(6)
        F[IX, ITH] = -v * s * dt
        F[IX, IV] = c * dt
        F[IY, ITH] = v * c * dt
        F[IY, IV] = s * dt
        F[ITH, IW] = dt
        p = self.p
        Q = np.diag([p.q_xy, p.q_xy, p.q_theta,
                     p.q_v * (1 + p.k_v_slip * p_slip),
                     p.q_w * (1 + p.k_w_slip * p_slip),
                     p.q_b]) * dt
        self.P = F @ self.P @ F.T + Q

    # ---------------------------------------------------------------- update core
    def _update(self, name, z, h, H, R):
        z = np.atleast_1d(np.asarray(z, float))
        H = np.atleast_2d(np.asarray(H, float))
        R = np.atleast_2d(np.asarray(R, float))
        innov = z - h
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        self.x = self.x + K @ innov
        self.x[ITH] = wrap(self.x[ITH])
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T          # Joseph form
        nis = float(innov @ np.linalg.solve(S, innov))
        self.last_innov[name] = (innov, nis)
        return innov, nis

    # ---------------------------------------------------------------- measurements
    def update_odom(self, v_odom, w_odom, r_odom=1.0):
        H = np.zeros((2, 6)); H[0, IV] = 1; H[1, IW] = 1
        R = np.diag([self.p.r_odom_v, self.p.r_odom_w]) / (r_odom + EPS)
        return self._update("odom", [v_odom, w_odom], self.x[[IV, IW]], H, R)

    def update_gyro(self, gyro_z, r_imu=1.0):
        H = np.zeros((1, 6)); H[0, IW] = 1; H[0, IB] = 1
        R = [[self.p.r_gyro / (r_imu + EPS)]]
        return self._update("gyro", [gyro_z], [self.x[IW] + self.x[IB]], H, R)

    def update_zupt(self, r_scale=1.0):
        H = np.zeros((1, 6)); H[0, IV] = 1
        return self._update("zupt", [0.0], [self.x[IV]], H, [[self.p.r_zupt * r_scale]])

    def update_zaru(self, r_scale=1.0):
        H = np.zeros((1, 6)); H[0, IW] = 1
        return self._update("zaru", [0.0], [self.x[IW]], H, [[self.p.r_zaru * r_scale]])

    # ---------------------------------------------------------------- accessors
    @property
    def bias(self):
        return self.x[IB]

    @property
    def bias_std(self):
        return math.sqrt(self.P[IB, IB])
