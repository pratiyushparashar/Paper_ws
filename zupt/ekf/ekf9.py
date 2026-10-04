"""
9-state IMU-driven EKF for a differential-drive robot (build step 5b).

    state X = [x, y, theta, u, s, omega, b_g, b_ax, b_ay]
      x, y      position of the IMU point (= base_footprint = ground-truth point), world (m)
      theta     heading (rad)
      u, s      forward / lateral velocity of the IMU point, level heading frame (m/s)
      omega     true yaw rate (rad/s)
      b_g       gyro z bias (rad/s)
      b_ax,b_ay accelerometer bias, level heading frame (m/s^2)

Why this exists (step 5b): the 6-state filter took velocity from the wheels, so a zero-velocity
update only repeated what the wheels already said at a stop. Here velocity is predicted by
integrating the gravity-compensated accelerometer. That drifts (accelerometer bias), and ZUPT
is what bounds the drift and makes the accelerometer bias observable.

Predict (input = gravity-compensated horizontal specific force a = (a_x, a_y), zero-order hold):
    x'  = x + (u cos th - s sin th) dt          u' = u + (a_x - b_ax + omega s) dt
    y'  = y + (u sin th + s cos th) dt          s' = s + (a_y - b_ay - omega u) dt
    th' = th + omega dt                         omega, b_g, b_ax, b_ay: random walks
(the omega*s / omega*u terms are the rotating-frame terms; exact for any body-fixed point)

Updates, each with its own R:
    odometry  z = [v_odom, w_odom]  h = [u, omega]          R = R_odom,0 / (r_odom + eps)
    gyro      z = gyro_z            h = omega + b_g         R = R_g,0    / (r_imu  + eps)
    ZUPT      z = [0, 0]            h = [u, s]              R = R_zupt,0 * R_scale(c)
    ZARU      z = 0                 h = omega               R = R_zaru,0 * R_scale(c)
    NHC       z = 0                 h = s - l * omega       R = R_nhc,0  / (1 - P_slip + eps),
                                                            skipped when P_slip >= nhc_p_skip

NHC (non-holonomic constraint): the wheel axle cannot move sideways unless it skids. The IMU sits
l = 0.3 m ahead of the axle, so its own lateral velocity while turning is omega*l, not 0 — hence
h = s - l*omega. NHC is applied all the time; the slip probability loosens it (skeleton change 5:
in an IMU-driven filter slip corrupts the wheel-based MEASUREMENTS, not the IMU-based prediction,
so P_slip acts through R_nhc here; Q inflation by P_slip is kept for omega only).
The Gazebo IMU yaw is never used (decision D4); roll/pitch are used for gravity compensation only.
"""
from dataclasses import dataclass, fields
import math

import numpy as np

IX, IY, ITH, IU, IS, IW, IBG, IBAX, IBAY = range(9)
N = 9
EPS = 1e-3


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Ekf9Params:
    lever_l: float           # m, IMU ahead of the wheel axle (robot.xacro: s2 = 2r = 0.3)
    # process noise densities (per second)
    q_xy: float
    q_theta: float
    q_acc: float             # velocity random walk from accelerometer noise/model error
    q_w: float
    q_bg: float
    q_ba: float
    k_w_slip: float          # Q_w *= (1 + k_w_slip * P_slip)
    nhc_p_skip: float        # NHC not applied at/above this slip probability
    # measurement noise (variances)
    r_odom_v: float
    r_odom_w: float
    r_gyro: float
    r_zupt: float
    r_zaru: float
    r_nhc: float
    # initial uncertainty (std)
    p0_xy: float
    p0_theta: float
    p0_v: float
    p0_w: float
    p0_bg: float
    p0_ba: float


def load_ekf9_params(path):
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = raw.get("ekf9", raw)
    known = {f.name for f in fields(Ekf9Params)}
    unknown = set(raw) - known - {"status"}
    if unknown:
        raise ValueError(f"Unknown EKF9 parameters: {sorted(unknown)}")
    missing = [k for k in known if raw.get(k) is None]
    if missing:
        raise ValueError(f"Placeholder EKF9 parameters (null): {sorted(missing)}")
    return Ekf9Params(**{k: float(raw[k]) for k in known})


class Ekf9:
    def __init__(self, p: Ekf9Params, x0=0.0, y0=0.0, theta0=0.0):
        self.p = p
        self.x = np.zeros(N)
        self.x[[IX, IY, ITH]] = x0, y0, theta0
        self.P = np.diag([p.p0_xy ** 2, p.p0_xy ** 2, p.p0_theta ** 2,
                          p.p0_v ** 2, p.p0_v ** 2, p.p0_w ** 2,
                          p.p0_bg ** 2, p.p0_ba ** 2, p.p0_ba ** 2])
        self.t = None
        self.acc = (0.0, 0.0)            # held accelerometer input
        self.last_innov = {}

    def set_accel(self, ax, ay):
        """Latest gravity-compensated horizontal specific force (level heading frame)."""
        self.acc = (float(ax), float(ay))

    # ---------------------------------------------------------------- predict
    def predict(self, t, p_slip=0.0):
        if self.t is None:
            self.t = t
            return
        dt = t - self.t
        if dt <= 0:
            return
        self.t = t
        x, y, th, u, s, w, bg, bax, bay = self.x
        ax, ay = self.acc
        c, sn = math.cos(th), math.sin(th)
        self.x = np.array([x + (u * c - s * sn) * dt,
                           y + (u * sn + s * c) * dt,
                           wrap(th + w * dt),
                           u + (ax - bax + w * s) * dt,
                           s + (ay - bay - w * u) * dt,
                           w, bg, bax, bay])
        F = np.eye(N)
        F[IX, ITH] = (-u * sn - s * c) * dt
        F[IX, IU] = c * dt
        F[IX, IS] = -sn * dt
        F[IY, ITH] = (u * c - s * sn) * dt
        F[IY, IU] = sn * dt
        F[IY, IS] = c * dt
        F[ITH, IW] = dt
        F[IU, IS] = w * dt
        F[IU, IW] = s * dt
        F[IU, IBAX] = -dt
        F[IS, IU] = -w * dt
        F[IS, IW] = -u * dt
        F[IS, IBAY] = -dt
        p = self.p
        Q = np.diag([p.q_xy, p.q_xy, p.q_theta, p.q_acc, p.q_acc,
                     p.q_w * (1 + p.k_w_slip * p_slip), p.q_bg, p.q_ba, p.q_ba]) * dt
        self.P = F @ self.P @ F.T + Q

    # ---------------------------------------------------------------- update core
    def _update(self, name, z, h, H, R):
        z = np.atleast_1d(np.asarray(z, float))
        H = np.atleast_2d(np.asarray(H, float))
        R = np.atleast_2d(np.asarray(R, float))
        innov = z - np.atleast_1d(h)
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        self.x = self.x + K @ innov
        self.x[ITH] = wrap(self.x[ITH])
        I_KH = np.eye(N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T          # Joseph form
        nis = float(innov @ np.linalg.solve(S, innov))
        self.last_innov[name] = (innov, nis)
        return innov, nis

    # ---------------------------------------------------------------- measurements
    def update_odom(self, v_odom, w_odom, r_odom=1.0):
        H = np.zeros((2, N)); H[0, IU] = 1; H[1, IW] = 1
        R = np.diag([self.p.r_odom_v, self.p.r_odom_w]) / (r_odom + EPS)
        return self._update("odom", [v_odom, w_odom], self.x[[IU, IW]], H, R)

    def update_gyro(self, gyro_z, r_imu=1.0):
        H = np.zeros((1, N)); H[0, IW] = 1; H[0, IBG] = 1
        R = [[self.p.r_gyro / (r_imu + EPS)]]
        return self._update("gyro", [gyro_z], [self.x[IW] + self.x[IBG]], H, R)

    def update_zupt(self, r_scale=1.0):
        H = np.zeros((2, N)); H[0, IU] = 1; H[1, IS] = 1
        R = np.eye(2) * self.p.r_zupt * r_scale
        return self._update("zupt", [0.0, 0.0], self.x[[IU, IS]], H, R)

    def update_zaru(self, r_scale=1.0):
        H = np.zeros((1, N)); H[0, IW] = 1
        return self._update("zaru", [0.0], [self.x[IW]], H, [[self.p.r_zaru * r_scale]])

    def update_nhc(self, p_slip=0.0):
        """Applied every IMU sample, so a merely inflated R would still clamp a real skid after
        a few dozen updates; above nhc_p_skip the constraint is therefore not applied at all."""
        if p_slip >= self.p.nhc_p_skip:
            return None
        H = np.zeros((1, N)); H[0, IS] = 1; H[0, IW] = -self.p.lever_l
        R = [[self.p.r_nhc / (1.0 - min(max(p_slip, 0.0), 1.0) + EPS)]]
        h = self.x[IS] - self.p.lever_l * self.x[IW]
        return self._update("nhc", [0.0], [h], H, R)

    # ---------------------------------------------------------------- accessors
    @property
    def gyro_bias(self):
        return self.x[IBG]

    @property
    def accel_bias(self):
        return self.x[IBAX], self.x[IBAY]

    def std(self, i):
        return math.sqrt(self.P[i, i])
