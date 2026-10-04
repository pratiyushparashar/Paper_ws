"""
Stationarity detector (spec rev 2.1 §3 + amendment A1). Pure Python, no ROS.

Call step() once per IMU sample, in time order. The detector is causal: it uses only the
current and past samples. It never reads the EKF state (skeleton change 3).

Safety principle: a false "stationary" is worse than a missed one. Every check can only
BLOCK a stop; only the IMU stillness evidence (Tier 4d, together with 4e) can support one,
and confirmation needs N consecutive candidate frames.

Stuck path (step 5b): with a NONZERO command the robot may still be immobilized (pushing an
obstacle, high-centred, wheels spinning). The wheels then report motion and odometry runs away.
Under command the IMU alone cannot tell standing still from constant-velocity driving, so this
path needs a second, velocity-sensitive confirmation: the LiDAR hypothesis test must show that
the scan is unchanged (H0) AND contradicts the motion the wheels claim (H1), see core/lidar.py.
Only then are the IMU stillness checks (4a, r_imu, 4d, 4e) evaluated. Odometry (4c) and P_slip
are not used on this path: wheels contradicting the robot is exactly the situation it detects.
"""
from collections import deque
from dataclasses import dataclass, field
import math

from .config import AXES, DetectorConfig


@dataclass
class ImuSample:
    t: float                 # s
    gyro_z: float            # rad/s
    accel_x: float           # m/s^2 (includes gravity components, raw)
    accel_y: float
    accel_z: float
    roll: float = 0.0        # rad, from the IMU's tilt estimate (yaw never used)
    pitch: float = 0.0


@dataclass
class Evidence:
    """Latest values of the other inputs at the time of an IMU sample. None = not available."""
    v_cmd: float = 0.0
    w_cmd: float = 0.0
    v_odom: float = None
    w_odom: float = None
    lidar_changed: float = None  # fraction of beams changed vs previous scan (Tier 3)
    lidar_age: float = None      # s since that scan
    lidar_out0: float = None     # H0 outlier fraction vs reference scan (stuck check)
    lidar_out1: float = None     # H1 outlier fraction (wheel-claimed motion)
    lidar_ref_age: float = None  # s the scan has been unchanged
    lidar_claim: float = None    # m, wheel-claimed translation since the reference
    r_imu: float = None
    r_odom: float = None
    p_slip: float = None


@dataclass
class DetectorOutput:
    t: float
    stationary: bool             # committed decision
    candidate: bool              # this frame passed every check
    confidence: float            # c in [0, 1]
    r_scale: float               # R(c)/R0, meaningful when stationary
    apply_update: bool           # stationary and allowed by the rate limit
    reason: str                  # first check that blocked, or "stationary"/"confirming"
    var: dict = field(default_factory=dict)
    acc_mean: tuple = (0.0, 0.0)


def per_axis_confidence(v, flat, veto):
    """Log-ramp: 1 at/below flat, 0 at/above veto."""
    if v <= flat:
        return 1.0
    if v >= veto:
        return 0.0
    return (math.log(veto) - math.log(v)) / (math.log(veto) - math.log(flat))


def gravity_compensated_horizontal(ax, ay, az, roll, pitch, g=9.80665):
    """Rotate body-frame specific force into a level frame (yaw-free) and return the
    horizontal components. For a level, still IMU this is ~(0, 0)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # R = Ry(pitch) * Rx(roll), applied to body vector
    x = cp * ax + sp * sr * ay + sp * cr * az
    y = cr * ay - sr * az
    return x, y


class StationarityDetector:
    def __init__(self, cfg: DetectorConfig, have_lidar=False, logger=print):
        self.cfg = cfg
        self.log = logger
        W = cfg.window
        self._buf = {a: deque(maxlen=W) for a in AXES}
        self._hx = deque(maxlen=W)
        self._hy = deque(maxlen=W)
        self._last_raw = None
        self._same_count = 0
        self._t_stop = None          # time the command last became zero
        self._cmd_moving = True      # assume moving until a zero command is seen
        self._consecutive = 0
        self._committed = False
        self._last_update_t = -math.inf
        self._stall_warned = False
        if cfg.status != "calibrated":
            self.log(f"[zupt] WARNING: parameter status is '{cfg.status}' (not calibrated)")
        if not have_lidar:
            self.log("[zupt] WARNING: no LiDAR input; Tier 3 disabled "
                     "(constant-velocity slides along featureless directions are harder to reject)")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _variance(buf):
        n = len(buf)
        m = sum(buf) / n
        return sum((x - m) ** 2 for x in buf) / n

    def _reject(self, t, reason, var=None, acc_mean=(0.0, 0.0), conf=0.0):
        self._consecutive = 0
        self._committed = False
        return DetectorOutput(t, False, False, conf, 1.0, False, reason, var or {}, acc_mean)

    # ------------------------------------------------------------------ main
    def step(self, s: ImuSample, e: Evidence) -> DetectorOutput:
        cfg = self.cfg

        # Buffers always fill (they are never cleared by Tier 1).
        for a in AXES:
            self._buf[a].append(getattr(s, a))
        hx, hy = gravity_compensated_horizontal(s.accel_x, s.accel_y, s.accel_z, s.roll, s.pitch)
        self._hx.append(hx)
        self._hy.append(hy)

        raw = (s.gyro_z, s.accel_x, s.accel_y, s.accel_z)
        self._same_count = self._same_count + 1 if raw == self._last_raw else 0
        self._last_raw = raw

        # Tier 1: commanded motion (unless the stuck path confirms immobilization)
        moving_cmd = abs(e.v_cmd) > cfg.eps_cmd or abs(e.w_cmd) > cfg.eps_cmd
        if moving_cmd:
            self._cmd_moving = True
            self._stall_warned = False
            if cfg.stuck_enable and self._lidar_says_stuck(e):
                return self._imu_stage(s, e, stuck=True)
            return self._reject(s.t, "tier1_command")
        if self._cmd_moving:                 # transition to zero command
            self._cmd_moving = False
            self._t_stop = s.t
            self._consecutive = 0
            self._committed = False

        # Tier 2: settle delay
        if self._t_stop is not None and s.t - self._t_stop < cfg.settle_delay:
            return self._reject(s.t, "tier2_settling")

        # Tier 3: LiDAR scan change (blocks only)
        if (e.lidar_changed is not None and self._lidar_fresh(e)
                and e.lidar_changed > cfg.lidar_changed_thresh):
            return self._reject(s.t, "tier3_lidar")

        return self._imu_stage(s, e, stuck=False)

    # ------------------------------------------------------------------ stages
    def _lidar_fresh(self, e):
        return e.lidar_age is None or e.lidar_age <= self.cfg.lidar_max_age

    def _lidar_says_stuck(self, e):
        c = self.cfg
        vals = (e.lidar_out0, e.lidar_out1, e.lidar_ref_age, e.lidar_claim, e.lidar_age)
        if any(v is None or v != v for v in vals):          # missing or nan
            return False
        return (e.lidar_age <= c.lidar_max_age
                and e.lidar_ref_age >= c.stuck_min_ref_age
                and e.lidar_claim >= c.stuck_min_claim
                and e.lidar_out0 <= c.stuck_max_out0
                and e.lidar_out1 - e.lidar_out0 >= c.stuck_margin)

    def _imu_stage(self, s, e, stuck):
        cfg = self.cfg
        # Tier 4a: frozen IMU = failed sensor, never "still"
        if self._same_count + 1 >= cfg.n_frozen:
            return self._reject(s.t, "tier4a_imu_frozen")

        # Tier 4b: learned evidence (slip is expected on the stuck path, so not checked there)
        if e.r_imu is not None and e.r_imu < cfg.r_imu_min:
            return self._reject(s.t, "tier4b_r_imu_low")
        if not stuck and e.p_slip is not None and e.p_slip >= cfg.p_slip_max:
            return self._reject(s.t, "tier4b_slip")

        # Tier 4c: odometry may block, never confirm; skipped when the wheels are not trusted
        # and on the stuck path
        wheels_trusted = e.r_odom is None or e.r_odom >= cfg.r_odom_min
        if (not stuck and wheels_trusted and e.v_odom is not None
                and (abs(e.v_odom) >= cfg.odom_v_eps or abs(e.w_odom or 0.0) >= cfg.odom_w_eps)):
            return self._reject(s.t, "tier4c_odometry")

        if len(self._buf[AXES[0]]) < cfg.window:
            return self._reject(s.t, "insufficient_window")

        # Tier 4d: per-axis variance with floor, hard veto, log-ramp confidence
        var = {a: max(self._variance(self._buf[a]), cfg.var_floor[a]) for a in AXES}
        for a in AXES:
            if var[a] >= cfg.veto[a]:
                return self._reject(s.t, f"tier4d_veto_{a}", var)
        conf = min(per_axis_confidence(var[a], cfg.flat[a], cfg.veto[a]) for a in AXES)

        # Tier 4e: gravity-compensated horizontal acceleration mean (amendment A1)
        mx = sum(self._hx) / len(self._hx)
        my = sum(self._hy) / len(self._hy)
        if abs(mx) >= cfg.acc_mean_thresh or abs(my) >= cfg.acc_mean_thresh:
            return self._reject(s.t, "tier4e_accel_mean", var, (mx, my), conf)

        if conf < cfg.c_min:
            out = self._reject(s.t, "low_confidence", var, (mx, my), conf)
            if not stuck:
                self._maybe_stall(s.t, var)
            return out

        # Candidate frame: hysteresis
        self._consecutive += 1
        if self._consecutive >= cfg.n_confirm:
            self._committed = True
        if not self._committed:
            return DetectorOutput(s.t, False, True, conf, 1.0, False, "confirming", var, (mx, my))

        apply = (s.t - self._last_update_t) >= 1.0 / cfg.max_update_rate_hz - 1e-9
        if apply:
            self._last_update_t = s.t
        return DetectorOutput(s.t, True, True, conf, cfg.r_scale(conf), apply,
                              "stuck" if stuck else "stationary", var, (mx, my))

    def _maybe_stall(self, t, var):
        if (not self._stall_warned and self._t_stop is not None
                and t - self._t_stop > self.cfg.stall_warn_time):
            noisy = {a: f"{var[a]:.2e}>{self.cfg.flat[a]:.2e}" for a in AXES if var[a] > self.cfg.flat[a]}
            self.log(f"[zupt] STALL t={t:.2f}: stopped {t - self._t_stop:.1f}s but not confirmed; "
                     f"axes above flat: {noisy}")
            self._stall_warned = True
