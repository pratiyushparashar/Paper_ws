"""
Detector parameters (spec rev 2.1 §3 + amendment A1).

Every threshold is a parameter; none is hard-coded in the detector. A parameter file is
either 'calibrated' (values derived by the calibration pipeline) or 'provisional'
(development only). A file containing any null value is a placeholder file: the detector
refuses to construct from it, so uncalibrated thresholds can never silently produce results.
"""
from dataclasses import dataclass, fields
import math


AXES = ("gyro_z", "accel_x", "accel_y")


@dataclass
class DetectorConfig:
    # --- provenance ---
    status: str                    # "calibrated" | "provisional"
    imu_rate_hz: float             # nominal IMU rate (Hz)

    # --- Tier 1/2 ---
    eps_cmd: float                 # |v_cmd|, |w_cmd| above this = commanded motion
    settle_delay: float            # s after the stop command before a stop may be confirmed

    # --- Tier 3 (LiDAR scan change; optional input) ---
    lidar_move_thresh: float       # m, median |Δrange| above this = moving

    # --- Tier 4a (frozen IMU) ---
    n_frozen: int                  # identical consecutive IMU samples => sensor failed

    # --- Tier 4b (learned evidence; optional inputs) ---
    r_imu_min: float               # IMU evidence ignored (=> not stationary) below this
    r_odom_min: float              # odometry gate skipped (wheels not trusted) below this
    p_slip_max: float              # slip probability at/above this => not stationary

    # --- Tier 4c (odometry gate; blocks only) ---
    odom_v_eps: float              # m/s
    odom_w_eps: float              # rad/s

    # --- Tier 4d (per-axis variance) ---
    window: int                    # W samples
    var_floor: dict                # per axis, (0.5*sigma)^2
    flat: dict                     # per axis
    veto: dict                     # per axis

    # --- Tier 4e (gravity-compensated accel mean; amendment A1) ---
    acc_mean_thresh: float         # m/s^2

    # --- decision / noise ---
    n_confirm: int                 # N consecutive candidates to commit
    c_min: float                   # minimum confidence for a candidate
    r0_scale: float                # R(c=1)/R0  (normally 1.0)
    rmax_scale: float              # R(c=c_min)/R0
    max_update_rate_hz: float      # ZUPT/ZARU applications per second while committed

    # --- diagnostics ---
    stall_warn_time: float         # s

    def __post_init__(self):
        missing = [f.name for f in fields(self) if getattr(self, f.name) is None]
        for name in ("var_floor", "flat", "veto"):
            d = getattr(self, name) or {}
            missing += [f"{name}.{a}" for a in AXES if d.get(a) is None]
        if missing:
            raise ValueError("Placeholder parameter file: missing/null values for "
                             + ", ".join(missing) + ". Run calibration first.")
        if self.status not in ("calibrated", "provisional"):
            raise ValueError(f"status must be 'calibrated' or 'provisional', got {self.status!r}")
        for a in AXES:
            if not (0 < self.var_floor[a] < self.flat[a] < self.veto[a]):
                raise ValueError(f"need 0 < var_floor < flat < veto for axis {a}")
        if not (0.0 < self.c_min <= 1.0):
            raise ValueError("c_min must be in (0, 1]")
        if self.window < 2 or self.n_confirm < 1 or self.n_frozen < 2:
            raise ValueError("window >= 2, n_confirm >= 1, n_frozen >= 2 required")

    @property
    def dt(self):
        return 1.0 / self.imu_rate_hz

    def r_scale(self, c):
        """R(c)/R0 = r0 * (rmax/r0)^((1-c)/(1-c_min)); c in [c_min, 1]."""
        if self.c_min >= 1.0:
            return self.r0_scale
        e = (1.0 - min(max(c, self.c_min), 1.0)) / (1.0 - self.c_min)
        return self.r0_scale * math.exp(e * math.log(self.rmax_scale / self.r0_scale))


def load_config(path):
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = raw.get("detector", raw)
    known = {f.name for f in fields(DetectorConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown parameters in {path}: {sorted(unknown)}")
    return DetectorConfig(**{k: raw.get(k) for k in known})
