"""Unit tests for zupt.segments (step 5c)."""
import math
import os

import numpy as np
import pytest

from zupt.segments import (analyse_segment, load_segment_config, run_segments, split_segments,
                           wheel_scale)

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "..", "config", "segment_params_dev.yaml")
DT = 0.02


@pytest.fixture
def cfg():
    return load_segment_config(CFG)


def trapezoid(T=6.0, vmax=0.3, ramp=0.5):
    t = np.arange(0, T, DT)
    v = np.clip(np.minimum(t / ramp, (T - t) / ramp), 0, 1) * vmax
    a = np.r_[np.diff(v) / DT, 0.0]
    return t, v, a


def session(v_wheel_fn=None, bias=0.02, still=1.0, seed=0):
    rng = np.random.default_rng(seed)
    tm, v, a = trapezoid()
    n0 = int(still / DT)
    t = np.r_[np.arange(n0) * DT, still + tm, still + tm[-1] + DT + np.arange(n0) * DT]
    vt = np.r_[np.zeros(n0), v, np.zeros(n0)]
    at = np.r_[np.diff(vt) / DT, 0.0]              # consistent with vt across the joins
    st = np.r_[np.ones(n0, bool), v <= 0, np.ones(n0, bool)]
    st[n0:n0 + len(v)] = False
    ax = at + bias + rng.normal(0, 2e-4, len(t))
    gz = rng.normal(0, 1e-3, len(t))
    vw = vt if v_wheel_fn is None else v_wheel_fn(t, vt)
    return t, ax, gz, st, vt, vw


def test_split_segments_needs_closing_stop():
    st = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0], bool)
    segs = split_segments(np.arange(9), st)
    assert len(segs) == 1
    assert segs[0][1] == (2, 5)


def test_honest_wheels_no_slip_and_bias_removed(cfg):
    t, ax, gz, st, vt, vw = session(bias=0.03)
    (r,) = run_segments(cfg, t, ax, gz, st, t, vw)
    true_d = float(np.sum(vt) * DT)
    assert r.d_imu == pytest.approx(true_d, abs=0.03)
    assert not r.slip and abs(r.z) < 3


def test_locked_wheel_slide_detected_and_masked(cfg):
    """Wheels report 0 for the middle 2 s while the body keeps moving (ice slide)."""
    def lock(t, vt):
        w = vt.copy(); w[(t > 2.5) & (t < 4.5)] = 0.0; return w
    t, ax, gz, st, vt, vw = session(lock)
    (r,) = run_segments(cfg, t, ax, gz, st, t, vw)
    assert r.slip and r.d_wheel < r.d_imu - 0.4
    m = (r.t > 2.6) & (r.t < 4.4)
    assert r.slip_mask[m].mean() > 0.9
    assert r.slip_mask[(r.t > 1.6) & (r.t < 2.3)].mean() < 0.1


def test_wheel_scale_recovered(cfg):
    res = []
    for seed in range(6):
        t, ax, gz, st, vt, vw = session(lambda t, v: 1.02 * v, seed=seed)
        res += run_segments(cfg, t, ax, gz, st, t, vw)
    for r in res:                                  # 2 % scale error is not "slip"
        assert not r.slip
    s, ss, n = wheel_scale(cfg, res)
    assert n == 6
    assert s == pytest.approx(1 / 1.02, abs=0.01)


def test_end_bias_handles_slope_parking(cfg):
    """Different tilt residual at the two stops (bias jumps by 0.04 across the segment)."""
    t, ax, gz, st, vt, vw = session(bias=0.0)
    lin = np.interp(t, [t[0], t[-1]], [0.0, 0.04])
    (r,) = run_segments(cfg, t, ax + lin, gz, st, t, vw)
    assert not r.slip
    assert abs(r.d_imu - float(np.sum(vt) * DT)) < 0.05


def test_short_stop_before_segment_uses_closing_stop_bias(cfg):
    t, ax, gz, st, vt, vw = session(bias=0.03, still=1.0)
    n0 = 50
    st2 = st.copy(); st2[:n0 - 4] = False            # only 4 still samples before the motion
    segs = split_segments(t, st2)
    assert len(segs) == 1
    (r,) = run_segments(cfg, t, ax, gz, st2, t, vw)
    assert abs(r.d_imu - float(np.sum(vt) * DT)) < 0.05


def test_decimate_imu_preserves_delta_v():
    """1 kHz IMU with a 55 ms 5.4 m/s^2 ramp: averaging to 50 Hz keeps the full delta-v
    (point-sampling at 50 Hz would keep 2 or 3 samples' worth: 0.216 or 0.324 m/s)."""
    import pandas as pd
    from zupt.replay import decimate_imu
    t = np.arange(0, 1.0, 0.001)
    a = np.where((t >= 0.3031) & (t < 0.3031 + 0.055), 5.4, 0.0)
    df = pd.DataFrame(dict(time_sec=np.floor(t).astype(int), time_nsec=((t % 1) * 1e9).round().astype(int),
                           angular_vel_z=0.0, linear_acc_x=a, linear_acc_y=0.0, linear_acc_z=9.8,
                           orient_w=1.0))
    d = decimate_imu(df, 50.0)
    assert len(d) == 50
    assert float(d.linear_acc_x.sum() * 0.02) == pytest.approx(5.4 * 0.055, abs=0.006)


def test_imprecise_scale_estimate_is_not_applied(cfg):
    from zupt.segments import scale_usable
    assert not scale_usable(cfg, 0.98, 0.03, 8)          # ±3 %: worse than doing nothing
    assert not scale_usable(cfg, 0.98, 0.002, 2)         # too few segments
    assert scale_usable(cfg, 0.98, 0.002, 8)


def test_decimate_imu_tolerates_dropped_messages():
    """25 % of 1 kHz messages missing at random: time bins keep delta-v and the 50 Hz grid."""
    import pandas as pd
    from zupt.replay import decimate_imu
    rng = np.random.default_rng(0)
    t = np.arange(0, 2.0, 0.001)
    a = np.where((t >= 0.5031) & (t < 0.5031 + 0.055), 5.4, 0.0)
    keep = rng.random(len(t)) > 0.25
    t, a = t[keep], a[keep]
    df = pd.DataFrame(dict(time_sec=np.floor(t).astype(int), time_nsec=((t % 1) * 1e9).round().astype(int),
                           angular_vel_z=0.0, linear_acc_x=a, linear_acc_y=0.0, linear_acc_z=9.8,
                           orient_w=1.0))
    d = decimate_imu(df, 50.0)
    tt = d.time_sec + d.time_nsec * 1e-9
    assert np.allclose(np.diff(tt), 0.02, atol=1e-6)
    assert float(d.linear_acc_x.sum() * 0.02) == pytest.approx(5.4 * 0.055, abs=0.02)
