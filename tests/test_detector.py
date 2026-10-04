"""
Unit tests for zupt.core.detector (spec rev 2.1 §3 + amendment A1).
Synthetic data with REALISTIC noise (SDF sigmas), never zero-noise — zero-noise tests were
what hid the old x+y summing bug.

Run:  cd ~/paper_ws && python3 -m pytest -q tests/
"""
import dataclasses
import math
import os

import numpy as np
import pytest

from zupt.core.config import DetectorConfig, load_config
from zupt.core.detector import (Evidence, ImuSample, StationarityDetector,
                                gravity_compensated_horizontal)

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, "..", "config", "zupt_params_dev.yaml")
G = 9.80665
SIG_G, SIG_A = 0.00873, 0.0002           # SDF noise


@pytest.fixture
def cfg():
    return load_config(PARAMS)


def quiet(cfg, **kw):
    return StationarityDetector(cfg, have_lidar=kw.pop("have_lidar", True), logger=lambda *_: None)


def still_samples(n, t0=0.0, rate=50.0, seed=0, ax0=0.0, ay0=0.0, gz0=0.0,
                  sig_g=SIG_G, sig_ax=SIG_A, sig_ay=SIG_A, roll=0.0, pitch=0.0):
    """Noisy IMU samples for a body with given specific-force offsets."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        out.append(ImuSample(t0 + k / rate,
                             gz0 + rng.normal(0, sig_g),
                             ax0 + rng.normal(0, sig_ax),
                             ay0 + rng.normal(0, sig_ay),
                             G + rng.normal(0, SIG_A),
                             roll, pitch))
    return out


def run(det, samples, ev=None, ev_fn=None):
    outs = []
    for s in samples:
        e = ev_fn(s) if ev_fn else (ev or Evidence(v_odom=0.0, w_odom=0.0))
        outs.append(det.step(s, e))
    return outs


def stop_then_still(det, n_still=150):
    """Commanded motion for 0.5 s, then zero command and a still robot."""
    moving = still_samples(25, t0=0.0, seed=1, ax0=0.5)
    run(det, moving, Evidence(v_cmd=0.5, v_odom=0.5, w_odom=0.0))
    return run(det, still_samples(n_still, t0=0.5, seed=2))


# ---------------------------------------------------------------- config
def test_placeholder_config_refused(cfg):
    d = dataclasses.asdict(cfg)
    d["flat"] = dict(d["flat"], accel_x=None)
    with pytest.raises(ValueError, match="Placeholder"):
        DetectorConfig(**d)


def test_threshold_ordering_enforced(cfg):
    d = dataclasses.asdict(cfg)
    d["veto"] = dict(d["veto"], gyro_z=d["flat"]["gyro_z"] / 2)
    with pytest.raises(ValueError):
        DetectorConfig(**d)


def test_r_scale_endpoints(cfg):
    assert cfg.r_scale(1.0) == pytest.approx(cfg.r0_scale)
    assert cfg.r_scale(cfg.c_min) == pytest.approx(cfg.rmax_scale)
    assert cfg.r_scale(1.0) < cfg.r_scale(0.75) < cfg.r_scale(cfg.c_min)


# ---------------------------------------------------------------- tiers 1-2
def test_tier1_command_blocks_even_when_imu_still(cfg):
    outs = run(quiet(cfg), still_samples(200), Evidence(v_cmd=0.3, v_odom=0.0, w_odom=0.0))
    assert not any(o.stationary for o in outs)
    assert all(o.reason == "tier1_command" for o in outs)


def test_genuine_stop_commits_after_settle_window_and_hysteresis(cfg):
    outs = stop_then_still(quiet(cfg))
    first = next(i for i, o in enumerate(outs) if o.stationary)
    t_commit = outs[first].t - 0.5
    assert t_commit >= cfg.settle_delay
    assert t_commit <= cfg.settle_delay + (cfg.window + cfg.n_confirm + 2) * cfg.dt
    assert all(o.stationary for o in outs[first:])


# ---------------------------------------------------------------- tier 4d
def test_summing_bug_regression_per_axis_not_summed(cfg):
    """accel_x and accel_y variances each ~0.6*flat: their SUM exceeds flat (old bug would
    reject), but each axis is flat, so a genuine stop must still commit."""
    s = math.sqrt(0.6 * cfg.flat["accel_x"])
    assert 2 * s ** 2 > cfg.flat["accel_x"]
    det = quiet(cfg)
    run(det, still_samples(25, seed=1, ax0=0.5), Evidence(v_cmd=0.5, v_odom=0.5, w_odom=0.0))
    outs = run(det, still_samples(200, t0=0.5, seed=3, sig_ax=s, sig_ay=s))
    assert sum(o.stationary for o in outs) > 100


def test_hard_veto_single_axis(cfg):
    """Gyro and accel_y flat, accel_x clearly moving -> never stationary."""
    det = quiet(cfg)
    outs = run(det, still_samples(300, sig_ax=math.sqrt(10 * cfg.veto["accel_x"])))
    assert not any(o.stationary for o in outs)
    assert any(o.reason == "tier4d_veto_accel_x" for o in outs)


def test_variance_floor_keeps_log_defined(cfg):
    det = quiet(cfg)
    outs = run(det, still_samples(200, sig_ax=1e-9, sig_ay=1e-9))
    assert all(math.isfinite(o.confidence) for o in outs)
    assert any(o.stationary for o in outs)


# ---------------------------------------------------------------- tier 4e (amendment A1)
def test_constant_deceleration_slide_rejected(cfg):
    """Real failure from session 20261004_235739: wheels locked, body sliding with constant
    deceleration -0.53 m/s^2; variance is as flat as standing still. Must NOT be stationary."""
    det = quiet(cfg)
    run(det, still_samples(25, seed=1, ax0=0.5), Evidence(v_cmd=0.5, v_odom=0.5, w_odom=0.0))
    outs = run(det, still_samples(50, t0=0.5, seed=4, ax0=-0.53))     # 1 s slide, odom = 0
    assert not any(o.stationary for o in outs)
    assert any(o.reason == "tier4e_accel_mean" for o in outs)


def test_slide_would_be_accepted_without_tier4e(cfg):
    """Shows the previous test is meaningful: variance-only detection commits on the slide."""
    c2 = dataclasses.replace(cfg, acc_mean_thresh=1e3)
    det = quiet(c2)
    run(det, still_samples(25, seed=1, ax0=0.5), Evidence(v_cmd=0.5, v_odom=0.5, w_odom=0.0))
    outs = run(det, still_samples(50, t0=0.5, seed=4, ax0=-0.53))
    assert any(o.stationary for o in outs)


def test_tilted_still_imu_is_gravity_compensated(cfg):
    """Robot parked on a 5 deg slope: raw accel_x ~ g*sin(5deg) = 0.85 m/s^2, but the
    gravity-compensated horizontal acceleration is ~0, so a genuine stop still commits."""
    p = math.radians(5.0)
    # specific force of a still body pitched by p (body frame): R^T * (0,0,g)
    ax0, az0 = -G * math.sin(p), G * math.cos(p)
    hx, hy = gravity_compensated_horizontal(ax0, 0.0, az0, 0.0, p)
    assert abs(hx) < 1e-9 and abs(hy) < 1e-9
    samples = still_samples(200, ax0=ax0, pitch=p)
    for s in samples:
        s.accel_z = az0 + (s.accel_z - G)
    outs = run(quiet(cfg), samples)
    assert any(o.stationary for o in outs)


# ---------------------------------------------------------------- tier 4a / 4b / 4c / 3
def test_frozen_imu_is_failure_not_stillness(cfg):
    frozen = [ImuSample(k * 0.02, 0.001, 0.0001, 0.0001, G) for k in range(200)]
    outs = run(quiet(cfg), frozen)
    assert not any(o.stationary for o in outs)
    assert outs[-1].reason == "tier4a_imu_frozen"


def test_odometry_blocks(cfg):
    outs = run(quiet(cfg), still_samples(200), Evidence(v_odom=0.05, w_odom=0.0))
    assert not any(o.stationary for o in outs)


def test_odometry_zero_cannot_confirm_alone(cfg):
    """Wheels read zero but IMU shows motion -> not stationary."""
    outs = run(quiet(cfg), still_samples(200, sig_ax=0.5), Evidence(v_odom=0.0, w_odom=0.0))
    assert not any(o.stationary for o in outs)


def test_learned_evidence_blocks(cfg):
    for ev in (Evidence(v_odom=0.0, w_odom=0.0, r_imu=0.1),
               Evidence(v_odom=0.0, w_odom=0.0, p_slip=0.9)):
        outs = run(quiet(cfg), still_samples(200), ev)
        assert not any(o.stationary for o in outs)


def test_lidar_blocks(cfg):
    outs = run(quiet(cfg), still_samples(200), Evidence(v_odom=0.0, w_odom=0.0, lidar_delta=0.1))
    assert not any(o.stationary for o in outs)


# ---------------------------------------------------------------- hysteresis / rate limit
def test_single_bad_frame_drops_immediately(cfg):
    det = quiet(cfg)
    outs = stop_then_still(det)
    assert outs[-1].stationary
    o = det.step(still_samples(1, t0=10.0, seed=9)[0], Evidence(v_odom=0.2, w_odom=0.0))
    assert not o.stationary
    o = det.step(still_samples(1, t0=10.02, seed=10)[0], Evidence(v_odom=0.0, w_odom=0.0))
    assert not o.stationary                      # must re-confirm N frames


def test_update_rate_limited(cfg):
    outs = stop_then_still(quiet(cfg), n_still=500)
    st = [o for o in outs if o.stationary]
    dur = st[-1].t - st[0].t
    n_upd = sum(o.apply_update for o in st)
    assert n_upd <= dur * cfg.max_update_rate_hz + 1
    assert n_upd >= dur * cfg.max_update_rate_hz - 1


def test_distrusted_odometry_does_not_block(cfg):
    """Faulty wheels (noisy v) must block when trusted, but are skipped when r_odom is low."""
    outs = run(quiet(cfg), still_samples(200), Evidence(v_odom=0.3, w_odom=0.0, r_odom=1.0))
    assert not any(o.stationary for o in outs)
    outs = run(quiet(cfg), still_samples(200), Evidence(v_odom=0.3, w_odom=0.0, r_odom=0.1))
    assert any(o.stationary for o in outs)
