"""
Unit tests for zupt.ekf.ekf6 (6-state EKF with ZUPT/ZARU and gyro bias).
Run:  cd ~/paper_ws && python3 -m pytest -q tests/
"""
import dataclasses
import math
import os

import numpy as np
import pytest

from zupt.ekf.ekf6 import IB, ITH, IV, IW, IX, IY, Ekf6, load_ekf_params, wrap

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, "..", "config", "ekf_params_dev.yaml")
SIG_G = 0.00873


@pytest.fixture
def p():
    return load_ekf_params(PARAMS)


def test_placeholder_params_refused(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("ekf:\n  q_xy: null\n")
    with pytest.raises(ValueError):
        load_ekf_params(str(f))


def test_straight_motion_integrates_position(p):
    ekf = Ekf6(p)
    for k in range(1, 501):                       # 5 s at 100 Hz, 1 m/s, no rotation
        ekf.predict(k * 0.01)
        ekf.update_odom(1.0, 0.0)
        ekf.update_gyro(0.0)
    assert ekf.x[IX] == pytest.approx(5.0, abs=0.05)
    assert ekf.x[IY] == pytest.approx(0.0, abs=1e-3)
    assert ekf.x[IV] == pytest.approx(1.0, abs=1e-3)


def test_turning_integrates_heading(p):
    ekf = Ekf6(p)
    for k in range(1, 201):                       # 2 s at 0.5 rad/s
        ekf.predict(k * 0.01)
        ekf.update_odom(0.0, 0.5)
        ekf.update_gyro(0.5)
    assert ekf.x[ITH] == pytest.approx(1.0, abs=0.02)


def test_heading_wraps(p):
    ekf = Ekf6(p, theta0=math.pi - 0.01)
    for k in range(1, 101):
        ekf.predict(k * 0.01)
        ekf.update_odom(0.0, 1.0)
        ekf.update_gyro(1.0)
    assert -math.pi <= ekf.x[ITH] <= math.pi
    assert wrap(ekf.x[ITH] - (math.pi - 0.01 + 1.0)) == pytest.approx(0.0, abs=0.02)


def test_zaru_recovers_known_bias_without_wheels(p):
    """Still robot, biased gyro, NO odometry: only ZARU makes the bias observable."""
    rng = np.random.default_rng(0)
    b_true = 0.01
    ekf = Ekf6(p)
    for k in range(1, 1501):                      # 30 s at 50 Hz
        ekf.predict(k * 0.02)
        ekf.update_gyro(b_true + rng.normal(0, SIG_G))
        ekf.update_zaru()
        ekf.update_zupt()
    assert ekf.bias == pytest.approx(b_true, abs=3 * ekf.bias_std + 1e-4)
    assert abs(math.degrees(ekf.x[ITH])) < 0.5


def test_bias_unobservable_without_zaru_or_wheels(p):
    """Still robot, biased gyro, no odometry, no ZARU: heading drifts and bias stays uncertain."""
    rng = np.random.default_rng(0)
    ekf = Ekf6(p)
    for k in range(1, 1501):
        ekf.predict(k * 0.02)
        ekf.update_gyro(0.01 + rng.normal(0, SIG_G))
    assert ekf.bias_std > 0.5 * p.p0_b
    assert abs(math.degrees(ekf.x[ITH])) > 5.0


def test_zupt_pulls_velocity_to_zero(p):
    ekf = Ekf6(p)
    ekf.x[IV] = 0.5
    ekf.predict(0.0)
    ekf.update_zupt()
    assert abs(ekf.x[IV]) < 0.01


def test_r_scale_weakens_update(p):
    a, b = Ekf6(p), Ekf6(p)
    a.x[IV] = b.x[IV] = 0.5
    a.update_zupt(r_scale=1.0)
    b.update_zupt(r_scale=1e4)
    assert abs(b.x[IV]) > abs(a.x[IV])


def test_covariance_stays_symmetric_psd(p):
    rng = np.random.default_rng(1)
    ekf = Ekf6(p)
    for k in range(1, 3001):
        ekf.predict(k * 0.01)
        ekf.update_odom(rng.normal(0.3, 0.05), rng.normal(0.1, 0.1))
        if k % 2 == 0:
            ekf.update_gyro(0.1 + rng.normal(0, SIG_G))
        if k % 500 < 50:
            ekf.update_zupt(); ekf.update_zaru()
    assert np.allclose(ekf.P, ekf.P.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(ekf.P) > -1e-12)


def test_p_slip_inflates_process_noise(p):
    a, b = Ekf6(p), Ekf6(p)
    a.predict(0.0); b.predict(0.0)
    a.predict(0.1, p_slip=0.0)
    b.predict(0.1, p_slip=1.0)
    assert b.P[IV, IV] > a.P[IV, IV]
    assert b.P[IW, IW] > a.P[IW, IW]
    assert b.P[IB, IB] == pytest.approx(a.P[IB, IB])


def test_low_reliability_reduces_odometry_influence(p):
    a, b = Ekf6(p), Ekf6(p)
    a.update_odom(1.0, 0.0, r_odom=1.0)
    b.update_odom(1.0, 0.0, r_odom=0.01)
    assert a.x[IV] > b.x[IV]
