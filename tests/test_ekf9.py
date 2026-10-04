"""
Unit tests for zupt.ekf.ekf9 (step 5b). A small truth simulator generates the IMU-point kinematics
of a differential-drive robot (axle at origin of motion, IMU l ahead), then the accelerometer and
gyro readings that point would see. Run: python3 -m pytest -q tests/
"""
import math
import os

import numpy as np
import pytest

from zupt.ekf.ekf9 import (Ekf9, IBAX, IBAY, IBG, IS, ITH, IU, IW, IX, IY, N,
                           load_ekf9_params)

HERE = os.path.dirname(os.path.abspath(__file__))
P9 = os.path.join(HERE, "..", "config", "ekf9_params_dev.yaml")
DT = 0.02


@pytest.fixture
def p():
    return load_ekf9_params(P9)


def truth(v_fn, w_fn, T, l=0.3, slide_fn=None):
    """Axle speed v(t), yaw rate w(t), optional extra lateral slide velocity of the whole body.
    Returns per-step dicts with the IMU point's world position, heading, body velocities and the
    IMU readings (horizontal specific force in the level heading frame, gyro z)."""
    out, th, xa, ya = [], 0.0, 0.0, 0.0
    prev = None
    for k in range(int(T / DT) + 1):
        t = k * DT
        v, w = v_fn(t), w_fn(t)
        sl = slide_fn(t) if slide_fn else 0.0
        u, s = v, w * l + sl                          # IMU-point velocity, body frame
        x, y = xa + l * math.cos(th), ya + l * math.sin(th)
        out.append(dict(t=t, x=x, y=y, th=th, u=u, s=s, w=w))
        xa += (v * math.cos(th) - sl * math.sin(th)) * DT
        ya += (v * math.sin(th) + sl * math.cos(th)) * DT
        th += w * DT
    for k in range(len(out)):                         # accel from body-frame kinematics
        a, b = out[k], out[min(k + 1, len(out) - 1)]
        du, ds = (b["u"] - a["u"]) / DT, (b["s"] - a["s"]) / DT
        a["ax"] = du - a["w"] * a["s"]
        a["ay"] = ds + a["w"] * a["u"]
    return out


def run(p, tr, bias=(0.0, 0.0), bg=0.0, zupt=None, nhc=True, odom=False, p_slip=0.0, seed=0):
    rng = np.random.default_rng(seed)
    f = Ekf9(p, tr[0]["x"], tr[0]["y"], tr[0]["th"])
    for r in tr:
        f.predict(r["t"], p_slip)
        f.set_accel(r["ax"] + bias[0] + rng.normal(0, 2e-4), r["ay"] + bias[1] + rng.normal(0, 2e-4))
        f.update_gyro(r["w"] + bg + rng.normal(0, 0.00873))
        if odom:
            f.update_odom(r["u"], r["w"])
        if nhc:
            f.update_nhc(p_slip)
        if zupt is not None and zupt(r["t"]):
            f.update_zupt(); f.update_zaru()
    return f


def test_jacobian_matches_numeric(p):
    f = Ekf9(p)
    f.x = np.array([1.0, 2.0, 0.7, 0.4, 0.05, 0.3, 0.01, 0.02, -0.03])
    f.t = 0.0
    f.set_accel(0.2, -0.1)
    x0 = f.x.copy()

    def g(x):
        h = Ekf9(p); h.x = x.copy(); h.t = 0.0; h.set_accel(0.2, -0.1); h.P = np.eye(N)
        h.predict(DT); return h.x
    J = np.zeros((N, N))
    for i in range(N):
        d = np.zeros(N); d[i] = 1e-6
        J[:, i] = (g(x0 + d) - g(x0 - d)) / 2e-6
    f.P = np.eye(N); f.predict(DT)
    Q = np.diag([p.q_xy, p.q_xy, p.q_theta, p.q_acc, p.q_acc, p.q_w, p.q_bg, p.q_ba, p.q_ba]) * DT
    F_est = f.P - Q                                    # = F F^T with P0 = I
    assert np.allclose(F_est, J @ J.T, atol=1e-8)


def test_constant_acceleration_integrates(p):
    tr = truth(lambda t: 0.5 * t, lambda t: 0.0, 2.0)
    f = run(p, tr)
    assert f.x[IU] == pytest.approx(1.0, abs=0.02)
    assert f.x[IX] == pytest.approx(tr[-1]["x"], abs=0.03)


def test_rotating_frame_terms_on_a_circle(p):
    """Constant v and w: IMU point lateral velocity is w*l, accel x = -w^2 l, accel y = w u.
    Without the rotating-frame terms the filter would spiral away."""
    tr = truth(lambda t: 0.3, lambda t: 0.4, 15.0)
    f = run(p, tr)
    assert f.x[IS] == pytest.approx(0.4 * 0.3, abs=0.01)
    assert math.hypot(f.x[IX] - tr[-1]["x"], f.x[IY] - tr[-1]["y"]) < 0.2


def test_zupt_bounds_drift_and_estimates_accel_bias(p):
    tr = truth(lambda t: 0.0, lambda t: 0.0, 20.0)
    b = (0.03, -0.02)
    free = run(p, tr, bias=b, nhc=False)
    held = run(p, tr, bias=b, zupt=lambda t: True)
    assert math.hypot(free.x[IX] - 0.3, free.x[IY]) > 2.0       # 0.5*0.036*400 ~ 7 m
    assert math.hypot(held.x[IX] - 0.3, held.x[IY]) < 0.01
    assert held.x[IBAX] == pytest.approx(b[0], abs=2e-3)
    assert held.x[IBAY] == pytest.approx(b[1], abs=2e-3)


def test_nhc_removes_lateral_drift_while_driving(p):
    tr = truth(lambda t: 0.3, lambda t: 0.0, 20.0)
    b = (0.0, 0.03)
    no = run(p, tr, bias=b, nhc=False, odom=True)
    yes = run(p, tr, bias=b, nhc=True, odom=True)
    assert abs(no.x[IY]) > 2.0
    assert abs(yes.x[IY]) < 0.05
    assert yes.x[IBAY] == pytest.approx(0.03, abs=5e-3)


def test_nhc_loosened_by_slip_probability(p):
    """Real lateral slide of 0.2 m/s for 3 s (skid). High P_slip switches NHC off so the IMU
    tracks the slide; P_slip = 0 wrongly clamps the lateral velocity."""
    sl = lambda t: 0.2 if 2.0 <= t < 5.0 else 0.0
    tr = truth(lambda t: 0.0, lambda t: 0.0, 6.0, slide_fn=sl)
    tight = run(p, tr, p_slip=0.0)
    loose = run(p, tr, p_slip=1.0)
    true_y = tr[-1]["y"]
    assert true_y == pytest.approx(0.6, abs=0.01)
    assert abs(loose.x[IY] - true_y) < 0.1
    assert abs(tight.x[IY] - true_y) > 0.3


def test_odometry_update_couples_u(p):
    tr = truth(lambda t: 0.3, lambda t: 0.0, 5.0)
    f = run(p, tr, bias=(0.05, 0.0), odom=True)
    assert f.x[IU] == pytest.approx(0.3, abs=0.01)
    assert f.x[IBAX] == pytest.approx(0.05, abs=0.01)


def test_zaru_estimates_gyro_bias(p):
    tr = truth(lambda t: 0.0, lambda t: 0.0, 20.0)
    f = run(p, tr, bg=0.01, zupt=lambda t: True)
    assert f.x[IBG] == pytest.approx(0.01, abs=1e-3)
    assert abs(f.x[ITH]) < 0.01
