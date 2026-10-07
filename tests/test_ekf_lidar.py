"""Unit tests for zupt.ekf.ekf_lidar (change 1): stochastic-cloning relative pose updates."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_scan_match import ROOM, CORRIDOR, raycast, AMIN, AINC, RMIN, RMAX  # noqa: E402

from zupt.core.lidar import scan_points
from zupt.core.scan_match import make_reference
from zupt.ekf.ekf9 import IU, IX, IY, ITH, load_ekf9_params
from zupt.ekf.ekf_lidar import EkfLidar

P9 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "ekf9_params_dev.yaml")


@pytest.fixture
def p():
    return load_ekf9_params(P9)


def drive(p, walls, v_true, wheel_v, T=4.0, use_lidar=True, seed=0, x0=0.0):
    """Straight line along +x at v_true; wheels report wheel_v (trusted); IMU sees the truth
    (zero accel at constant speed after the start). Scans at 10 Hz, IMU at 50 Hz."""
    rng = np.random.default_rng(seed)
    f = EkfLidar(p, x0, 0.0, 0.0)
    f.x[IU] = v_true
    t = 0.0
    while t <= T + 1e-9:
        f.predict(t)
        f.set_accel(0.0, 0.0)
        f.update_gyro(rng.normal(0, 0.00873))
        f.update_nhc()
        if abs(round(t * 100) % 1) < 1e-9:
            f.update_odom(wheel_v, 0.0)
        if use_lidar and round(t * 50) % 5 == 0:
            r = raycast((x0 + v_true * t, 0.0, 0.0), walls, rng)
            pts, _, _ = scan_points(r, AMIN, AINC, RMIN, RMAX)
            f.scan(pts, make_reference(r, AMIN, AINC, RMIN, RMAX))
        t = round(t + 0.02, 10)
    return f


def test_lidar_overrules_locked_wheels_in_a_room(p):
    """Wheels say 0 (locked on ice) while the robot moves 0.3 m/s. Without LiDAR the filter
    believes the trusted wheels (0.6 m error after 2 s). With LiDAR most of the motion is
    recovered (~77 %) even though the wrong wheels stay fully trusted; the remainder needs the
    wheels to be distrusted (reliability), and the LiDAR innovation is then large (NIS >> 3),
    which is itself a usable slip signal."""
    walls = ROOM
    no = drive(p, walls, 0.3, 0.0, T=2.0, use_lidar=False, x0=-0.6)
    yes = drive(p, walls, 0.3, 0.0, T=2.0, use_lidar=True, x0=-0.6)
    true_x = -0.6 + 0.6
    assert abs(no.x[IX] - true_x) > 0.55
    assert abs(yes.x[IX] - true_x) < 0.3 * abs(no.x[IX] - true_x)
    assert yes.last_innov["lidar"][1] > 20


def test_corridor_scan_does_not_fake_standstill(p):
    """Featureless corridor: the scan cannot see along-axis motion, so it must not drag the
    estimate towards zero; honest wheels at 0.3 m/s keep the estimate right."""
    f = drive(p, CORRIDOR, 0.3, 0.3, T=3.0)
    assert abs(f.x[IX] - 0.9) < 0.05
    assert abs(f.x[IY]) < 0.02


def test_relative_jacobian_numeric(p):
    f = EkfLidar(p, 1.0, 2.0, 0.4)
    f.x[[9, 10, 11]] = [0.7, 1.8, 0.3]
    h0 = f.predicted_relative()
    J = np.zeros((3, 12))
    for i in range(12):
        g = EkfLidar(p); g.x = f.x.copy(); g.x[i] += 1e-6
        J[:, i] = (g.predicted_relative() - h0) / 1e-6
    dx, dy = f.x[IX] - 0.7, f.x[IY] - 1.8
    c, s = math.cos(0.3), math.sin(0.3)
    H = np.zeros((3, 12))
    H[0, [0, 1, 9, 10, 11]] = c, s, -c, -s, -s * dx + c * dy
    H[1, [0, 1, 9, 10, 11]] = -s, c, s, -c, -c * dx - s * dy
    H[2, 2], H[2, 11] = 1, -1
    assert np.allclose(J, H, atol=1e-5)
