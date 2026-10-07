"""Unit tests for zupt.core.scan_match (change 1)."""
import math

import numpy as np

from zupt.core.lidar import scan_points
from zupt.core.scan_match import make_reference, match

AMIN, AINC, N = -math.radians(135), math.radians(0.5), 541
RMIN, RMAX = 0.1, 2.5


def raycast(pose, walls, rng=None, sigma=0.001):
    """Ranges from a sensor at pose (x, y, th) against segments ((x0,y0),(x1,y1))."""
    x, y, th = pose
    a = AMIN + AINC * np.arange(N) + th
    r = np.full(N, np.inf)
    for (x0, y0), (x1, y1) in walls:
        ex, ey = x1 - x0, y1 - y0
        for i, ang in enumerate(a):
            dx, dy = math.cos(ang), math.sin(ang)
            den = dx * ey - dy * ex
            if abs(den) < 1e-12:
                continue
            t = ((x0 - x) * ey - (y0 - y) * ex) / den
            u = ((x0 - x) * dy - (y0 - y) * dx) / den
            if t > 0 and 0 <= u <= 1:
                r[i] = min(r[i], t)
    r[r >= RMAX] = np.inf
    if rng is not None:
        r = r + rng.normal(0, sigma, N)
    return r


ROOM = [((-1.5, -1.2), (2.0, -1.2)), ((2.0, -1.2), (2.0, 1.4)), ((2.0, 1.4), (-1.5, 1.4)),
        ((-1.5, 1.4), (-1.5, -1.2)), ((0.6, 0.3), (1.0, 0.3)), ((1.0, 0.3), (1.0, 0.7))]
CORRIDOR = [((-50, 1.8), (50, 1.8)), ((-50, -1.8), (50, -1.8))]


def ref_at(pose, walls, seed):
    return make_reference(raycast(pose, walls, np.random.default_rng(seed)), AMIN, AINC, RMIN, RMAX)


def cur_at(pose, walls, seed):
    pts, _, _ = scan_points(raycast(pose, walls, np.random.default_rng(seed)), AMIN, AINC, RMIN, RMAX)
    return pts


def test_recovers_translation_and_rotation_in_a_room():
    ref = ref_at((0, 0, 0), ROOM, 1)
    true = (0.04, -0.015, math.radians(2.0))
    r = match(ref, cur_at(true, ROOM, 2), init=(0, 0, 0))
    assert r.ok
    assert abs(r.pose[0] - true[0]) < 3e-3 and abs(r.pose[1] - true[1]) < 3e-3
    assert abs(r.pose[2] - true[2]) < math.radians(0.1)
    sd = np.sqrt(np.diag(r.cov))
    assert np.all(sd < 0.01)


def test_corridor_is_degenerate_along_axis_only():
    ref = ref_at((0, 0, 0), CORRIDOR, 3)
    r = match(ref, cur_at((0.05, 0.01, 0.0), CORRIDOR, 4), init=(0, 0, 0))
    assert r.degenerate < 0.02
    sd = np.sqrt(np.diag(r.cov))
    assert sd[0] > 1.0 and sd[1] < 0.01              # along-wall unknown, across-wall known
    assert abs(r.pose[0]) < 1e-3                     # along-wall: keeps the initial guess
    assert abs(r.pose[1] - 0.01) < 3e-3              # lateral and heading still measured
    assert abs(r.pose[2]) < math.radians(0.1)


def test_bad_initial_guess_within_gate_converges():
    ref = ref_at((0, 0, 0), ROOM, 5)
    true = (0.10, 0.05, math.radians(-4))
    r = match(ref, cur_at(true, ROOM, 6), init=(0.0, 0.0, 0.0))
    assert r.ok and abs(r.pose[0] - 0.10) < 5e-3 and abs(r.pose[1] - 0.05) < 5e-3


def test_corridor_keeps_prediction_along_axis():
    """If the EKF predicts the true along-corridor motion, the result keeps it."""
    ref = ref_at((0, 0, 0), CORRIDOR, 9)
    r = match(ref, cur_at((0.05, 0.0, 0.0), CORRIDOR, 10), init=(0.05, 0, 0))
    assert abs(r.pose[0] - 0.05) < 1e-3


def test_too_few_points_not_ok():
    walls = [((1.0, -0.2), (1.0, 0.2))]
    ref = ref_at((0, 0, 0), walls, 7)
    r = match(ref, cur_at((0.01, 0, 0), walls, 8))
    assert not r.ok
