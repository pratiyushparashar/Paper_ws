"""Unit tests for zupt.core.lidar (Tier 3 change fraction + stuck hypothesis test)."""
import math

import numpy as np

from zupt.core.lidar import ScanMotionCheck

AMIN, AINC, N = -math.radians(135), math.radians(0.5), 541
RMIN, RMAX = 0.1, 2.5


def raycast(x, walls, rng=None, sigma=0.001):
    """Ranges from sensor at (x, 0), heading +x, against axis-aligned segments
    walls = [((x0,y0),(x1,y1)), ...]."""
    a = AMIN + AINC * np.arange(N)
    r = np.full(N, np.inf)
    for (x0, y0), (x1, y1) in walls:
        for i, ang in enumerate(a):
            dx, dy = math.cos(ang), math.sin(ang)
            if y0 == y1 and abs(dy) > 1e-9:
                t = y0 / dy
                px = x + t * dx
                if t > 0 and min(x0, x1) <= px <= max(x0, x1):
                    r[i] = min(r[i], t)
            if x0 == x1 and abs(dx) > 1e-9:
                t = (x0 - x) / dx
                py = t * dy
                if t > 0 and min(y0, y1) <= py <= max(y0, y1):
                    r[i] = min(r[i], t)
    r[r >= RMAX] = np.inf
    if rng is not None:
        r = r + rng.normal(0, sigma, N)
    return r


CORRIDOR = [((-50, 1.8), (50, 1.8)), ((-50, -1.8), (50, -1.8))]
END_WALL = CORRIDOR + [((1.5, -5), (1.5, 5))]


def feed(chk, ranges_list, claim_v, dt=0.1):
    out = []
    for k, r in enumerate(ranges_list):
        if k:
            for _ in range(10):
                chk.add_odom(claim_v, 0.0, dt / 10)
        out.append(chk.step(k * dt, r, AMIN, AINC, RMIN, RMAX))
    return out


def test_still_scan_is_unchanged():
    rng = np.random.default_rng(0)
    ev = feed(ScanMotionCheck(), [raycast(0.0, END_WALL, rng) for _ in range(10)], 0.0)
    assert all(e.changed == 0.0 for e in ev[1:])
    assert all(e.out0 < 0.02 for e in ev)


def test_stuck_against_wall_contradicts_wheels():
    rng = np.random.default_rng(1)
    ev = feed(ScanMotionCheck(), [raycast(0.0, END_WALL, rng) for _ in range(15)], 0.3)
    last = ev[-1]
    assert last.claim > 0.3 and last.ref_age >= 1.0
    assert last.out0 < 0.02 and last.out1 - last.out0 > 0.1


def test_featureless_corridor_refuses():
    """Wheels claim 0.3 m/s along a rib-less corridor, robot actually still: the scan cannot
    distinguish the hypotheses, so the margin must stay small (no stuck evidence)."""
    rng = np.random.default_rng(2)
    ev = feed(ScanMotionCheck(), [raycast(0.0, CORRIDOR, rng) for _ in range(15)], 0.3)
    assert ev[-1].claim > 0.3
    assert ev[-1].out1 - ev[-1].out0 < 0.05


def test_moving_toward_wall_changes_scan_and_resets_reference():
    rng = np.random.default_rng(3)
    ev = feed(ScanMotionCheck(), [raycast(0.03 * k, END_WALL, rng) for k in range(15)], 0.3)
    assert all(e.changed > 0.02 for e in ev[1:])
    assert max(e.ref_age for e in ev) <= 0.2           # reference keeps resetting
    assert all(e.claim < 0.1 for e in ev)              # so the stuck check never sees a claim
