#!/usr/bin/env python3
"""
Check whether a recording's IMU captures velocity changes (step 5c sim-fidelity check).

    python3 tools/check_imu_integration.py --session <id> [--data ~/paper_ws/data/raw]

For every commanded speed change it compares the integrated forward acceleration (after the
loader's averaging to 50 Hz) with the ground-truth speed change, and reports the raw IMU rate,
dropped messages and any ground-truth stop the IMU did not register (wall impacts).
Pass criterion used in docs/step5c.md: |∫a dt - Δv_gt| <= 0.02 m/s for every start/stop.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from zupt.replay import load_session  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--data", default=os.path.expanduser("~/paper_ws/data/raw"))
    a = ap.parse_args()
    raw = pd.read_csv(os.path.join(a.data, f"imu_{a.session}.csv"), usecols=["time_sec", "time_nsec"])
    tr = raw.time_sec + raw.time_nsec * 1e-9
    d = np.diff(tr.values)
    rate = 1 / np.median(d)
    print(f"raw IMU: {len(raw)} msgs, median rate {rate:.0f} Hz, gaps > 2 periods: "
          f"{int((d > 2 / rate).sum())} (logger dropping messages if > 0)")
    imu, odom, cmd, gt = load_session(a.data, a.session)
    t = imu.t.values
    # forward specific force with gravity removed (raw accel_x contains g*sin(pitch) on slopes)
    from zupt.core.detector import gravity_compensated_horizontal
    from zupt.replay import quat_to_rpy
    roll, pitch, _ = quat_to_rpy(imu.orient_x, imu.orient_y, imu.orient_z, imu.orient_w)
    ax_h = np.array([gravity_compensated_horizontal(x, y, z, r, p)[0] for x, y, z, r, p in
                     zip(imu.linear_acc_x, imu.linear_acc_y, imu.linear_acc_z, roll, pitch)])
    from zupt.evaluation.eval_5b import gt_body_rates
    tg = gt.t.values
    u_gt, w_gt = gt_body_rates(gt, tg)
    sp = np.hypot(np.diff(gt.gt_x), np.diff(gt.gt_y)) / np.diff(gt.t)
    worst = 0.0
    print("\n  t_cmd     cmd v   ∫a dt   Δv_gt   error   (straight-line speed changes only)")
    for k in range(len(cmd)):
        t0 = cmd.t.iat[k]
        t1 = min(t0 + 1.0, cmd.t.iat[k + 1] - 0.05) if k + 1 < len(cmd) else t0 + 1.0
        if t1 - t0 < 0.3 or np.max(np.abs(np.interp(np.linspace(t0 - 0.1, t1, 30), tg, w_gt))) > 0.05:
            continue                                   # too short or turning: not a clean test
        dv_gt = float(np.interp(t1, tg, u_gt) - np.interp(t0 - 0.1, tg, u_gt))
        if abs(dv_gt) < 0.05:
            continue
        m = (t > t0 - 0.1) & (t <= t1)
        dv = float(np.sum(ax_h[m]) * np.median(np.diff(t)))
        err = dv - dv_gt
        worst = max(worst, abs(err))
        print(f"  {t0:7.2f}  {cmd.linear_x.iat[k]:+6.3f}  {dv:+6.3f}  {dv_gt:+6.3f}  {err:+6.3f}")
    # stops the IMU did not see: GT speed drops > 0.2 m/s within 0.2 s with no command change
    unseen = 0
    for i in np.flatnonzero((sp[:-10] > 0.2) & (sp[10:] < 0.02)):
        tt = tg[i + 1]
        if np.any(np.abs(cmd.t.values - tt) < 0.5):
            continue
        m = (t > tt - 0.1) & (t <= tt + 0.5)
        dv = float(np.sum(ax_h[m]) * np.median(np.diff(t)))
        dv_gt = float(np.interp(tt + 0.5, tg, u_gt) - np.interp(tt - 0.1, tg, u_gt))
        print(f"  uncommanded stop at {tt:.2f} s (impact?): GT Δv {dv_gt:+.3f}, IMU ∫a dt {dv:+.3f}")
        unseen += abs(dv - dv_gt) > 0.05
        break
    print(f"\nworst start/stop error {worst:.3f} m/s -> {'PASS' if worst <= 0.02 else 'FAIL'} (<= 0.02)")
    if unseen:
        print("impact NOT captured by the IMU")


if __name__ == "__main__":
    main()
