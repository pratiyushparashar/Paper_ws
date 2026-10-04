"""
DEVELOPMENT evaluation of LiDAR Tier 3 + stuck detection (step 5b). Not paper results.

    python3 -m zupt.evaluation.eval_stuck [--data ...]

Part A — normal driving (Tests C/D as recorded): the stuck path must never commit, and Tier 3
         must not create false stationary samples.
Part B — synthetic stuck: in every ground-truth-still stretch >= 3 s the command and the wheel
         odometry are overwritten with 0.3 m/s ("wheels spinning, robot pinned"), from 0.5 s
         after the stretch begins. IMU and LiDAR stay the REAL recorded still data. This tests
         the logic; it does not reproduce the vibration of real spinning wheels (Test E does).
         EKF9 with trusted wheels, IMU accel bias injected; stuck path on vs off.
"""
import argparse
import math
import os
import sys

import numpy as np

from zupt.core.config import load_config
from zupt.core.detector import Evidence, ImuSample, StationarityDetector, gravity_compensated_horizontal
from zupt.ekf.ekf9 import Ekf9, ITH, IX, IY, load_ekf9_params
from zupt.evaluation.dev_eval import gt_on
from zupt.replay import gt_truth, lidar_evidence, load_scans, load_session, quat_to_rpy
import dataclasses

SESSIONS = {"20261005_024700": "Test C slope + ice", "20261005_025210": "Test D rumble strips"}
BIAS = (0.03, -0.02)


def still_stretches(gt, min_len=3.0):
    t = gt.t.values
    still, _ = gt_truth(t, gt)
    out, k = [], 0
    while k < len(t):
        if still[k]:
            j = k
            while j + 1 < len(t) and still[j + 1]:
                j += 1
            if t[j] - t[k] >= min_len:
                out.append((t[k], t[j]))
            k = j + 1
        else:
            k += 1
    return out


def replay(sess, scans, det_cfg, p9, stuck_windows=(), bias=BIAS, use_lidar=True):
    imu, odom, cmd, gt = sess
    in_stuck = lambda tt: any(a <= tt < b for a, b in stuck_windows)
    ov = odom.linear_x.values.copy()
    ow = odom.angular_z.values.copy()
    for k, tt in enumerate(odom.t.values):
        if in_stuck(tt):
            ov[k] = 0.3
    lev = lidar_evidence(scans, odom, odom_v=ov, odom_w=ow) if use_lidar else []
    lt = np.array([e.t for e in lev])
    roll, pitch, _ = quat_to_rpy(imu.orient_x, imu.orient_y, imu.orient_z, imu.orient_w)
    roll, pitch = roll.values, pitch.values
    ax = imu.linear_acc_x.values + bias[0]
    ay = imu.linear_acc_y.values + bias[1]
    az, gz = imu.linear_acc_z.values, imu.angular_vel_z.values
    _, _, gyaw = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    f = Ekf9(p9, gt.gt_x.iat[0], gt.gt_y.iat[0], float(gyaw.iat[0]))
    det = StationarityDetector(det_cfg, have_lidar=use_lidar, logger=lambda *_: None)
    ev = sorted([(tt, 0, k) for k, tt in enumerate(imu.t.values)] +
                [(tt, 1, k) for k, tt in enumerate(odom.t.values)])
    ct = cmd.t.values
    last, log = None, []
    for tt, kind, k in ev:
        f.predict(tt)
        if kind == 1:
            last = (ov[k], ow[k])
            f.update_odom(ov[k], ow[k])
            continue
        hx, hy = gravity_compensated_horizontal(ax[k], ay[k], az[k], roll[k], pitch[k])
        f.set_accel(hx, hy)
        f.update_gyro(gz[k])
        f.update_nhc(0.0)
        e = Evidence()
        ci = np.searchsorted(ct, tt, side="right") - 1
        if ci >= 0:
            e.v_cmd, e.w_cmd = cmd.linear_x.iat[ci], cmd.angular_z.iat[ci]
        if in_stuck(tt):
            e.v_cmd = 0.3
        if last is not None:
            e.v_odom, e.w_odom = last
        li = np.searchsorted(lt, tt, side="right") - 1
        if li >= 0:
            L = lev[li]
            e.lidar_changed, e.lidar_age = L.changed, tt - L.t
            e.lidar_out0, e.lidar_out1 = L.out0, L.out1
            e.lidar_ref_age, e.lidar_claim = L.ref_age, L.claim
        o = det.step(ImuSample(tt, gz[k], ax[k], ay[k], az[k], roll[k], pitch[k]), e)
        if o.apply_update:
            f.update_zupt(o.r_scale)
            f.update_zaru(o.r_scale)
        log.append((tt, o.stationary, o.reason, f.x[IX], f.x[IY], f.x[ITH], e.v_cmd))
    return log


def summarize(log, gt, stuck_windows):
    t = np.array([r[0] for r in log])
    st = np.array([r[1] for r in log])
    reason = np.array([r[2] for r in log])
    x = np.array([r[3] for r in log]); y = np.array([r[4] for r in log])
    still, v_gt = gt_truth(t, gt)
    gx, gy, _ = gt_on(t, gt)
    pe = np.hypot(x - gx, y - gy)
    res = dict(false_st=int((st & ~still).sum()),
               max_v_false=float(v_gt[st & ~still].max()) if (st & ~still).any() else 0.0,
               stuck_commits=int((reason == "stuck").sum()),
               false_stuck=int(((reason == "stuck") & ~still).sum()),
               pos_rmse=float(np.sqrt(np.mean(pe ** 2))), pos_fin=float(pe[-1]))
    lat, cover, drift = [], [], []
    for a, b in stuck_windows:
        m = (t >= a) & (t < b)
        s_ = (reason == "stuck") & m
        lat.append(t[s_][0] - a if s_.any() else math.nan)
        cover.append(s_.sum() / max(m.sum(), 1))
        drift.append(pe[m][-1] - pe[m][0] if m.any() else math.nan)
    res.update(latency=lat, coverage=cover, drift=drift)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/paper_ws/data/raw"))
    ap.add_argument("--det-params", default=os.path.expanduser("~/paper_ws/config/zupt_params_dev.yaml"))
    ap.add_argument("--ekf9-params", default=os.path.expanduser("~/paper_ws/config/ekf9_params_dev.yaml"))
    a = ap.parse_args(argv)
    det_cfg, p9 = load_config(a.det_params), load_ekf9_params(a.ekf9_params)
    off_cfg = dataclasses.replace(det_cfg, stuck_enable=False)
    print("DEVELOPMENT EVALUATION (step 5b, LiDAR + stuck) — provisional; not paper results.\n")
    for sid, name in SESSIONS.items():
        sess = load_session(a.data, sid)
        scans = load_scans(a.data, sid)
        gt = sess[3]
        print(f"  {name} ({sid})")
        rA0 = summarize(replay(sess, scans, det_cfg, p9, use_lidar=False), gt, [])
        rA1 = summarize(replay(sess, scans, det_cfg, p9), gt, [])
        print(f"    A normal driving   no LiDAR : false stationary {rA0['false_st']:3d}   pos RMSE {rA0['pos_rmse']:.3f}")
        print(f"                       LiDAR+stuck: false stationary {rA1['false_st']:3d}   "
              f"stuck commits {rA1['stuck_commits']}   pos RMSE {rA1['pos_rmse']:.3f}")
        win = [(s + 0.5, e) for s, e in still_stretches(gt)]
        rB0 = summarize(replay(sess, scans, off_cfg, p9, win), gt, win)
        rB1 = summarize(replay(sess, scans, det_cfg, p9, win), gt, win)
        print(f"    B synthetic stuck, {len(win)} windows ({sum(e - s for s, e in win):.1f} s total)")
        for i, (s, e) in enumerate(win):
            print(f"      window {i}: {e - s:4.1f} s   stuck OFF: pos error grows {rB0['drift'][i]:+.3f} m"
                  f"   stuck ON: {rB1['drift'][i]:+.3f} m  (detected after {rB1['latency'][i]:.2f} s,"
                  f" {100 * rB1['coverage'][i]:.0f}% of window)")
        print(f"      whole session pos RMSE: stuck OFF {rB0['pos_rmse']:.3f}   ON {rB1['pos_rmse']:.3f};"
              f"   false stuck samples {rB1['false_stuck']}, false stationary {rB1['false_st']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
