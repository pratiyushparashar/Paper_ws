"""
Run the 6-state EKF (+ detector) over a recorded session and score it against ground truth.

    python3 -m zupt.ekf.run --session 20261005_000755 20261003_022102 20261004_235739

Variants (same data, same parameters, one switch each):
    no_zupt        odometry + gyro only
    zupt           + ZUPT (v = 0) when the detector commits
    zupt_zaru      + ZARU (omega = 0) when the detector commits
    *_wheels_distrusted   same three with wheel yaw rate distrusted (R_odom_w x1e4), i.e.
                          the bias is no longer observable from wheels-vs-gyro. Isolates what
                          ZARU contributes when wheel odometry cannot be trusted (slip/skid).

Ground truth is used only for scoring (and for the reference "true bias" = mean gyro while
GT says still). It never enters the detector or the filter.
"""
import argparse
import dataclasses
import math
import os
import sys

import numpy as np

from zupt.core.config import load_config
from zupt.core.detector import Evidence, ImuSample, StationarityDetector
from zupt.ekf.ekf6 import Ekf6, IB, ITH, IV, IX, IY, load_ekf_params, wrap
from zupt.replay import gt_truth, load_session, quat_to_rpy

VARIANTS = {
    "no_zupt":                     dict(zupt=False, zaru=False, distrust=False),
    "zupt":                        dict(zupt=True,  zaru=False, distrust=False),
    "zupt_zaru":                   dict(zupt=True,  zaru=True,  distrust=False),
    "no_zupt_wheels_distrusted":   dict(zupt=False, zaru=False, distrust=True),
    "zupt_wheels_distrusted":      dict(zupt=True,  zaru=False, distrust=True),
    "zupt_zaru_wheels_distrusted": dict(zupt=True,  zaru=True,  distrust=True),
}


def run_variant(sess, det_cfg, ekf_p, zupt, zaru, distrust):
    imu, odom, cmd, gt = sess
    p = dataclasses.replace(ekf_p, r_odom_w=ekf_p.r_odom_w * 1e4) if distrust else ekf_p
    roll, pitch, _ = quat_to_rpy(imu.orient_x, imu.orient_y, imu.orient_z, imu.orient_w)
    _, _, gyaw = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    ekf = Ekf6(p, gt.gt_x.iat[0], gt.gt_y.iat[0], float(gyaw.iat[0]))   # init from first GT pose
    det = StationarityDetector(det_cfg, have_lidar=False, logger=lambda *_: None)

    # merged, time-ordered event stream: IMU (50 Hz) and odometry (100 Hz)
    ev = [(t, 0, k) for k, t in enumerate(imu.t.values)] + [(t, 1, k) for k, t in enumerate(odom.t.values)]
    ev.sort()
    ct = cmd.t.values if cmd is not None and len(cmd) else np.array([])
    last_odom = None
    n_zupt = 0
    log = []
    for t, kind, k in ev:
        ekf.predict(t)
        if kind == 1:
            last_odom = (odom.linear_x.iat[k], odom.angular_z.iat[k])
            ekf.update_odom(*last_odom)
            continue
        ekf.update_gyro(imu.angular_vel_z.iat[k])
        e = Evidence()
        ci = np.searchsorted(ct, t, side="right") - 1
        if ci >= 0:
            e.v_cmd, e.w_cmd = cmd.linear_x.iat[ci], cmd.angular_z.iat[ci]
        if last_odom is not None:
            e.v_odom, e.w_odom = last_odom
        s = ImuSample(t, imu.angular_vel_z.iat[k], imu.linear_acc_x.iat[k], imu.linear_acc_y.iat[k],
                      imu.linear_acc_z.iat[k], roll.iat[k], pitch.iat[k])
        o = det.step(s, e)
        if o.apply_update:
            if zupt:
                ekf.update_zupt(o.r_scale); n_zupt += 1
            if zaru:
                ekf.update_zaru(o.r_scale)
        log.append((t, ekf.x[IX], ekf.x[IY], ekf.x[ITH], ekf.x[IV], ekf.x[IB], ekf.bias_std, o.stationary))
    return np.array(log, dtype=float), n_zupt


def score(log, gt, imu):
    t = log[:, 0]
    _, _, gyaw = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    gyaw = np.unwrap(gyaw.values)
    gx = np.interp(t, gt.t.values, gt.gt_x.values)
    gy = np.interp(t, gt.t.values, gt.gt_y.values)
    gth = np.interp(t, gt.t.values, gyaw)
    pe = np.hypot(log[:, 1] - gx, log[:, 2] - gy)
    he = np.degrees(np.abs(np.vectorize(wrap)(log[:, 3] - gth)))
    return dict(pos_rmse=float(np.sqrt(np.mean(pe ** 2))), pos_final=float(pe[-1]),
                head_rmse=float(np.sqrt(np.mean(he ** 2))), head_max=float(he.max()),
                head_final=float(he[-1]), bias_final=float(log[-1, 5]), bias_std=float(log[-1, 6]),
                dur=float(t[-1] - t[0]))


def true_bias(imu, gt):
    still, _ = gt_truth(imu.t.values, gt)
    return float(imu.angular_vel_z.values[still].mean()) if still.any() else math.nan


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", nargs="+", required=True)
    ap.add_argument("--data", default=os.path.expanduser("~/paper_ws/data/raw"))
    ap.add_argument("--det-params", default=os.path.expanduser("~/paper_ws/config/zupt_params_dev.yaml"))
    ap.add_argument("--ekf-params", default=os.path.expanduser("~/paper_ws/config/ekf_params_dev.yaml"))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args(argv)
    det_cfg = load_config(a.det_params)
    ekf_p = load_ekf_params(a.ekf_params)
    print("NOTE: provisional detector + EKF parameters (development check, not paper results)")
    for sid in a.session:
        sess = load_session(a.data, sid)
        tb = true_bias(sess[0], sess[3])
        print(f"\n=== {sid}  (reference bias from GT-still periods: {tb:+.5f} rad/s) ===")
        print(f"{'variant':30s} {'pos RMSE':>9s} {'pos fin':>8s} {'hdg RMSE':>9s} {'hdg max':>8s} "
              f"{'hdg fin':>8s} {'bias est':>10s} {'±1σ':>8s} {'#ZUPT':>6s}")
        for name in a.variants:
            log, n = run_variant(sess, det_cfg, ekf_p, **VARIANTS[name])
            m = score(log, sess[3], sess[0])
            print(f"{name:30s} {m['pos_rmse']:8.3f}m {m['pos_final']:7.3f}m {m['head_rmse']:8.2f}° "
                  f"{m['head_max']:7.2f}° {m['head_final']:7.2f}° {m['bias_final']:+10.5f} "
                  f"{m['bias_std']:8.5f} {n:6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
