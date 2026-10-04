"""
DEVELOPMENT evaluation of ZUPT/ZARU (not paper results).

    python3 -m zupt.evaluation.dev_eval [--ws-mobile-scripts ~/ws_mobile/scripts]

Part 1 — nominal data, project baselines vs the new EKF:
    B1  raw wheel odometry (dead reckoning)
    B2  existing 3-state Fixed EKF (ws_mobile ekf_fusion.run_ekf_core) with Gazebo's EXACT IMU heading
        (privileged: real IMUs do not give drift-free yaw)
    B3  same Fixed EKF, heading from integrated raw gyro (realistic, decision D8)
    E0  new 6-state EKF, no ZUPT/ZARU
    E1  new 6-state EKF + ZUPT + ZARU
Part 2 — odometry faults injected offline into the flat-floor session (wheels only):
    noise   v, w += N(0, 0.2) on every odometry sample           oracle r_odom = 0.25
    frozen  odometry holds its last value for 20 s (dropout)      oracle r_odom = 0 in window
    scale   v x 1.3 (over-reporting, slip-like)                   oracle r_odom = 0.77
  each run with r_odom = 1 (no reliability information) and with the ORACLE r_odom (upper
  bound for what the LSTM can deliver), without and with ZUPT+ZARU.

Caveats printed with the results: parameters are provisional and were tuned on these same
sessions; 3 short sessions; oracle reliability is an upper bound, not the LSTM.
"""
import argparse
import importlib
import math
import os
import sys

import numpy as np
import pandas as pd

from zupt.core.config import load_config
from zupt.core.detector import Evidence, ImuSample, StationarityDetector
from zupt.ekf.ekf6 import Ekf6, ITH, IX, IY, load_ekf_params, wrap
from zupt.replay import load_session, quat_to_rpy

SESSIONS = {"20261005_000755": "gyro bias, still",
            "20261003_022102": "flat floor, 13 stops",
            "20261004_235739": "slip patch"}


# ------------------------------------------------------------------ metrics
def gt_on(t, gt):
    _, _, gyaw = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    gyaw = np.unwrap(gyaw.values)
    return (np.interp(t, gt.t.values, gt.gt_x.values), np.interp(t, gt.t.values, gt.gt_y.values),
            np.interp(t, gt.t.values, gyaw))


def metrics(t, x, y, th, gt):
    gx, gy, gth = gt_on(t, gt)
    pe = np.hypot(x - gx, y - gy)
    he = np.degrees(np.abs((th - gth + np.pi) % (2 * np.pi) - np.pi))
    return dict(pos=float(np.sqrt(np.mean(pe ** 2))), pos_fin=float(pe[-1]),
                hdg=float(np.sqrt(np.mean(he ** 2))), hdg_fin=float(he[-1]))


# ------------------------------------------------------------------ baselines
def baseline_odom(sess):
    imu, odom, cmd, gt = sess
    _, _, oyaw = quat_to_rpy(odom.orient_x, odom.orient_y, odom.orient_z, odom.orient_w)
    return metrics(odom.t.values, odom.pos_x.values, odom.pos_y.values, oyaw.values, gt)


def baseline_fixed_ekf(sess, ekf_fusion, exact_heading):
    imu, odom, cmd, gt = sess
    t = imu.t.values
    ox = np.interp(t, odom.t, odom.pos_x)
    oy = np.interp(t, odom.t, odom.pos_y)
    _, _, oyaw = quat_to_rpy(odom.orient_x, odom.orient_y, odom.orient_z, odom.orient_w)
    oth = np.interp(t, odom.t, np.unwrap(oyaw.values))
    if cmd is not None and len(cmd):
        ci = np.searchsorted(cmd.t.values, t, side="right") - 1
        v = np.where(ci >= 0, cmd.linear_x.values[np.clip(ci, 0, None)], 0.0)
        w = np.where(ci >= 0, cmd.angular_z.values[np.clip(ci, 0, None)], 0.0)
    else:
        v = w = np.zeros_like(t)
    _, _, gyaw0 = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    th0 = float(gyaw0.iat[0])
    if exact_heading:
        _, _, iyaw = quat_to_rpy(imu.orient_x, imu.orient_y, imu.orient_z, imu.orient_w)
        ith = np.unwrap(iyaw.values)
    else:
        ith = th0 + np.cumsum(imu.angular_vel_z.values * np.r_[0, np.diff(t)])
    out = ekf_fusion.run_ekf_core(ox, oy, oth, v, w, ith, t, gt.gt_x.iat[0], gt.gt_y.iat[0], th0)
    return metrics(t, out[0], out[1], out[2], gt)


# ------------------------------------------------------------------ new EKF with faults
def run_new(sess, det_cfg, ekf_p, zupt_zaru, odom_v, odom_w, r_odom):
    imu, odom, cmd, gt = sess
    roll, pitch, _ = quat_to_rpy(imu.orient_x, imu.orient_y, imu.orient_z, imu.orient_w)
    _, _, gyaw = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    ekf = Ekf6(ekf_p, gt.gt_x.iat[0], gt.gt_y.iat[0], float(gyaw.iat[0]))
    det = StationarityDetector(det_cfg, have_lidar=False, logger=lambda *_: None)
    ev = sorted([(tt, 0, k) for k, tt in enumerate(imu.t.values)] +
                [(tt, 1, k) for k, tt in enumerate(odom.t.values)])
    ct = cmd.t.values if cmd is not None and len(cmd) else np.array([])
    last = None
    log = []
    for tt, kind, k in ev:
        ekf.predict(tt)
        if kind == 1:
            last = (odom_v[k], odom_w[k], r_odom[k])
            ekf.update_odom(odom_v[k], odom_w[k], r_odom=r_odom[k])
            continue
        ekf.update_gyro(imu.angular_vel_z.iat[k])
        e = Evidence()
        ci = np.searchsorted(ct, tt, side="right") - 1
        if ci >= 0:
            e.v_cmd, e.w_cmd = cmd.linear_x.iat[ci], cmd.angular_z.iat[ci]
        if last is not None:
            e.v_odom, e.w_odom, e.r_odom = last
        o = det.step(ImuSample(tt, imu.angular_vel_z.iat[k], imu.linear_acc_x.iat[k],
                               imu.linear_acc_y.iat[k], imu.linear_acc_z.iat[k],
                               roll.iat[k], pitch.iat[k]), e)
        if zupt_zaru and o.apply_update:
            ekf.update_zupt(o.r_scale)
            ekf.update_zaru(o.r_scale)
        log.append((tt, ekf.x[IX], ekf.x[IY], ekf.x[ITH]))
    log = np.array(log)
    return metrics(log[:, 0], log[:, 1], log[:, 2], log[:, 3], gt)


def faults(odom, rng):
    v, w, n = odom.linear_x.values.astype(float), odom.angular_z.values.astype(float), len(odom)
    out = {"none": (v, w, np.ones(n), np.ones(n))}
    out["noise"] = (v + rng.normal(0, 0.2, n), w + rng.normal(0, 0.2, n), np.ones(n), np.full(n, 0.25))
    t = odom.t.values
    moving = np.where(np.abs(v) > 0.1)[0]
    t0 = t[moving[len(moving) // 3]]                       # start of window: during driving
    win = (t >= t0) & (t < t0 + 20.0)
    i0 = np.argmax(win)
    vf, wf = v.copy(), w.copy()
    vf[win], wf[win] = v[i0 - 1], w[i0 - 1]
    out["frozen"] = (vf, wf, np.ones(n), np.where(win, 0.0, 1.0))
    out["scale"] = (v * 1.3, w, np.ones(n), np.full(n, 1 / 1.3))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/paper_ws/data/raw"))
    ap.add_argument("--det-params", default=os.path.expanduser("~/paper_ws/config/zupt_params_dev.yaml"))
    ap.add_argument("--ekf-params", default=os.path.expanduser("~/paper_ws/config/ekf_params_dev.yaml"))
    ap.add_argument("--ws-mobile-scripts", default=os.path.expanduser("~/ws_mobile/scripts"))
    a = ap.parse_args(argv)
    det_cfg, ekf_p = load_config(a.det_params), load_ekf_params(a.ekf_params)

    sys.path.insert(0, a.ws_mobile_scripts)           # read-only import of the existing EKF
    saved = sys.argv; sys.argv = [saved[0]]
    ekf_fusion = importlib.import_module("ekf_fusion")
    sys.argv = saved

    print("DEVELOPMENT EVALUATION — provisional parameters tuned on these same sessions;"
          " not paper results.\n")
    print("PART 1 — nominal sessions (position RMSE m / heading RMSE deg)")
    rows = []
    for sid, desc in SESSIONS.items():
        sess = load_session(a.data, sid)
        n = len(sess[1])
        ones = np.ones(n)
        r = {"B1 wheel odometry": baseline_odom(sess),
             "B2 Fixed EKF, exact Gazebo heading": baseline_fixed_ekf(sess, ekf_fusion, True),
             "B3 Fixed EKF, gyro-integrated heading": baseline_fixed_ekf(sess, ekf_fusion, False),
             "E0 new EKF, no ZUPT/ZARU": run_new(sess, det_cfg, ekf_p, False, sess[1].linear_x.values,
                                                 sess[1].angular_z.values, ones),
             "E1 new EKF + ZUPT + ZARU": run_new(sess, det_cfg, ekf_p, True, sess[1].linear_x.values,
                                                 sess[1].angular_z.values, ones)}
        for name, m in r.items():
            rows.append((desc, name, m["pos"], m["hdg"], m["pos_fin"], m["hdg_fin"]))
    df = pd.DataFrame(rows, columns=["session", "method", "pos_rmse", "hdg_rmse", "pos_final", "hdg_final"])
    for sess_name, g in df.groupby("session", sort=False):
        print(f"\n  {sess_name}")
        for _, rr in g.iterrows():
            print(f"    {rr.method:40s} pos {rr.pos_rmse:7.3f}  hdg {rr.hdg_rmse:7.2f}"
                  f"   (final {rr.pos_final:6.3f} m, {rr.hdg_final:6.2f}°)")

    print("\nPART 2 — wheel-odometry faults on the flat-floor session (pos RMSE m / heading RMSE deg)")
    sess = load_session(a.data, "20261003_022102")
    F = faults(sess[1], np.random.default_rng(0))
    print(f"    {'fault':8s} {'reliability':12s} {'no ZUPT/ZARU':>20s} {'+ZUPT+ZARU':>20s} {'pos gain':>9s}")
    for fname, (v, w, r1, r_or) in F.items():
        for rname, rr in (("r_odom=1", r1), ("oracle r", r_or)):
            if fname == "none" and rname == "oracle r":
                continue
            m0 = run_new(sess, det_cfg, ekf_p, False, v, w, rr)
            m1 = run_new(sess, det_cfg, ekf_p, True, v, w, rr)
            gain = 100 * (m0["pos"] - m1["pos"]) / m0["pos"]
            print(f"    {fname:8s} {rname:12s} {m0['pos']:8.3f} / {m0['hdg']:6.2f}°  "
                  f"{m1['pos']:8.3f} / {m1['hdg']:6.2f}°  {gain:+8.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
