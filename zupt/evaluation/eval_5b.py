"""
DEVELOPMENT evaluation of step 5b (IMU-driven 9-state EKF + ZUPT/ZARU + NHC). Not paper results.

    python3 -m zupt.evaluation.eval_5b [--data ...] [--sessions ...]

Accelerometer bias is injected offline into the recorded body-frame accelerometer (bias is
additive, so this equals recording with that bias). Gazebo's own accel bias is ~1e-4 m/s^2;
realistic residual MEMS bias after a start-up calibration is ~0.01-0.03 m/s^2 (1-3 mg).

Wheel-odometry modes (what the reliability model could tell the filter):
    trust          r_odom = 1 always
    oracle-scale   r_odom = 0 where the wheels are wrong vs ground truth, R = R0/(r+eps)
    oracle-gate    same oracle, but the update is SKIPPED where r_odom = 0
    off            no wheel odometry at all (IMU + constraints only; worst case for the wheels)
Constraint variants: none | ZUPT+ZARU | NHC | all.
"""
import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

from zupt.core.config import load_config
from zupt.core.detector import Evidence, ImuSample, StationarityDetector, gravity_compensated_horizontal
from zupt.ekf.ekf9 import Ekf9, ITH, IX, IY, IBAX, IBAY, load_ekf9_params
from zupt.evaluation.dev_eval import metrics
from zupt.replay import load_session, quat_to_rpy

SESSIONS = {"20261003_022102": "flat floor, 13 stops",
            "20261004_235739": "slip patch",
            "20261005_024700": "Test C slope + ice slide",
            "20261005_025210": "Test D rumble strips"}
BIASES = [(0.03, -0.02), (-0.02, 0.03), (0.01, 0.025)]          # m/s^2, body x/y
VARIANTS = {"none": (False, False), "ZUPT+ZARU": (True, False), "NHC": (False, True), "all": (True, True)}


def gt_body_rates(gt, t):
    """Ground-truth forward speed of the base point and yaw rate, interpolated onto t."""
    _, _, yaw = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    yaw = np.unwrap(yaw.values)
    tt = gt.t.values
    dt = np.diff(tt)
    vx, vy = np.diff(gt.gt_x.values) / dt, np.diff(gt.gt_y.values) / dt
    ym = yaw[:-1]
    u = vx * np.cos(ym) + vy * np.sin(ym)
    w = np.diff(yaw) / dt
    tm = tt[:-1] + dt / 2
    return np.interp(t, tm, u), np.interp(t, tm, w)


def oracle_r(odom, gt, tol_v=0.05, tol_w=0.05):
    u, w = gt_body_rates(gt, odom.t.values)
    bad = (np.abs(odom.linear_x.values - u) > tol_v) | (np.abs(odom.angular_z.values - w) > tol_w)
    return np.where(bad, 0.0, 1.0)


def run9(sess, det_cfg, p9, zupt, nhc, odom_mode, bias=(0.0, 0.0), odom_override=None):
    """odom_override = (v, w, r_oracle) replaces the recorded odometry (offline fault injection)."""
    imu, odom, cmd, gt = sess
    roll, pitch, _ = quat_to_rpy(imu.orient_x, imu.orient_y, imu.orient_z, imu.orient_w)
    roll, pitch = roll.values, pitch.values
    ax = imu.linear_acc_x.values + bias[0]
    ay = imu.linear_acc_y.values + bias[1]
    az = imu.linear_acc_z.values
    gz = imu.angular_vel_z.values
    _, _, gyaw = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    f = Ekf9(p9, gt.gt_x.iat[0], gt.gt_y.iat[0], float(gyaw.iat[0]))
    det = StationarityDetector(det_cfg, have_lidar=False, logger=lambda *_: None)
    r_or = oracle_r(odom, gt) if odom_mode.startswith("oracle") else np.ones(len(odom))
    ov, ow = odom.linear_x.values, odom.angular_z.values
    if odom_override is not None:
        ov, ow, r_f = odom_override
        if odom_mode.startswith("oracle"):
            r_or = np.minimum(r_or, r_f)
    ev = sorted([(tt, 0, k) for k, tt in enumerate(imu.t.values)] +
                [(tt, 1, k) for k, tt in enumerate(odom.t.values)])
    ct = cmd.t.values if cmd is not None and len(cmd) else np.array([])
    last = None
    log, n_zupt = [], 0
    for tt, kind, k in ev:
        f.predict(tt)
        if kind == 1:
            r = r_or[k]
            last = (ov[k], ow[k], r)
            if odom_mode == "off" or (odom_mode == "oracle-gate" and r < 0.5):
                continue
            f.update_odom(ov[k], ow[k], r_odom=r)
            continue
        hx, hy = gravity_compensated_horizontal(ax[k], ay[k], az[k], roll[k], pitch[k])
        f.set_accel(hx, hy)
        f.update_gyro(gz[k])
        if nhc:
            f.update_nhc(0.0)
        e = Evidence()
        ci = np.searchsorted(ct, tt, side="right") - 1
        if ci >= 0:
            e.v_cmd, e.w_cmd = cmd.linear_x.iat[ci], cmd.angular_z.iat[ci]
        if last is not None and odom_mode != "off":
            e.v_odom, e.w_odom, e.r_odom = last
        o = det.step(ImuSample(tt, gz[k], ax[k], ay[k], az[k], roll[k], pitch[k]), e)
        if zupt and o.apply_update:
            f.update_zupt(o.r_scale)
            f.update_zaru(o.r_scale)
            n_zupt += 1
        log.append((tt, f.x[IX], f.x[IY], f.x[ITH]))
    log = np.array(log)
    m = metrics(log[:, 0], log[:, 1], log[:, 2], log[:, 3], gt)
    m["bax"], m["bay"], m["n_zupt"] = f.x[IBAX], f.x[IBAY], n_zupt
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/paper_ws/data/raw"))
    ap.add_argument("--det-params", default=os.path.expanduser("~/paper_ws/config/zupt_params_dev.yaml"))
    ap.add_argument("--ekf9-params", default=os.path.expanduser("~/paper_ws/config/ekf9_params_dev.yaml"))
    ap.add_argument("--sessions", nargs="*", default=list(SESSIONS))
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    det_cfg, p9 = load_config(a.det_params), load_ekf9_params(a.ekf9_params)
    rows = []
    for sid in a.sessions:
        sess = load_session(a.data, sid)
        for mode, (vname, (z, n)), b in itertools.product(
                ["trust", "oracle-scale", "oracle-gate", "off"], VARIANTS.items(), BIASES):
            m = run9(sess, det_cfg, p9, z, n, mode, b)
            rows.append(dict(session=SESSIONS.get(sid, sid), odom=mode, constraints=vname,
                             bias=f"{b[0]:+.2f},{b[1]:+.2f}", **m))
    df = pd.DataFrame(rows)
    if a.csv:
        df.to_csv(a.csv, index=False)
    print("DEVELOPMENT EVALUATION (step 5b) — provisional parameters; injected accel bias "
          f"{BIASES}; mean over bias draws. Not paper results.\n")
    g = df.groupby(["session", "odom", "constraints"], sort=False)[["pos", "pos_fin", "hdg"]].mean()
    for sess_name, block in g.groupby(level=0, sort=False):
        print(f"  {sess_name}")
        print(f"    {'odometry':13s} {'constraints':11s} {'pos RMSE':>9s} {'final':>8s} {'hdg RMSE':>9s}")
        for (_, mode, c), r in block.iterrows():
            print(f"    {mode:13s} {c:11s} {r.pos:9.3f} {r.pos_fin:8.3f} {r.hdg:8.2f}°")
        print()
    # ---------------- Part 2: wheel faults (flat-floor session), realistic middle case
    from zupt.evaluation.dev_eval import faults
    sid = "20261003_022102"
    if sid in a.sessions:
        sess = load_session(a.data, sid)
        F = faults(sess[1], np.random.default_rng(0))
        print("  wheel faults on the flat-floor session (pos RMSE m / heading RMSE deg; mean over bias draws)")
        print(f"    {'fault':7s} {'odometry':12s} {'NHC only':>16s} {'NHC+ZUPT+ZARU':>16s} {'ZUPT gain':>10s}")
        for fname, (v, w, _, r_f) in F.items():
            for mode in ("trust", "oracle-gate"):
                res = {}
                for vname in ("NHC", "all"):
                    z, n = VARIANTS[vname]
                    ms = [run9(sess, det_cfg, p9, z, n, mode, b, (v, w, r_f)) for b in BIASES]
                    res[vname] = (np.mean([m["pos"] for m in ms]), np.mean([m["hdg"] for m in ms]))
                g = 100 * (res["NHC"][0] - res["all"][0]) / res["NHC"][0]
                print(f"    {fname:7s} {mode:12s} {res['NHC'][0]:8.3f} / {res['NHC'][1]:5.2f}° "
                      f"{res['all'][0]:8.3f} / {res['all'][1]:5.2f}° {g:+9.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
