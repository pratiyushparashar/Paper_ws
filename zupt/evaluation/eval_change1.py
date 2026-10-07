"""
DEVELOPMENT evaluation of change 1 (LiDAR scan matching in the EKF). Not paper results.

    python3 -m zupt.evaluation.eval_change1 [--data ...] [--sessions ...]

Variants (all: IMU-driven EKF9, NHC, ZUPT/ZARU, LiDAR Tier 3 + stuck path; accel bias injected):
  5b           wheels trusted, no scan matching
  5b+5c        + retroactive slip correction at stops
  LiDAR        + scan matching (stochastic cloning), wheels trusted
  LiDAR+5c     + both
  LiDAR, no wheels   scan matching + IMU, wheel odometry not used at all
  oracle+LiDAR       wheels skipped where they disagree with ground truth (upper bound)
Also reported per session: scan-match acceptance, share of scans with an unobservable direction
(corridor-like), and the median LiDAR NIS (wheel/LiDAR disagreement shows up as large NIS).
"""
import argparse
import os
import sys

import numpy as np

from zupt.core.config import load_config
from zupt.core.detector import Evidence, ImuSample, StationarityDetector, gravity_compensated_horizontal
from zupt.core.lidar import scan_points
from zupt.core.scan_match import make_reference
from zupt.ekf.ekf9 import Ekf9, ITH, IX, IY, load_ekf9_params
from zupt.ekf.ekf_lidar import EkfLidar
from zupt.evaluation.dev_eval import metrics
from zupt.evaluation.eval_5c import mask_to_odom, oracle_gate
from zupt.replay import lidar_evidence, load_scans, load_session, quat_to_rpy
from zupt.segments import load_segment_config, run_segments

SESSIONS = {"20261005_024700": "Test C (50 Hz)",
            "20261005_025210": "Test D strips (50 Hz)",
            "20261005_035723": "Test E stuck (50 Hz)",
            "20261007_201210": "S1 slip patch (1 kHz)",
            "20261007_201651": "S2 Test C (1 kHz)",
            "20261007_202005": "S3 flat stop-go (1 kHz)"}
BIAS = (0.03, -0.02)


def pipeline(sess, scans, det_cfg, p9, use_lidar, gate=None, bias=BIAS, sm_cfg=None):
    imu, odom, cmd, gt = sess
    roll, pitch, _ = quat_to_rpy(imu.orient_x, imu.orient_y, imu.orient_z, imu.orient_w)
    roll, pitch = roll.values, pitch.values
    ax = imu.linear_acc_x.values + bias[0]
    ay = imu.linear_acc_y.values + bias[1]
    az, gz = imu.linear_acc_z.values, imu.angular_vel_z.values
    _, _, gyaw = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    x0, y0, th0 = gt.gt_x.iat[0], gt.gt_y.iat[0], float(gyaw.iat[0])
    f = EkfLidar(p9, x0, y0, th0, sm_cfg) if use_lidar else Ekf9(p9, x0, y0, th0)
    ov, ow = odom.linear_x.values, odom.angular_z.values
    gate = np.zeros(len(odom), bool) if gate is None else gate
    lev = lidar_evidence(scans, odom) if scans is not None else []
    lt = np.array([e.t for e in lev])
    det = StationarityDetector(det_cfg, have_lidar=scans is not None, logger=lambda *_: None)
    ev = ([(tt, 0, k) for k, tt in enumerate(imu.t.values)] +
          [(tt, 1, k) for k, tt in enumerate(odom.t.values)])
    if use_lidar and scans is not None:
        ev += [(tt, 2, k) for k, tt in enumerate(scans.t.values)]
    ev.sort()
    ct = cmd.t.values
    last = None
    T, X, Y, TH, ST, HX = [], [], [], [], [], []
    nis, n_ok, n_deg, n_sc = [], 0, 0, 0
    for tt, kind, k in ev:
        f.predict(tt)
        if kind == 1:
            last = (ov[k], ow[k], 0.0 if gate[k] else 1.0)
            if not gate[k]:
                f.update_odom(ov[k], ow[k])
            continue
        if kind == 2:
            r = np.array(scans.ranges.iat[k].split(";"), float)
            a = (scans.angle_min.iat[k], scans.angle_increment.iat[k],
                 scans.range_min.iat[k], scans.range_max.iat[k])
            pts, _, _ = scan_points(r, *a)
            m = f.scan(pts, make_reference(r, *a))
            if m is not None:
                n_sc += 1
                if m.ok:
                    n_ok += 1
                    sd = np.sqrt(np.diag(m.cov))
                    n_deg += bool(sd[0] > 1 or sd[1] > 1 or sd[2] > 1)
                    nis.append(f.last_innov["lidar"][1])
            continue
        hx, hy = gravity_compensated_horizontal(ax[k], ay[k], az[k], roll[k], pitch[k])
        f.set_accel(hx, hy)
        f.update_gyro(gz[k])
        f.update_nhc(0.0)
        e = Evidence()
        ci = np.searchsorted(ct, tt, side="right") - 1
        if ci >= 0:
            e.v_cmd, e.w_cmd = cmd.linear_x.iat[ci], cmd.angular_z.iat[ci]
        if last is not None:
            e.v_odom, e.w_odom, e.r_odom = last
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
        T.append(tt); X.append(f.x[IX]); Y.append(f.x[IY]); TH.append(f.x[ITH])
        ST.append(o.stationary); HX.append(hx)
    T = np.array(T)
    return dict(t=T, st=np.array(ST), hx=np.array(HX), gz=gz,
                m=metrics(T, np.array(X), np.array(Y), np.array(TH), gt),
                scan=dict(n=n_sc, ok=n_ok, deg=n_deg, nis=float(np.median(nis)) if nis else float("nan"),
                          nis95=float(np.percentile(nis, 95)) if nis else float("nan")))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/paper_ws/data/raw"))
    ap.add_argument("--det-params", default=os.path.expanduser("~/paper_ws/config/zupt_params_dev.yaml"))
    ap.add_argument("--ekf9-params", default=os.path.expanduser("~/paper_ws/config/ekf9_params_dev.yaml"))
    ap.add_argument("--seg-params", default=os.path.expanduser("~/paper_ws/config/segment_params_dev.yaml"))
    ap.add_argument("--sessions", nargs="*", default=list(SESSIONS))
    a = ap.parse_args(argv)
    det_cfg, p9 = load_config(a.det_params), load_ekf9_params(a.ekf9_params)
    sc = load_segment_config(a.seg_params)
    print("DEVELOPMENT EVALUATION (change 1, LiDAR scan matching) — provisional parameters; "
          f"accel bias {BIAS}; not paper results.\n")
    print(f"  {'session':26s} {'5b':>13s} {'5b+5c':>13s} {'LiDAR':>13s} {'LiDAR+5c':>13s} "
          f"{'LiDAR,no whl':>13s} {'oracle+LiDAR':>13s}   scans ok/degen  NIS med/95%")
    print(f"  {'':26s} " + " ".join([f"{'RMSE / final':>13s}"] * 6))
    for sid in a.sessions:
        sess = load_session(a.data, sid)
        scans = load_scans(a.data, sid)
        odom, gt = sess[1], sess[3]
        out = {}
        for name, lid in (("5b", False), ("LiDAR", True)):
            p1 = pipeline(sess, scans, det_cfg, p9, lid)
            segs = run_segments(sc, p1["t"], p1["hx"], p1["gz"], p1["st"], odom.t.values,
                                odom.linear_x.values)
            g = mask_to_odom(segs, odom.t.values)
            p2 = pipeline(sess, scans, det_cfg, p9, lid, gate=g) if g.any() else p1
            out[name], out[name + "+5c"] = p1, p2
        out["nowheels"] = pipeline(sess, scans, det_cfg, p9, True, gate=np.ones(len(odom), bool))
        out["oracle"] = pipeline(sess, scans, det_cfg, p9, True, gate=oracle_gate(odom, gt))
        cell = lambda k: f"{out[k]['m']['pos']:5.3f}/{out[k]['m']['pos_fin']:5.2f}"
        s = out["LiDAR"]["scan"]
        print(f"  {SESSIONS.get(sid, sid):26s} {cell('5b'):>13s} {cell('5b+5c'):>13s} {cell('LiDAR'):>13s} "
              f"{cell('LiDAR+5c'):>13s} {cell('nowheels'):>13s} {cell('oracle'):>13s}   "
              f"{s['ok']:4d}/{s['n']:4d} {s['deg']:4d}  {s['nis']:5.1f}/{s['nis95']:6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
