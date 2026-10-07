"""
DEVELOPMENT evaluation of the ZUPT-bounded segment check (step 5c). Not paper results.

    python3 -m zupt.evaluation.eval_5c [--data ...]

Pipeline per session (accelerometer bias injected, 3 draws):
  pass 1  "live":      EKF9, wheels trusted, NHC + ZUPT/ZARU (+ LiDAR stuck path when scans exist)
  segment check:       each segment judged at its closing stop -> slip flag, per-sample slip mask
  pass 2  "corrected": same filter re-run with the wheel updates SKIPPED on the slip-masked
                       samples (what a fixed-lag re-filter does at each stop; uses only data
                       available at that stop)
  oracle:              wheels skipped wherever they truly disagree with ground truth (upper bound)
Wheel-scale part: odometry v and w multiplied by 1.02 or 0.98 (wheel radius error); the scale is
estimated from the non-slip segments of the same session and the odometry re-scaled.
Ground-truth slip definitions (evaluation only): segment slip if |d_wheel - d_gt| >= 0.15 m;
sample slip if |v_wheel - u_gt| >= 0.1 m/s.
"""
import argparse
import os
import sys

import numpy as np

from zupt.core.config import load_config
from zupt.core.detector import Evidence, ImuSample, StationarityDetector, gravity_compensated_horizontal
from zupt.ekf.ekf9 import Ekf9, ITH, IX, IY, load_ekf9_params
from zupt.evaluation.dev_eval import metrics
from zupt.evaluation.eval_5b import gt_body_rates
from zupt.replay import lidar_evidence, load_scans, load_session, quat_to_rpy
from zupt.segments import load_segment_config, run_segments, scale_usable, wheel_scale

SESSIONS = {"20261003_022102": "flat floor, 13 stops",
            "20261004_235739": "slip patch",
            "20261005_024700": "Test C slope + ice slide",
            "20261005_025210": "Test D rumble strips",
            "20261005_035723": "Test E stuck at wall"}
BIASES = [(0.03, -0.02), (-0.02, 0.03), (0.01, 0.025)]
SLIP_SEG, SLIP_V = 0.15, 0.1


def pipeline(sess, scans, det_cfg, p9, bias, ov, ow, gate=None):
    """gate: boolean per odometry sample -> wheel update skipped (r_odom = 0 for the detector)."""
    imu, odom, cmd, gt = sess
    roll, pitch, _ = quat_to_rpy(imu.orient_x, imu.orient_y, imu.orient_z, imu.orient_w)
    roll, pitch = roll.values, pitch.values
    ax = imu.linear_acc_x.values + bias[0]
    ay = imu.linear_acc_y.values + bias[1]
    az, gz = imu.linear_acc_z.values, imu.angular_vel_z.values
    _, _, gyaw = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    f = Ekf9(p9, gt.gt_x.iat[0], gt.gt_y.iat[0], float(gyaw.iat[0]))
    lev = lidar_evidence(scans, odom, odom_v=ov, odom_w=ow) if scans is not None else []
    lt = np.array([e.t for e in lev])
    det = StationarityDetector(det_cfg, have_lidar=scans is not None, logger=lambda *_: None)
    gate = np.zeros(len(odom), bool) if gate is None else gate
    ev = sorted([(tt, 0, k) for k, tt in enumerate(imu.t.values)] +
                [(tt, 1, k) for k, tt in enumerate(odom.t.values)])
    ct = cmd.t.values
    last = None
    T, X, Y, TH, ST, HX = [], [], [], [], [], []
    for tt, kind, k in ev:
        f.predict(tt)
        if kind == 1:
            last = (ov[k], ow[k], 0.0 if gate[k] else 1.0)
            if not gate[k]:
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
    return dict(t=T, x=np.array(X), y=np.array(Y), th=np.array(TH), st=np.array(ST),
                hx=np.array(HX), gz=gz, m=metrics(T, np.array(X), np.array(Y), np.array(TH), gt))


def mask_to_odom(segs, odom_t):
    g = np.zeros(len(odom_t), bool)
    for r in segs:
        if not r.slip:
            continue
        idx = np.searchsorted(r.t, odom_t, side="right") - 1
        inside = (odom_t >= r.t[0]) & (odom_t <= r.t[-1])
        g |= inside & r.slip_mask[np.clip(idx, 0, len(r.t) - 1)]
    return g


def score(segs, gt, imu_t, ov_odom, odom_t):
    u_gt, _ = gt_body_rates(gt, imu_t)
    tp = fp = fn = tn = 0
    s_tp = s_fp = s_fn = 0
    for r in segs:
        k = (imu_t >= r.t0) & (imu_t <= r.t1)
        d_gt = float(np.sum(np.interp(r.t, imu_t, u_gt) * np.r_[np.diff(r.t), 0.02]))
        true = abs(r.d_wheel - d_gt) >= SLIP_SEG
        tp += r.slip and true; fp += r.slip and not true; fn += (not r.slip) and true
        tn += (not r.slip) and not true
        sm = np.abs(r.v_wheel - np.interp(r.t, imu_t, u_gt)) >= SLIP_V
        s_tp += int((r.slip_mask & sm).sum()); s_fp += int((r.slip_mask & ~sm).sum())
        s_fn += int((~r.slip_mask & sm).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, s_tp=s_tp, s_fp=s_fp, s_fn=s_fn)


def oracle_gate(odom, gt):
    u, _ = gt_body_rates(gt, odom.t.values)
    return np.abs(odom.linear_x.values - u) >= SLIP_V


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
    print("DEVELOPMENT EVALUATION (step 5c) — provisional parameters; mean over 3 accel-bias draws;"
          " not paper results.\n")
    tot = dict(tp=0, fp=0, fn=0, tn=0, s_tp=0, s_fp=0, s_fn=0)
    print("PART 1 — slip detection at stops and retroactive correction (pos RMSE / final, m)")
    for sid in a.sessions:
        sess = load_session(a.data, sid)
        scans = load_scans(a.data, sid)
        imu, odom, cmd, gt = sess
        ov, ow = odom.linear_x.values, odom.angular_z.values
        live, corr, orc, nseg, nslip = [], [], [], 0, 0
        for b in BIASES:
            p1 = pipeline(sess, scans, det_cfg, p9, b, ov, ow)
            segs = run_segments(sc, p1["t"], p1["hx"], p1["gz"], p1["st"], odom.t.values, ov)
            g = mask_to_odom(segs, odom.t.values)
            p2 = pipeline(sess, scans, det_cfg, p9, b, ov, ow, gate=g)
            po = pipeline(sess, scans, det_cfg, p9, b, ov, ow, gate=oracle_gate(odom, gt))
            live.append(p1["m"]); corr.append(p2["m"]); orc.append(po["m"])
            s = score(segs, gt, p1["t"], ov, odom.t.values)
            for k in tot:
                tot[k] += s[k]
            nseg += len(segs); nslip += sum(r.slip for r in segs)
        f = lambda L, k: np.mean([m[k] for m in L])
        print(f"  {SESSIONS.get(sid, sid):26s} segments {nseg // 3:2d} (flagged {nslip / 3:.1f})  "
              f"live {f(live, 'pos'):6.3f} / {f(live, 'pos_fin'):6.3f}   "
              f"corrected {f(corr, 'pos'):6.3f} / {f(corr, 'pos_fin'):6.3f}   "
              f"oracle {f(orc, 'pos'):6.3f} / {f(orc, 'pos_fin'):6.3f}")
    print(f"\n  segment slip flags (all sessions x 3 draws): TP {tot['tp']}  FP {tot['fp']}  "
          f"FN {tot['fn']}  TN {tot['tn']}")
    pr = tot["s_tp"] / max(tot["s_tp"] + tot["s_fp"], 1)
    rc = tot["s_tp"] / max(tot["s_tp"] + tot["s_fn"], 1)
    print(f"  per-sample slip labels inside segments: precision {pr:.2f}  recall {rc:.2f}"
          f"  (true slip = |v_wheel - u_gt| >= {SLIP_V} m/s)")

    print("\nPART 2 — wheel scale error (odometry v, w x s_true), estimated from the same session")
    for s_true in (1.02, 0.98):
        for sid in a.sessions:
            sess = load_session(a.data, sid)
            scans = load_scans(a.data, sid)
            imu, odom, cmd, gt = sess
            ov, ow = odom.linear_x.values * s_true, odom.angular_z.values * s_true
            b = BIASES[0]
            p1 = pipeline(sess, scans, det_cfg, p9, b, ov, ow)
            segs = run_segments(sc, p1["t"], p1["hx"], p1["gz"], p1["st"], odom.t.values, ov)
            s_est, s_sig, n = wheel_scale(sc, segs)
            if n == 0:
                print(f"    s_true {s_true:.2f}  {SESSIONS.get(sid, sid):26s} no usable segments")
                continue
            use = scale_usable(sc, s_est, s_sig, n)
            p3 = pipeline(sess, scans, det_cfg, p9, b, ov * s_est, ow * s_est) if use else p1
            print(f"    s_true {s_true:.2f}  {SESSIONS.get(sid, sid):26s} n {n:2d}  "
                  f"est wheel factor {1 / s_est:.4f} ± {s_sig / s_est ** 2:.4f}  "
                  f"{'APPLIED    ' if use else 'not applied'}  "
                  f"pos RMSE {p1['m']['pos']:6.3f} -> {p3['m']['pos']:6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
