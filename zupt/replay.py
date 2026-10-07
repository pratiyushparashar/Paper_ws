"""
Replay a recorded raw session through the detector and score it against ground truth.

    python3 -m zupt.replay --session 20261003_022102 [--data data/raw] [--params config/zupt_params_dev.yaml]

Ground truth labels (never used by the detector):
    truly still  <=>  |v_gt| < 0.005 m/s and |w_gt| < 0.005 rad/s  (GT backward-differenced)

Reports, per IMU sample:
    false stationary  = detector says stationary while GT says moving   (safety metric)
    recall            = fraction of GT-still samples (after cmd stop) the detector commits
and, per stop command: latency from GT standstill to first commit, and any false commits.
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

from zupt.core.config import load_config
from zupt.core.detector import Evidence, ImuSample, StationarityDetector

V_STILL = 0.005
W_STILL = 0.005


def _t(df):
    return df["time_sec"].astype(float) + df["time_nsec"].astype(float) * 1e-9


def quat_to_rpy(x, y, z, w):
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def decimate_imu(imu, target_hz=50.0):
    """IMU recorded faster than target (launch arg imu_rate) -> averages over fixed TIME bins of
    1/target_hz s. Averaging rates and specific forces = integrating them over the bin, which is
    what a real IMU's internal filtering does; point-sampling at 50 Hz loses most of a 55 ms
    start/stop ramp and all of a 1 ms contact impulse (step 5c finding). Bins are by time, not
    by message count, so messages the logger dropped only thin a bin instead of shifting every
    later bin. Orientation and the other columns: last sample of the bin; timestamp = bin end."""
    t = imu["time_sec"].astype(float).values + imu["time_nsec"].astype(float).values * 1e-9
    if len(t) < 3:
        return imu
    rate = 1.0 / float(np.median(np.diff(t)))
    if rate < 1.5 * target_hz:
        return imu
    b = np.floor((t - t[0]) * target_hz + 1e-9).astype(int)
    avg_cols = [c for c in ("angular_vel_x", "angular_vel_y", "angular_vel_z",
                            "linear_acc_x", "linear_acc_y", "linear_acc_z") if c in imu]
    out = imu.groupby(b).last().reset_index(drop=True)
    out[avg_cols] = imu[avg_cols].groupby(b).mean().reset_index(drop=True)
    te = t[0] + (np.unique(b) + 1) / target_hz
    out["time_sec"] = np.floor(te).astype(int)
    out["time_nsec"] = np.round((te - np.floor(te)) * 1e9).astype(int)
    if "timestamp" in out:
        out["timestamp"] = te
    full = len(imu) / len(out)
    if full < 0.9 * rate / target_hz:
        print(f"[decimate_imu] note: {100 * (1 - full * target_hz / rate):.0f}% of IMU messages "
              f"missing (logger dropped them); bins averaged over the messages present")
    return out


def load_session(data_dir, sid, imu_hz=50.0):
    p = lambda name: os.path.join(data_dir, f"{name}_{sid}.csv")
    imu = decimate_imu(pd.read_csv(p("imu")), imu_hz)
    odom = pd.read_csv(p("odom"))
    cmd = pd.read_csv(p("cmd_vel")) if os.path.exists(p("cmd_vel")) else None
    gt = pd.read_csv(p("ground_truth"))
    gt = gt[~gt["child_frame_id"].isin(["base_footprint", "base_link"])].reset_index(drop=True)
    for d in (imu, odom, gt) + ((cmd,) if cmd is not None else ()):
        d["t"] = _t(d)
    return imu, odom, cmd, gt


def load_scans(data_dir, sid):
    """Scan CSV or None (sessions recorded before the LiDAR was logged have none)."""
    p = os.path.join(data_dir, f"scan_{sid}.csv")
    if not os.path.exists(p):
        return None
    sc = pd.read_csv(p)
    sc["t"] = _t(sc)
    return sc


def lidar_evidence(scans, odom, lever_l=0.3, odom_v=None, odom_w=None, **check_kw):
    """Run ScanMotionCheck over a session. Wheel claims come from the odometry twist
    (odom_v/odom_w override it, e.g. to replay a wheel fault). Returns a list of LidarEvidence."""
    from zupt.core.lidar import ScanMotionCheck
    chk = ScanMotionCheck(**check_kw)
    ot = odom.t.values
    ov = odom.linear_x.values if odom_v is None else odom_v
    ow = odom.angular_z.values if odom_w is None else odom_w
    out, j = [], 0
    for k in range(len(scans)):
        ts = scans.t.iat[k]
        while j < len(ot) and ot[j] <= ts:
            if j > 0:
                chk.add_odom(ov[j - 1], ow[j - 1], ot[j] - ot[j - 1], lever_l)
            j += 1
        r = np.array(scans.ranges.iat[k].split(";"), float)
        out.append(chk.step(ts, r, scans.angle_min.iat[k], scans.angle_increment.iat[k],
                            scans.range_min.iat[k], scans.range_max.iat[k]))
    return out


def gt_truth(imu_t, gt):
    _, _, yaw = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    yaw = np.unwrap(yaw)
    # Backward differences: "did the robot move between the previous GT sample and this one?"
    # (central differences leak the next sample's motion backwards in time and mislabel the
    # last still sample before a start as moving.)
    dt = np.r_[np.inf, np.diff(gt.t.values)]
    v = np.hypot(np.r_[0, np.diff(gt.gt_x.values)], np.r_[0, np.diff(gt.gt_y.values)]) / dt
    w = np.r_[0, np.diff(yaw)] / dt
    v_i = np.interp(imu_t, gt.t.values, v)
    w_i = np.interp(imu_t, gt.t.values, w)
    return (v_i < V_STILL) & (np.abs(w_i) < W_STILL), v_i


def zoh(src_t, src_v, t):
    """Latest value at or before t (zero-order hold); None before the first sample."""
    idx = np.searchsorted(src_t, t, side="right") - 1
    return idx


def run(data_dir, sid, params, quiet=False):
    cfg = load_config(params)
    imu, odom, cmd, gt = load_session(data_dir, sid)
    roll, pitch, _ = quat_to_rpy(imu.orient_x, imu.orient_y, imu.orient_z, imu.orient_w)

    t = imu.t.values
    oi = zoh(odom.t.values, None, t)
    ci = zoh(cmd.t.values, None, t) if cmd is not None and len(cmd) else np.full(len(t), -1)

    det = StationarityDetector(cfg, have_lidar=False, logger=(lambda *_: None) if quiet else print)
    rows = []
    for k in range(len(t)):
        s = ImuSample(t[k], imu.angular_vel_z.iat[k], imu.linear_acc_x.iat[k],
                      imu.linear_acc_y.iat[k], imu.linear_acc_z.iat[k], roll.iat[k], pitch.iat[k])
        e = Evidence()
        if ci[k] >= 0:
            e.v_cmd = cmd.linear_x.iat[ci[k]]
            e.w_cmd = cmd.angular_z.iat[ci[k]]
        if oi[k] >= 0:
            e.v_odom = odom.linear_x.iat[oi[k]]
            e.w_odom = odom.angular_z.iat[oi[k]]
        o = det.step(s, e)
        rows.append((o.t, o.stationary, o.candidate, o.confidence, o.reason, o.acc_mean[0]))

    out = pd.DataFrame(rows, columns=["t", "stationary", "candidate", "conf", "reason", "acc_mean_x"])
    still, v_gt = gt_truth(t, gt)
    out["gt_still"] = still
    out["v_gt"] = v_gt
    out["cmd_zero"] = [ci[k] < 0 or (abs(cmd.linear_x.iat[ci[k]]) <= cfg.eps_cmd
                                     and abs(cmd.angular_z.iat[ci[k]]) <= cfg.eps_cmd)
                       for k in range(len(t))]
    return out, cmd, cfg


def report(out, cmd, sid):
    fs = out.stationary & ~out.gt_still
    n_st = int(out.stationary.sum())
    still_after = out.gt_still & out.cmd_zero
    print(f"\n=== session {sid}: {len(out)} IMU samples, {out.t.iloc[-1] - out.t.iloc[0]:.1f} s ===")
    print(f"committed stationary samples : {n_st}")
    print(f"FALSE stationary samples     : {int(fs.sum())}"
          + (f"  (max |v_gt| during them {out.v_gt[fs].max():.3f} m/s)" if fs.any() else ""))
    print(f"recall on GT-still (cmd 0)   : {out.stationary[still_after].mean():.3f}")
    print("blocking reasons while GT moving & cmd 0:",
          out[~out.gt_still & out.cmd_zero].reason.value_counts().to_dict())

    if cmd is None or not len(cmd):
        return
    zero = (cmd.linear_x.abs() < 1e-6) & (cmd.angular_z.abs() < 1e-6)
    stops = cmd.t[zero & ~zero.shift(1, fill_value=False)].values
    nxt = list(cmd.t[~zero].values)
    print("\nper stop:")
    for ts in stops:
        later = [x for x in nxt if x > ts]
        te = later[0] if later else out.t.iloc[-1]
        seg = out[(out.t >= ts) & (out.t < te)]
        gs = seg[seg.gt_still]
        t_still = gs.t.iloc[0] - ts if len(gs) else math.nan
        com = seg[seg.stationary]
        t_com = com.t.iloc[0] - ts if len(com) else math.nan
        bad = int((seg.stationary & ~seg.gt_still).sum())
        print(f"  stop {ts:8.2f}s  dur {te - ts:5.1f}s  GT still after {t_still:5.3f}s  "
              f"committed after {t_com:5.3f}s  false-commit samples {bad}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, nargs="+")
    ap.add_argument("--data", default=os.path.expanduser("~/paper_ws/data/raw"))
    ap.add_argument("--params", default=os.path.expanduser("~/paper_ws/config/zupt_params_dev.yaml"))
    a = ap.parse_args(argv)
    worst = 0
    for sid in a.session:
        out, cmd, _ = run(a.data, sid, a.params)
        report(out, cmd, sid)
        worst = max(worst, int((out.stationary & ~out.gt_still).sum()))
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
