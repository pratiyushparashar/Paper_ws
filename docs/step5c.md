# Step 5c — ZUPT-bounded segment check (2026-10-05)

DEVELOPMENT results: provisional parameters, set partly on the same sessions. Not paper results.

## Idea
Between two confirmed zero-velocity intervals (stops, or stuck commits) the velocity is known to
be 0 at BOTH ends. That pins the accelerometer bias, so the integrated accelerometer becomes a
distance sensor for that segment. At the closing stop the IMU distance is compared with the wheel
distance; a significant disagreement = wheel slip. Consequences, all available at the stop:
1. retroactive correction (re-filter the segment with the slipping wheel samples skipped),
2. self-supervised slip labels (per-sample mask) for the LSTM — no ground truth needed,
3. wheel-scale calibration from non-slip segments.

## Built
- `zupt/segments.py` — segment split, bias at both ends (falls back to the other end if a stop is
  too short), centripetal lever-arm correction, misclosure removal, σ model, z-test, per-sample
  slip mask, weighted wheel-scale estimate. `config/segment_params_dev.yaml`.
- `zupt/evaluation/eval_5c.py` — live vs retroactively corrected vs oracle; flag/label scores;
  injected wheel scale error ±2 %.
- Sim: launch arg `imu_rate` (default 50 = unchanged), IMU noise scaled by sqrt(rate/50);
  `zupt.replay.load_session` block-averages faster IMU recordings to 50 Hz.
- `tools/check_imu_integration.py` — pass/fail check that a recording's IMU captures speed
  changes (|∫a dt − Δv_gt| ≤ 0.02 m/s) and impacts.

## Results (mean of 3 accel-bias draws; pos RMSE / final error, m)
| session | live | corrected at stops | oracle wheel gating |
|---|---|---|---|
| flat floor | 0.165 / 0.184 | 0.165 / 0.184 | 0.147 / 0.157 |
| slip patch | 0.126 / 0.143 | 0.126 / 0.143 | 0.008 / 0.010 |
| Test C ice slide | 1.214 / 2.973 | **0.069 / 0.155** | 0.078 / 0.171 |
| Test D strips | 0.068 / 0.138 | 0.068 / 0.138 | 0.070 / 0.145 |
| Test E stuck | 0.139 / 0.189 | 0.139 / 0.189 | 0.063 / 0.078 |

- Segment slip flags: 3 TP, 0 FP, 12 FN, 54 TN (all sessions × 3 draws). Per-sample labels:
  precision 0.99, recall 0.31. => Large slips (Test C, 2.97 m) are caught and fixed at the next
  stop; slips of 0.2–0.35 m (slip patch, flat) are missed. No false alarms.
- Wheel scale (±2 % injected): estimate uncertainty ±2–3 % per session (flat, 8 segments), up to
  ±47 % on the strips — cannot resolve a 2 % error with the current recordings. Applying such an
  estimate made things WORSE (Test D 0.159 -> 0.791 m, Test C 1.284 -> 1.380 m), so the correction
  is now applied only when σ ≤ 0.5 % with ≥ 5 segments (`scale_usable`); with the current data it is
  never applied and nothing gets worse.

## Root cause of the limits: Gazebo IMU point-sampling (sim artifact)
Gazebo reports the INSTANTANEOUS acceleration at the 50 Hz output instants. A start/stop is a
~55 ms ramp at ~5.4 m/s², so 2 or 3 samples land on it: the integrated Δv is 0.214 or 0.319
instead of 0.295 m/s (errors up to 0.1 m/s per event, `check_imu_integration.py` FAIL on all
existing sessions). A wall impact (Δv in ~1 ms) is missed completely (Test E). Real IMUs filter
and average internally, so they preserve Δv. This dominates segment error (median 0.08 m, worst
0.6 m on flat) and therefore the slip detection threshold and the scale estimate.

Fix (opt-in): record with `imu_rate:=1000` (= physics step) and average to 50 Hz in the loader.
Must be verified with one recording (check tool PASS, logger not dropping messages) before it
is adopted for the step 7 calibration/evaluation data.

## 1000 Hz verification (session 20261005_045207) — ADOPTED
`check_imu_integration.py`: start 0 -> 0.5 m/s: ∫a dt 0.500 vs GT 0.500 (error 0.000); wall impact
GT −0.500 vs IMU −0.501 (captured; raw spike −501 m/s² in 1 ms). PASS. Raw rate 1000 Hz, 7 gaps in
29 729 messages (logger start-up). Launch default changed to `imu_rate:=1000` for all recordings
from now on; the earlier 50 Hz sessions remain development data only.

## 5c re-test at 1000 Hz (2026-10-07; sessions 20261007_201210 slip patch, _201651 Test C, _202005 flat stop-and-go)
Recorded on the slow PC: the logger dropped 24–26 % of IMU messages in the two long sessions
(gaps ≤ 38 ms); the loader now averages over fixed time bins, so this only thins bins.
IMU check (gravity-compensated; earlier tool version wrongly used raw accel_x, which contains
g·sin(pitch) on the ramp): slip patch PASS 0.011 m/s, flat PASS 0.015 m/s, Test C 0.028 m/s on the
ramp/ice transitions and 0.058 m/s at the slope-to-floor knee (tilt-transition limitation, 4 %);
through the 1.4 m/s ice slide itself the integrated IMU velocity matches GT to 0.001 m/s.

| | 50 Hz (old) | 1000 Hz |
|---|---|---|
| segment slip flags | TP 3 FP 0 FN 12 | TP 15 FP 0 FN 12 |
| per-sample labels | precision 0.99, recall 0.31 | precision 1.00, recall 0.54 |
| Test C final error live → corrected | 2.97 → 0.155 m | 3.22 → 0.173 m |
| slip patch RMSE live → corrected (oracle) | 0.126 → 0.126 (0.008) | 0.137 → 0.090 (0.017) |
| flat stop-and-go (41 segments, no slip) | — | 0 flags; unchanged 0.055 |
| wheel scale σ (flat) | ±2–3 % | ±0.5 % |

Wheel scale with 12 straight segments: injected 1.02 -> estimated 1.0148 ± 0.0051 (applied,
RMSE 0.089 -> 0.061 m); injected 0.98 -> 0.9750 ± 0.0049 (correct within 1σ, just outside the
0.5 % gate, not applied). On the slip patch the estimate was 1.0326 ± 0.0084 for a true 0.98 —
unflagged slip segments contaminate it (correctly not applied, but a robust estimator with outlier
rejection is needed before the gate is relaxed). The σ model (sigma0, k_misclosure) is still the
50 Hz one; re-tuning belongs to step 7 calibration data, not these sessions.
