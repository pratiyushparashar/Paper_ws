# Change 1 — LiDAR scan matching inside the EKF (2026-10-07)

DEVELOPMENT results: provisional parameters partly set on these sessions; one accel-bias draw
(0.03, −0.02 m/s²). Not paper results.

## Built
- `zupt/core/scan_match.py` — 2D point-to-line ICP (Gauss-Newton, Huber, shrinking gate,
  field-of-view handling), normals from ±3-beam line fits, covariance σ_r²(JᵀWJ)⁻¹·inflation,
  degeneracy handling by solution remapping: directions with eigenvalue < 2 % of the largest keep
  the EKF prediction and get variance 100. Quality gates only (points, RMS); NO innovation gate,
  so a correct scan is not rejected because the filter believes wrong wheels.
- `zupt/ekf/ekf_lidar.py` — EKF9 + stochastic cloning: pose cloned at each scan, scan k matched to
  scan k−1 (initialised with the EKF-predicted relative pose), relative-pose update.
- `zupt/ekf/ekf9.py` — state size generalised (needed for the clone).
- `zupt/evaluation/eval_change1.py`; tests `test_scan_match.py` (5), `test_ekf_lidar.py` (3).

## Scan matcher alone (scan-to-scan vs ground truth)
| world | degenerate scans | error along / across / heading (rms per scan) |
|---|---|---|
| terrain lanes (C, D, S2) | 0–3 of ~550–760 | 3 mm / 0.1 mm / 0.005° (ribs make the lanes observable) |
| indoor world (S1, S3) | 85–100 % | along-corridor unobservable → correctly reported, not used |
| Test E (open area, wall ahead) | 450 / 455 | lateral unobservable, forward observable |
Unit-test corridor without degeneracy handling: matcher claimed ±1 cm along the axis while 5 cm
wrong (noisy normals) — the handling is necessary, not cosmetic. Covariance inflation 25 set from
along-lane z-scores (1.5–1.7 at 10).

## In the EKF (pos RMSE / final error, m)
| session | 5b | 5b+5c | LiDAR | LiDAR+5c | LiDAR, no wheels | oracle+LiDAR |
|---|---|---|---|---|---|---|
| Test C (50 Hz) | 1.214/2.97 | 0.069/0.16 | 0.962/2.35 | **0.023/0.01** | 0.121/0.23 | 0.026/0.03 |
| Test D strips | 0.068/0.14 | 0.068/0.14 | 0.060/0.11 | 0.060/0.11 | 0.102/0.15 | 0.063/0.12 |
| Test E stuck | 0.139/0.19 | 0.139/0.19 | **0.021/0.03** | 0.021/0.03 | 0.010/0.01 | 0.002/0.00 |
| S1 slip patch (1 kHz) | 0.137/0.22 | 0.090/0.20 | 0.137/0.22 | 0.089/0.20 | **0.027/0.05** | 0.015/0.02 |
| S2 Test C (1 kHz) | 1.550/3.21 | 0.082/0.17 | 1.273/2.64 | **0.057/0.12** | 0.099/0.17 | 0.029/0.06 |
| S3 flat (1 kHz) | 0.055/0.09 | 0.055/0.09 | **0.027/0.04** | 0.027/0.04 | 0.104/0.14 | 0.052/0.12 |

## Findings
1. Nominal driving: LiDAR halves the error where the geometry is observable (flat S3 −51 %,
   stuck Test E −85 %, strips −12 %); where it is not (indoor corridor, S1) it changes nothing and
   does no harm (degeneracy handling).
2. Against wrongly-TRUSTED wheels (ice slide) LiDAR alone recovers only ~20 % (2.97 → 2.35 m):
   100 Hz confident wheel updates outvote 10 Hz scans. LiDAR + 5c is the best combination
   (Test C final 0.01 m at 50 Hz, 0.12 m at 1 kHz) => 5c stays: it is complementary to LiDAR.
3. Fixed-weight trade-off (Test C final / S3 RMSE / D RMSE): inflation 1: 0.31 / 0.046 / 0.082;
   5: 1.24 / 0.038 / 0.080; 25: 2.35 / 0.027 / 0.060. No single LiDAR weight serves both slip and
   nominal driving — the weight must follow the WHEELS' reliability. This is the paper's
   motivation for learned reliability (steps 8–9).
4. LiDAR NIS is a strong slip cue: 95th percentile 97–113 on the ice-slide sessions vs ≤ 10
   nominal (median 0.1–0.2 everywhere). Candidate LSTM input / consistency trigger (step 8).
5. With wheels removed, IMU (1 kHz) + LiDAR + ZUPT/NHC is competitive (0.01–0.23 m final) and on
   the slip patch better than trusting the slipping wheels (0.05 vs 0.22 m).
6. Indoor world is LiDAR-degenerate along its corridor: evaluation of change 1 needs the terrain
   world or added features; record new indoor sessions with this in mind.
