# Step 5b — IMU-driven EKF, NHC, LiDAR Tier 3, stuck detection (2026-10-05)

All numbers are DEVELOPMENT results: provisional parameters, thresholds partly set on the same
sessions (Tests C/D), one run per test. Not paper results.

## Why
Tests C/D showed ZUPT gave zero benefit in the 6-state filter: velocity came from the wheels,
and at a real stop the wheels already read 0. ZUPT only matters when something integrates and
drifts. Decision: make the filter IMU-driven and add the constraints that bound it.

## What was built
- `zupt/ekf/ekf9.py` — state [x, y, θ, u, s, ω, b_g, b_ax, b_ay]; predict integrates the
  gravity-compensated accelerometer (rotating-frame terms included); updates: odometry, gyro,
  ZUPT (u = s = 0), ZARU, NHC (h = s − l·ω, l = 0.3 m IMU-to-axle lever arm).
  NHC is applied every IMU sample, loosened by P_slip via R/(1−P_slip+ε) and switched off at
  P_slip ≥ 0.5 (an inflated R alone still clamps a real skid after a few dozen 50 Hz updates).
- `zupt/core/lidar.py` — Tier 3 (fraction of beams changed) + stuck hypothesis test
  (H0 "not moved" vs H1 "moved as wheels claim", point-to-line outlier fractions, points outside
  the reference field of view treated as unknown, 1.5 s look-back reference).
- Detector stuck path: under a nonzero command, stationary only if LiDAR shows the scan unchanged
  AND contradicting the wheel claim, then the IMU checks (4a, r_imu, 4d, 4e) must also pass.
  Odometry gate and P_slip are not used on this path.
- Accelerometer bias: injected OFFLINE (additive, identical to recording with it); allows several
  bias draws per session. Sim left unchanged (its accel bias is ~1e-4 m/s²).

## Findings
1. Velocity-from-wheels baseline equivalent is "trusted wheels + NHC" (the unicycle model imposes
   NHC implicitly). Without NHC the lateral velocity is unobservable and drifts (8–256 m).
2. ZUPT/ZARU now matter, in proportion to how long the wheels are not trusted
   (pos RMSE m, mean of 3 accel-bias draws ±1–3 mg):

   | case | NHC only | NHC + ZUPT/ZARU |
   |---|---|---|
   | flat floor, wheels trusted | 0.955 (hdg 14.9°) | 0.165 (hdg 0.74°) |
   | flat, wheels correctly weighted (oracle) | 0.143 | 0.141 |
   | flat, wheel noise fault, oracle → wheels gated | 97.99 | 0.354 |
   | flat, wheels frozen 20 s, oracle | 0.552 | 0.325 |
   | wheels off entirely: flat / slip / Test C / Test D | 98 / 99 / 98 / 694 | 0.35 / 0.11 / 0.19 / 1.67 |

3. Test C ice slide (2.92 m, wheels locked): with oracle reliability gating the wheels during the
   slide, final error 2.97 m → 0.08 m. Recovered by accelerometer integration, not by ZUPT.
4. Accelerometer bias is estimated to within ~5 mm/s² on every session (0.02 on Test D y).
5. R = R0/(r+ε) with ε = 1e-3 does not remove a sensor that updates at 100 Hz:
   Test C oracle-scale 0.237 m final vs oracle-gate 0.082 m. → step 9 must GATE (skip) the update
   below a reliability threshold, not only scale R.
6. Rumble strips: IMU-only dead reckoning degrades sharply (1.67 m vs 0.19–0.35 m elsewhere) —
   rough terrain makes the IMU less reliable, the wheels more: terrain-conditioned reliability.
7. Stuck (synthetic: real still IMU + LiDAR, wheels/command overwritten with 0.3 m/s):
   detected after 0.76–0.84 s, error growth per episode +0.19–0.21 m instead of +1.0–7.0 m;
   0 false stuck and 0 false stationary on normal driving (Tests C/D).
8. In-lane LiDAR geometry is near-degenerate (two parallel walls); only the ribs reveal motion
   along a lane. Median statistics are blind there; fractions + point-to-line distances are not.
9. Tier 4e threshold (0.05 m/s²) must exceed residual accel bias + tilt error. Bias up to 0.03
   was used here; larger residual bias needs a start-up calibration (robot still at boot), which
   is not EKF feedback (change 3 still holds).

## Open / next
- Test E (real stuck: push into a wall) — checks IMU vibration from spinning wheels.
- Detection latency leaves ~0.2 m per stuck episode; could be removed by retroactive correction.
- Realistic accel noise (sim σ 0.0002 is very low) requires recalibrating the detector (step 7).
- Wheel odometry on slopes measures along-slope speed (u/cos φ); 0.4 % at 5°, ignored.
