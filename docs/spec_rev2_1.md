# Framework Specification — rev 2.1

Status: LOCKED design. Supersedes spec_rev2.md. Parameter VALUES marked (cal) come from calibration,
(pre) are fixed in docs/preregistration.md before evaluation. Changes to this file require a logged
reason in docs/decisions.md.

Date: 2026-10-03 · Workspace: ~/paper_ws · Robot: paper_robot_sim (copy of mobile_robot)

---

## 0. Decisions

| ID | Decision |
|---|---|
| D1 | Skeleton changes **2, 3, 4, 5, 6 adopted now**. Change **1 (LiDAR scan matching inside the EKF) deferred** until the rest of the project is complete. |
| D2 | Offline first: core is pure Python (no ROS) over recorded CSVs; a thin ROS node wraps it last. |
| D3 | One LSTM-adaptive EKF with state **[x, y, θ, v, ω, b_g]**. ZUPT (v = 0) and ZARU (ω = 0) are measurement updates in this same EKF. |
| D4 | The EKF does **not** use Gazebo IMU orientation (exact in sim, unrealistic). Heading comes from integrating ω. |
| D5 | A **gyro-bias IMU variant** is added (bias_mean (pre), proposed 0.01 rad/s). |
| D6 | **Safety principle:** a false "stationary" is worse than a missed one. When uncertain, the detector says "not stationary". |
| D7 | Prediction uses a constant-velocity model. `cmd_vel` is not a control input; it is used only by detector Tier 1. |
| D8 | Comparison baselines (Fixed / Heuristic / old LSTM 3-state EKFs) receive the **same realistic heading** (integrated raw gyro), not the exact Gazebo orientation, so comparisons are fair. |

---

## 1. Architecture

```
Sensors (IMU rate+accel, wheel odom, LiDAR scan, cmd_vel)
  → Preprocessing (time sync, outlier handling)
  → Temporal window X(t−T:t): sensor features, sensor-derived terrain cues, EKF innovation history
  → LSTM → r_IMU, r_odom, r_LiDAR, P_slip
       ├─ R(t): R_i = R_i,0 / (r_i + ε)
       ├─ Q(t): from P_slip
       └─ Stationarity detector (raw sensors + r_i + P_slip + LiDAR scan change)
  → EKF [x, y, θ, v, ω, b_g]: odom, gyro updates; ZUPT/ZARU when detector commits
  → (until change 1) existing post-EKF LiDAR stage with r_LiDAR
  → Final pose
```

Rules:
- The detector never reads the EKF's state or outputs (change 3).
- No prior-reliability feedback r(t−1) into the window (change 4). `cov_trace` excluded for the same reason.
- No ground-truth terrain label anywhere in the inputs (change 6).

---

## 2. EKF (zupt/ekf)

**State** X = [x, y, θ, v, ω, b_g]

**Predict** (dt from timestamps)
- x += v cosθ dt; y += v sinθ dt; θ += ω dt
- v, ω, b_g: random walk

**Process noise (change 5)**
Q(t) = diag(q_x, q_y, q_θ, q_v·(1 + k_v·P_slip), q_ω·(1 + k_ω·P_slip), q_b)·dt
q_*, k_v, k_ω (cal) — tuned on calibration runs only.

**Measurements**
| Name | z | h(X) | Rate | R |
|---|---|---|---|---|
| Odometry | [v_odom, ω_odom] | [v, ω] | 100 Hz | R_odom,0 / (r_odom + ε) |
| Gyro | gyro_z | ω + b_g | 50 Hz | R_g,0 / (r_imu + ε), R_g,0 = σ_g² from SDF |
| ZUPT | 0 | v | when committed | R(c)_v |
| ZARU | 0 | ω | when committed | R(c)_ω |

ε = 1e-3. Accelerometer is not a measurement (used only by the detector and as LSTM features).

**Interface for change 1:** measurement models are a list of (z, h, H, R) objects so a LiDAR pose
measurement can be added later without restructuring the filter.

**Ablation variants:** (i) fixed R, Q, no ZUPT · (ii) LSTM R only · (iii) LSTM R + P_slip Q ·
(iv) (iii) + ZUPT · (v) (iv) + ZARU · (vi) (v) with fixed R_ZUPT instead of R(c) ·
(vii) (v) with detector without LSTM evidence.

---

## 3. Stationarity detector (zupt/core)

Outputs per step: is_stationary, confidence c ∈ [0, 1], R_scale, tier, diagnostics.

| Tier | Rule |
|---|---|
| 1 Command | \|v_cmd\| or \|ω_cmd\| > ε_cmd → NOT stationary; record t_stop |
| 2 Settle | t − t_stop < settle_delay (cal) → NOT stationary |
| 3 LiDAR | scan-change Δ = median\|r_t − r_{t−k}\| over valid beams; Δ > lidar_move_thresh (cal) → NOT stationary. Can only block. |
| 4a Sensor health | **Frozen-sensor check:** IMU (or odometry) values unchanged for ≥ n_frozen samples → sensor failed → NOT stationary |
| 4b Learned evidence | r_imu < r_imu_min → IMU evidence invalid → NOT stationary · P_slip ≥ p_slip_max → NOT stationary |
| 4c Odometry gate | \|v_odom\| ≥ odom_v_eps or \|ω_odom\| ≥ odom_w_eps → NOT stationary (blocks only; never confirms). r_odom low → gate skipped, not trusted to confirm |
| 4d IMU variance | per-axis variance (gyro_z, accel_x, accel_y; never summed), raw native-rate IMU, window W, floor v_floor,i = (0.5σ_i)²; any v_i ≥ veto_i → NOT stationary; c_i = log-ramp between flat_i and veto_i; c = min c_i |

- Candidate: all tiers pass and c ≥ c_min (pre). Commit after N consecutive candidates; drop on first failure.
- R(c) = R₀ · (R_max/R₀)^((1−c)/(1−c_min)). Rate limit zupt_max_rate_hz.
- Stall warning per stop (thresholds never changed at runtime). Startup warning if no LiDAR.
- Placeholder values in the parameter file → detector refuses to commit and logs an error.

---

## 4. LSTM (zupt/learning)

- Method unchanged from Phase 2: LOBO folds, clean-baseline weight 5×, train-only normalisation,
  trivial-predictor comparison, single test evaluation.
- **Features:** new-EKF innovations (odom v, odom ω, gyro); raw gyro_z, accel_x/y/z; odom v, ω;
  wheel–IMU disagreement (ω_odom − gyro_z); wheel–LiDAR disagreement (existing estimator);
  **terrain cues:** windowed variance of accel_z and high-passed accel_x (vibration).
- **Excluded:** r(t−1), cov_trace, any terrain ID.
- **Outputs:** r_imu, r_odom, r_lidar, P_slip (sigmoid).
- **Labels:** from fault manifests (as before); P_slip from **physical** slip windows in the terrain world
  (|v_odom − v_gt| > slip_thresh (pre)) plus injected slip events.
- Fault injection ported to the new signals: IMU faults target **gyro rate** (not yaw); odometry faults
  target v/ω.

---

## 5. LiDAR until change 1

- Existing post-EKF ray-cast stage kept unchanged; r_LiDAR scales its weight.
- Scan-change Δ (detector Tier 3) is independent of scan matching and used from the start.
- **When change 1 lands:** add LiDAR pose measurement to the EKF → regenerate features → retrain LSTM →
  rerun evaluation. One extra retrain/rerun cycle is budgeted for this.

---

## 6. Simulation prerequisites

1. Logger writing raw native-rate IMU, odom, cmd_vel, GT, scan to ~/paper_ws/data.
2. Gyro-bias IMU variant (D5).
3. Terrain/slip world: low-μ floor patches (e.g. μ = 0.1) and a gentle ramp (few degrees); near-flat only.
4. **Slip verification test (gate):** drive across a low-μ patch; wheel-odom distance must exceed GT distance.
   Optionally test Gazebo WheelSlip plugin for gradual slip.

---

## 7. Calibration and evaluation

- Detector thresholds from post-stop windows labelled by GT velocity (STILL vs still-moving);
  analyzer refuses on overlap. Calibration and evaluation runs disjoint.
- Detector metrics: false-stationary rate (primary), longest false run, recall, latency (median, p95).
- Estimator metrics: ATE, RPE, heading drift, |b̂_g − b_true|, recovery time.
- Key experiments: correlated false-stationary (slip on ramp, frozen IMU), graceful degradation,
  failure → recovery, terrain–fault interaction.
- Statistics: mean ± std across sessions/seeds; paired tests.

---

## 8. Known limitations (stated in the paper)
1. Until change 1, no absolute position fix inside the EKF; position drift bounded only by the post-EKF LiDAR stage.
2. Variance-based stillness is blind to constant-rate motion; LiDAR Tier 3 covers it only within 2.5 m of walls.
3. Gazebo noise is idealised; rigid-body physics cannot model deformable terrain (sand, mud).
4. Data from this setup is not pooled with Jazzy/Harmonic-era baselines.

---

## 9. Build order (one step at a time; each verified and committed)
1. Logger → ~/paper_ws/data
2. Baseline recording; measure stationary noise
3. Sim variants: gyro-bias IMU, terrain/slip world; **slip verification gate**
4. Detector core + tests (nonzero noise, summing-bug regression, floor, frozen sensor, hysteresis, veto)
5. 6-state EKF (fixed R/Q) + ZUPT/ZARU + tests; bias recovery on the gyro-bias variant
6. Port fault injection to new signals
7. Labels, features, LSTM retraining
8. Integration: LSTM → R, P_slip → Q, LSTM evidence → detector
9. preregistration.md → evaluation + ablation
10. Change 1 (scan matching) → retrain → rerun
11. ROS node wrapper

---

## Amendment A1 (2026-10-05) — Tier 4e, acceleration mean check
Evidence: docs/decisions.md 2026-10-05 (slide with flat variance, mean -0.53 m/s^2).
- Gravity-compensate accel using IMU roll/pitch only: a_h = R(roll,pitch)·a − g.
- Over window W: |mean(a_h,x)| ≥ acc_mean_thresh or |mean(a_h,y)| ≥ acc_mean_thresh → NOT stationary (blocks only).
- acc_mean_thresh (cal); must exceed accel bias + noise of the mean (~1e-4 + σ/√W).
- Tests must include: constant-deceleration slide with flat variance → NOT stationary.
