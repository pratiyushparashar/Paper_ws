# Decisions log — paper_ws

## 2026-09-28 — Simulation baseline
- Platform: ROS 2 Lyrical, Gazebo Sim 10.5.0. ws_mobile (Kushagra) was built on Jazzy/Harmonic
  and is left untouched; paper_robot_sim is a renamed copy of its mobile_robot package.
- Launch fix: gz_args "-r -v -v4" -> "-r -v4" (Gazebo 10 rejects two verbosity flags).
- Ground truth fix: bridge gz_topic_name "/model/differential_drive_robot/pose" ->
  "/differential_drive_robot/pose" (verified: gz.msgs.Pose published there, nothing on old name);
  "lasy" -> "lazy".
- Odometry: tag "odom_publisher_frequency" was ignored (measured ~48.7 Hz, the plugin default);
  renamed to "odom_publish_frequency", kept at 100 Hz to match the replicated paper's Table 1
  (measured ~97 Hz at RTF ~0.98).
- Verified rates: imu ~48 Hz, odom ~97 Hz, scan ~9.6 Hz (162/540 finite, min 1.90 m = corridor
  inner face), ground_truth_pose ~48.5 Hz.
- Consequence: data from this setup must not be pooled with the Jazzy-era baselines. Final paper
  results should be regenerated on this setup (to discuss with Kushagra).

## ZUPT framework: rev 2 (spec: docs/spec_rev2.md, to be written)
- Relaxation removed; replaced by a stall diagnostic warning.
- Graded confidence (log-ramp, min over axes) scales R; the binary gate uses c >= c_min.
- Variance-only blind spot (constant-rate motion) accepted; mitigated by LiDAR in Tier 3.

## 2026-10-03 — First baseline session (20261003_022102, 173 s, 13 stops)
- Measured stationary noise matches SDF: gyro_z 8.82e-3 rad/s, accel_x 2.01e-4, accel_y 1.99e-4 m/s².
- All stops settle to GT-still within 0.08–0.24 s (speeds up to 0.64 m/s) -> settle_delay ~0.3 s.
- 0.5 s window variance, STILL p99 vs post-stop-moving p10: accel_x 6.9e-8 vs 1.5e-2; accel_y 6.2e-8 vs 6.8e-7;
  gyro_z 1.3e-4 vs 6.5e-5 (OVERLAP) -> gyro alone cannot detect coasting; min-over-axes design confirmed.
- Wheel odometry drifts on flat floor: max 2.6 m / 36 deg heading over 30 m path, accumulated while turning (skid).
- Gazebo IMU orientation = GT to 1e-4 deg (confirms D4: do not use it). Integrated gyro drift 0.68 deg / 173 s
  (bias ~0) -> gyro-bias variant needed.
- Logger metadata duration/avg rates wrong (clock starts before /clock); CSVs correct. To fix later.

## 2026-10-05 — Step 3 tests (slip world 20261004_235739, gyro bias 20261005_000755)
- SLIP GATE PASSED. Stops on mu=0.1 patch: wheels zero in 0.02 s, body slides 22-23 cm over ~0.95 s
  (decel 0.53 m/s^2 ~ mu*g*wheel load share; caster frictionless). Restart on patch: odom 0.60 m vs GT 0.38 m
  (+59% wheel spin). Off-patch stop: 1.7 cm in 0.12 s.
- GYRO BIAS PASSED: 0.00988 +/- 0.00015 rad/s (set 0.01), stable; integrated heading drift 41 deg / 73 s;
  Gazebo IMU orientation unaffected by bias (confirms D4).
- DESIGN FINDING: during the slide, accel variance (3.1e-8) equals still variance, but accel mean = -0.53 m/s^2.
  Variance-only Tier 4 + settle 0.3 s would commit a FALSE stationary. Added Tier 4e: gravity-compensated planar
  acceleration mean check (roll/pitch from IMU; yaw not used). LiDAR Tier 3 likely blind to along-corridor slides
  (degeneracy) -- to verify with scan data.

## 2026-10-05 — Step 3 tests (slip world 20261004_235739, gyro bias 20261005_000755)
- SLIP GATE PASSED. Stops on mu=0.1 patch: wheels zero in 0.02 s, body slides 22-23 cm over ~0.95 s
  (decel 0.53 m/s^2 ~ mu*g*wheel load share; caster frictionless). Restart on patch: odom 0.60 m vs GT 0.38 m
  (+59% wheel spin). Off-patch stop: 1.7 cm in 0.12 s.
- GYRO BIAS PASSED: 0.00988 +/- 0.00015 rad/s (set 0.01), stable; integrated heading drift 41 deg / 73 s;
  Gazebo IMU orientation unaffected by bias (confirms D4).
- DESIGN FINDING: during the slide, accel variance (3.1e-8) equals still variance, but accel mean = -0.53 m/s^2.
  Variance-only Tier 4 + settle 0.3 s would commit a FALSE stationary. Added Tier 4e: gravity-compensated planar
  acceleration mean check (roll/pitch from IMU; yaw not used). LiDAR Tier 3 likely blind to along-corridor slides
  (degeneracy) -- to verify with scan data.
