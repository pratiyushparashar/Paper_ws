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
