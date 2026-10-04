#!/usr/bin/env python3
"""
Generate src/paper_robot_sim/model/terrain_world.sdf (build step 5, Phase 4 terrain).

    python3 tools/gen_terrain_world.py            # writes the SDF next to the other worlds

Six parallel lanes along +x, each 3.6 m wide between walls (LiDAR range 2.5 m, so a side wall
is always in view). Walls carry small ribs every 1.5 m so motion ALONG a lane changes the scan
(avoids corridor degeneracy for LiDAR Tier 3 / later scan matching).

Lane layout (x): 1..4 plain floor, 4..12 terrain, 12..15 plain floor. Open cross-corridors at
x < 1 and x > 15 connect the lanes. Lane i centre: y = 3.8 * i. Spawn lane i:
    ros2 launch paper_robot_sim gazebo_model.launch.py world:=terrain_world.sdf y:=<3.8*i>

Only mechanisms verified in this simulator are used: contact friction (mu) and rigid geometry.
Wheels have mu = 1.0, the caster ~0, so effective traction is ~mu * g * (wheel load share).
"""
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "src", "paper_robot_sim", "model", "terrain_world.sdf")

LANE_W = 3.6            # inner width
WALL_T = 0.2
PITCH = LANE_W + WALL_T  # 3.8 m lane pitch
N_LANES = 6
X_LANE0, X_LANE1 = 1.0, 15.0     # lane walls span
X_T0, X_T1 = 4.0, 12.0           # terrain section
WALL_H = 1.0
RIB = 0.1                        # rib size (protrudes into lane)
RIB_STEP = 1.5
PATCH_T = 0.002                  # friction patch thickness (top at z = 0.002)

COL = {"ice": "0.35 0.65 0.95", "severe": "0.20 0.45 0.85", "mild": "0.55 0.80 0.95",
       "rough": "0.55 0.45 0.35", "ramp": "0.60 0.60 0.60", "wall": "0.55 0.27 0.08",
       "rib": "0.30 0.15 0.05"}


def lane_y(i):
    return PITCH * i


def friction_xml(mu):
    return (f"<surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>"
            if mu is not None else "")


def box(name, x, y, z, sx, sy, sz, color, mu=None, roll=0.0, pitch=0.0, yaw=0.0):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.4f} {y:.4f} {z:.4f} {roll:.5f} {pitch:.5f} {yaw:.5f}</pose>
      <link name="link">
        <collision name="col">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          {friction_xml(mu)}
        </collision>
        <visual name="vis">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <material><ambient>{color} 1</ambient><diffuse>{color} 1</diffuse></material>
        </visual>
      </link>
    </model>"""


def patch(name, i, x0, x1, mu, color):
    return box(name, (x0 + x1) / 2, lane_y(i), PATCH_T / 2, x1 - x0, LANE_W, PATCH_T, color, mu)


def rumble_strips(name, i, x0, x1, seed):
    """One static model, many thin collision strips across the lane (1 cm +/- 0.5 cm)."""
    rng = random.Random(seed)
    cols = []
    x, k = x0, 0
    while x < x1:
        h = 0.005 + 0.01 * rng.random()
        cols.append((k, x, h))
        x += 0.25
        k += 1
    links = "".join(f"""
        <collision name="c{k}">
          <pose>{x - (x0 + x1) / 2:.4f} 0 {h / 2:.4f} 0 0 0</pose>
          <geometry><box><size>0.06 {LANE_W:.3f} {h:.4f}</size></box></geometry>
        </collision>
        <visual name="v{k}">
          <pose>{x - (x0 + x1) / 2:.4f} 0 {h / 2:.4f} 0 0 0</pose>
          <geometry><box><size>0.06 {LANE_W:.3f} {h:.4f}</size></box></geometry>
          <material><ambient>{COL['rough']} 1</ambient><diffuse>{COL['rough']} 1</diffuse></material>
        </visual>""" for k, x, h in cols)
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{(x0 + x1) / 2:.4f} {lane_y(i):.4f} 0 0 0 0</pose>
      <link name="link">{links}
      </link>
    </model>"""


def ramp(name, i, x0, run, rise_sign, mu, color, t=0.1):
    """Tilted slab whose TOP surface goes from z=z_start at x0 to z_start+rise over `run` metres.
    rise_sign=+1: up along +x starting at z=0; -1: down along +x ending at z=0."""
    theta = math.atan2(RAMP_RISE, run)
    length = math.hypot(run, RAMP_RISE)
    beta = -theta * rise_sign                 # SDF pitch: negative pitch raises +x end
    xc, zc = x0 + run / 2, RAMP_RISE / 2      # centre of top surface
    nx, nz = math.sin(beta), math.cos(beta)   # top-surface normal (rotated z axis)
    return box(name, xc - nx * t / 2, lane_y(i), zc - nz * t / 2, length, LANE_W, t, color, mu,
               pitch=beta)


RAMP_DEG = 5.0
RAMP_RUN = 3.0
RAMP_RISE = RAMP_RUN * math.tan(math.radians(RAMP_DEG))


def plateau(name, i, x0, x1, mu, color):
    h = RAMP_RISE
    return box(name, (x0 + x1) / 2, lane_y(i), h / 2, x1 - x0, LANE_W, h, color, mu)


def boundary_walls():
    """N_LANES + 1 walls; neighbouring lanes share the wall between them."""
    return [box(f"lane_wall_{k}", (X_LANE0 + X_LANE1) / 2, lane_y(0) - PITCH / 2 + PITCH * k,
                WALL_H / 2, X_LANE1 - X_LANE0, WALL_T, WALL_H, COL["wall"])
            for k in range(N_LANES + 1)]


def lane_walls(i):
    """Ribs on both side walls of lane i."""
    out = []
    for side, sgn in (("l", 1), ("r", -1)):
        x = X_LANE0 + 0.75
        k = 0
        while x < X_LANE1 - 0.5:
            yr = lane_y(i) + sgn * (LANE_W / 2 - RIB / 2)
            out.append(box(f"lane{i}_rib_{side}{k}", x, yr, WALL_H / 2, RIB, RIB, WALL_H, COL["rib"]))
            x += RIB_STEP
            k += 1
    return out


def world():
    m = boundary_walls()
    for i in range(N_LANES):
        m += lane_walls(i)
    # lane 0: smooth control (nothing)
    m.append(patch("lane1_ice", 1, X_T0, X_T1, 0.1, COL["ice"]))
    m.append(patch("lane2_mild", 2, X_T0, X_T1, 0.3, COL["mild"]))
    m.append(rumble_strips("lane3_rough", 3, X_T0, X_T1, seed=3))
    # lane 4: up (grippy) -> plateau -> down (ice)
    m.append(ramp("lane4_ramp_up", 4, X_T0, RAMP_RUN, +1, 1.0, COL["ramp"]))
    m.append(plateau("lane4_plateau", 4, X_T0 + RAMP_RUN, X_T0 + RAMP_RUN + 2.0, 1.0, COL["ramp"]))
    m.append(ramp("lane4_ramp_down_ice", 4, X_T0 + RAMP_RUN + 2.0, RAMP_RUN, -1, 0.1, COL["ice"]))
    # lane 5: mixed sequence
    m.append(patch("lane5_ice", 5, 4.0, 6.0, 0.1, COL["ice"]))
    m.append(rumble_strips("lane5_rough", 5, 6.0, 8.0, seed=5))
    m.append(patch("lane5_mild", 5, 8.0, 10.0, 0.3, COL["mild"]))
    m.append(patch("lane5_severe", 5, 10.0, 12.0, 0.05, COL["severe"]))
    # outer room
    y0, y1 = -LANE_W / 2 - WALL_T - 2.5, lane_y(N_LANES - 1) + LANE_W / 2 + WALL_T + 2.5
    x0, x1 = -4.0, 19.0
    cx, cy, sx, sy = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    m += [box("outer_s", cx, y0, 1.0, sx, 0.3, 2.0, COL["wall"]),
          box("outer_n", cx, y1, 1.0, sx, 0.3, 2.0, COL["wall"]),
          box("outer_w", x0, cy, 1.0, 0.3, sy, 2.0, COL["wall"]),
          box("outer_e", x1, cy, 1.0, 0.3, sy, 2.0, COL["wall"])]
    floor = f"""
    <model name="floor">
      <static>true</static>
      <pose>{cx:.3f} {cy:.3f} -0.05 0 0 0</pose>
      <link name="floor_link">
        <collision name="floor_collision"><geometry><box><size>{sx + 2} {sy + 2} 0.1</size></box></geometry></collision>
        <visual name="floor_visual"><geometry><box><size>{sx + 2} {sy + 2} 0.1</size></box></geometry>
          <material><ambient>0.80 0.78 0.72 1</ambient><diffuse>0.80 0.78 0.72 1</diffuse></material></visual>
      </link>
    </model>"""
    return f"""<?xml version="1.0" ?>
<!-- GENERATED by tools/gen_terrain_world.py — edit the generator, not this file. -->
<sdf version="1.6">
  <world name="terrain_world">
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.3 0.3 0.3 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
{floor}
{''.join(m)}
  </world>
</sdf>
"""


if __name__ == "__main__":
    with open(OUT, "w") as f:
        f.write(world())
    print(f"wrote {os.path.normpath(OUT)}")
    print(f"ramp: {RAMP_DEG} deg, run {RAMP_RUN} m, rise {RAMP_RISE:.3f} m")
    for i, name in enumerate(["smooth", "ice mu0.1", "mild mu0.3", "rough strips", "slope up/plateau/ice down", "mixed"]):
        print(f"  lane {i}: y = {lane_y(i):5.2f}   {name}")
