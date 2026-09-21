"""Run or visualize the first tennis-project physics milestone.

Headless check:
    scripts/run python examples/tennis_ball_flight.py

Interactive macOS view:
    scripts/run mjpython examples/tennis_ball_flight.py --viewer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

# The repository is intentionally a non-packaged uv project; make its root
# importable when this example is launched by file path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.ballistics import BallFlightConfig, simulate_ball_flight


COURT_XML = r"""
<mujoco model="tennis_ball_flight_v0">
  <visual>
    <global offwidth="960" offheight="540"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.75 0.75 0.75"/>
  </visual>
  <worldbody>
    <geom name="court" type="plane" size="15 8 0.05" rgba="0.12 0.38 0.25 1"/>
    <geom name="net" type="box" pos="0 0 0.457" size="0.025 5.49 0.457"
          rgba="0.92 0.92 0.92 0.72" contype="0" conaffinity="0"/>
    <geom name="near_baseline" type="box" pos="-11.885 0 0.006"
          size="0.025 4.115 0.006" rgba="1 1 1 1"/>
    <geom name="far_baseline" type="box" pos="11.885 0 0.006"
          size="0.025 4.115 0.006" rgba="1 1 1 1"/>
    <geom name="left_sideline" type="box" pos="0 -4.115 0.006"
          size="11.885 0.025 0.006" rgba="1 1 1 1"/>
    <geom name="right_sideline" type="box" pos="0 4.115 0.006"
          size="11.885 0.025 0.006" rgba="1 1 1 1"/>
    <camera name="court_camera" pos="-15 -14 9"
            xyaxes="0.683 -0.731 0 0.292 0.273 0.917"/>
    <body name="ball" mocap="true" pos="10.5 0 1.4">
      <geom type="sphere" size="0.0335" rgba="0.82 0.95 0.12 1"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--slowdown", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BallFlightConfig(duration_s=2.0)
    result = simulate_ball_flight(
        position_m=np.array([10.5, 0.0, 1.4]),
        velocity_m_s=np.array([-19.0, 0.6, 4.5]),
        config=config,
    )
    metrics = result.metrics()
    metrics["clears_regulation_net"] = bool(
        result.net_crossing_m is not None and result.net_crossing_m[2] > 0.914
    )
    metrics["first_bounce_in_near_singles_court"] = bool(
        result.first_bounce_m is not None
        and -11.885 <= result.first_bounce_m[0] <= 0.0
        and abs(result.first_bounce_m[1]) <= 4.115
    )
    print(json.dumps(metrics, indent=2))

    if not args.viewer:
        return

    import mujoco.viewer

    model = mujoco.MjModel.from_xml_string(COURT_XML)
    data = mujoco.MjData(model)
    ball_mocap_id = model.body("ball").mocapid[0]
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for position in result.positions_m:
            if not viewer.is_running():
                break
            data.mocap_pos[ball_mocap_id] = position
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(config.dt_s / max(args.slowdown, 1e-3))


if __name__ == "__main__":
    main()
