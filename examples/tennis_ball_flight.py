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

from tennis_vla.physics import BallFlightConfig, court_scene_xml, simulate_ball_flight


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
    print(json.dumps(metrics, indent=2))

    if not args.viewer:
        return

    import mujoco.viewer

    model = mujoco.MjModel.from_xml_string(court_scene_xml(config.radius_m))
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
