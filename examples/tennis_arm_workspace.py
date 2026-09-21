"""Audit or view the pinned 7-DoF Sawyer reference arm with a tennis racket.

    scripts/run python examples/tennis_arm_workspace.py
    scripts/run mjpython examples/tennis_arm_workspace.py --viewer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.arm import audit_workspace, home_configuration, make_sawyer_racket_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results/tennis/sawyer_workspace_v0.json"))
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    report = audit_workspace(samples=args.samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={args.output.resolve()}")

    if not args.viewer:
        return
    import mujoco.viewer

    model = make_sawyer_racket_model()
    data = mujoco.MjData(model)
    data.qpos[:] = home_configuration(model)
    data.ctrl[:] = data.qpos
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
