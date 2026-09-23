"""Render the optional mobile base tracking a commanded path.

This is a mobility demonstration, not a strike.  No planner commands the base
yet -- stance selection is the next stage -- so the path here is a fixed
minimum-jerk tour through a few stances, chosen only to exercise the slide-x,
slide-y and yaw joints.  The arm is held at its ready pose throughout.  The
printed tracking error is what the base actuators actually achieved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render_tennis_strike import (
    HEADLIGHT_AMBIENT,
    HEADLIGHT_DIFFUSE,
    HEADLIGHT_SPECULAR,
)
from tennis_vla.environment import make_tennis_contact_model
from tennis_vla.robot import (
    BOLTED_BASE_POSITION_M,
    MobileBase,
    arm_layout,
    tennis_ready_configuration,
)

# (time s, x displacement m, y displacement m, yaw rad).  Displacements are
# from the bolted stance, so the tour starts and ends where the fixed arm sits.
TOUR = (
    (0.0, 0.0, 0.0, 0.0),
    (1.2, 0.0, 2.5, 0.35),
    (2.6, 2.0, 0.0, 0.0),
    (4.0, 0.0, -2.5, -0.35),
    (5.2, 0.0, 0.0, 0.0),
)


def _minimum_jerk(fraction: float) -> float:
    fraction = min(max(fraction, 0.0), 1.0)
    return fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)


def _minimum_jerk_rate(fraction: float) -> float:
    if not 0.0 <= fraction <= 1.0:
        return 0.0
    return 30.0 * fraction**2 * (1.0 - fraction) ** 2


def commanded_stance(time_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-jerk position and velocity between tour waypoints."""
    for (t0, *a), (t1, *b) in zip(TOUR, TOUR[1:]):
        if time_s <= t1:
            fraction = (time_s - t0) / (t1 - t0)
            delta = np.asarray(b) - np.asarray(a)
            return (
                np.asarray(a) + _minimum_jerk(fraction) * delta,
                _minimum_jerk_rate(fraction) * delta / (t1 - t0),
            )
    return np.asarray(TOUR[-1][1:], dtype=float), np.zeros(3)


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Behind the robot looking down the court, so lateral travel reads as
    # left-right motion across the frame.
    camera.lookat[:] = [BOLTED_BASE_POSITION_M[0] + 1.2, 0.0, 0.5]
    camera.distance = 7.5
    camera.azimuth = 0.0
    camera.elevation = -30.0
    return camera


def render(output: Path, *, width: int, height: int, fps: float) -> dict[str, float]:
    model = make_tennis_contact_model(base=MobileBase())
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    model.vis.headlight.ambient[:] = HEADLIGHT_AMBIENT
    model.vis.headlight.diffuse[:] = HEADLIGHT_DIFFUSE
    model.vis.headlight.specular[:] = HEADLIGHT_SPECULAR

    layout = arm_layout(model)
    data = mujoco.MjData(model)
    ready = tennis_ready_configuration(model)
    data.qpos[layout.arm_qpos] = ready
    data.ctrl[layout.arm_actuators] = ready
    # Park the ball on the far court, out of shot.
    ball = model.joint("ball_free").qposadr[0]
    data.qpos[ball : ball + 3] = [10.0, 6.0, 0.0335]
    mujoco.mj_forward(model, data)

    duration_s = TOUR[-1][0] + 0.4
    steps = int(round(duration_s / model.opt.timestep))
    frame_every = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    maximum_error = np.zeros(3)
    frames: list[Image.Image] = []
    camera = _camera()
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for step in range(steps):
            target, target_velocity = commanded_stance(data.time)
            # Same velocity compensation the arm servo uses: lead the position
            # command so the actuator's own damping does not drag the base.
            gain = model.actuator_gainprm[layout.base_actuators, 0]
            damping = -model.actuator_biasprm[layout.base_actuators, 2]
            data.ctrl[layout.base_actuators] = target + damping * target_velocity / gain
            data.ctrl[layout.arm_actuators] = ready
            # Gravity compensation keeps the arm at ready while the base moves.
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[layout.arm_dof] = data.qfrc_bias[layout.arm_dof]
            mujoco.mj_step(model, data)
            maximum_error = np.maximum(
                maximum_error, np.abs(data.qpos[layout.base_qpos] - target)
            )
            if step % frame_every:
                continue
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            x, y, yaw = data.qpos[layout.base_qpos]
            for row, text in enumerate(
                (
                    f"t = {data.time:4.2f} s   mobile base demo (no planner)",
                    f"base x {BOLTED_BASE_POSITION_M[0] + x:+6.2f} m  "
                    f"y {y:+5.2f} m  yaw {np.rad2deg(yaw):+5.1f} deg",
                )
            ):
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    draw.text((10 + dx, 8 + 14 * row + dy), text, fill=(0, 0, 0))
                draw.text((10, 8 + 14 * row), text, fill=(235, 235, 235))
            frames.append(frame.convert("P", palette=Image.ADAPTIVE, colors=96))

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000.0 / fps)),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return {
        "frames": len(frames),
        "size_bytes": output.stat().st_size,
        "max_x_tracking_error_m": float(maximum_error[0]),
        "max_y_tracking_error_m": float(maximum_error[1]),
        "max_yaw_tracking_error_rad": float(maximum_error[2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/media/mobile_base.gif"))
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--height", type=int, default=315)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    for key, value in render(args.output, width=args.width, height=args.height, fps=args.fps).items():
        print(f"{key}={value}")
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
