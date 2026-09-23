"""Render the canonical tennis strike execution to an animated GIF.

The animation replays one real ``execute_strike`` run: the fed ball, its court
bounce, the tracked swing, live MuJoCo contact, the measured return over the
net, and the bounded recovery to the ready pose.  Arm and ball states come from
a read-only observer on the 1 kHz execution loop, so the frames show the
executed motion rather than the plan.  Only the coast after the recovery ends
is continued with the analytical flight model, which is how the tracked reports
score the return.

Rendering-only settings (headlight, camera, ball trail) never touch physics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.robot import arm_layout, tennis_ready_configuration
from tennis_vla.environment import make_tennis_contact_model
from tennis_vla.control import (
    StrikeExecutionConfig,
    execute_strike,
    simulate_mujoco_ball_flight,
)
from tennis_vla.planning import plan_safe_center_strikes


CANONICAL_POSITION_M = np.array([10.5, 0.0, 1.4])
CANONICAL_VELOCITY_M_S = np.array([-17.75, 0.0, 4.6])
CANONICAL_PLAN_SEED = 40_001

# The contact model is calibrated for physics and renders almost black under
# the default headlight.  These are visualisation-only.
HEADLIGHT_AMBIENT = 0.60
HEADLIGHT_DIFFUSE = 0.70
HEADLIGHT_SPECULAR = 0.20

CAMERA_AZIMUTH_DEG = 212.0
TRAIL_LENGTH = 14
TRAIL_STRIDE_S = 0.02
TRAIL_RADIUS_M = 0.05
TRAIL_RGBA = (0.95, 0.98, 0.25)


class StateRecorder:
    """Subsample executed states so the animation stays small."""

    def __init__(self, sample_period_s: float) -> None:
        self._sample_period_s = sample_period_s
        self._next_sample_s = 0.0
        self.samples: list[tuple[float, np.ndarray]] = []

    def __call__(self, time_s: float, data: mujoco.MjData) -> None:
        if time_s + 1e-12 < self._next_sample_s:
            return
        self._next_sample_s = time_s + self._sample_period_s
        self.samples.append((time_s, data.qpos.copy()))


def _keyframed(time_s: float, contact_s: float, offsets: list[float], values: list[float]) -> float:
    return float(np.interp(time_s, [contact_s + offset for offset in offsets], values))


def _tracking_camera(
    time_s: float, ball_m: np.ndarray, contact_s: float
) -> mujoco.MjvCamera:
    """Follow the ball, pushing in for contact and pulling back for the return."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [ball_m[0], ball_m[1] * 0.4, max(float(ball_m[2]), 0.8)]
    camera.distance = _keyframed(
        time_s, contact_s, [-contact_s, -0.9, -0.15, 0.3, 1.5], [12.0, 8.5, 4.4, 6.0, 9.5]
    )
    camera.azimuth = CAMERA_AZIMUTH_DEG
    camera.elevation = _keyframed(
        time_s, contact_s, [-contact_s, -0.15, 0.3, 1.5], [-24.0, -12.0, -14.0, -18.0]
    )
    return camera


def _add_ball_trail(
    scene: mujoco.MjvScene, ball_positions_m: np.ndarray, index: int, stride: int
) -> None:
    identity = np.eye(3).flatten()
    size = np.array([TRAIL_RADIUS_M, 0.0, 0.0])
    for step in range(1, TRAIL_LENGTH):
        source = index - step * stride
        if source < 0 or scene.ngeom >= scene.maxgeom:
            break
        alpha = 0.55 * (1.0 - step / TRAIL_LENGTH)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size,
            ball_positions_m[source].astype(np.float64),
            identity,
            np.array([*TRAIL_RGBA, alpha], dtype=np.float32),
        )
        scene.ngeom += 1


def _shaded_text(
    draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]
) -> None:
    """Keep captions readable over both the dark sky and the bright court."""
    x, y = xy
    for offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + offset[0], y + offset[1]), text, fill=(0, 0, 0))
    draw.text(xy, text, fill=fill)


def _caption(
    image: Image.Image,
    time_s: float,
    phase: tuple[str, tuple[int, int, int]],
    footer: str | None,
) -> None:
    draw = ImageDraw.Draw(image)
    _shaded_text(draw, (10, 8), f"t = {time_s:5.3f} s", (235, 235, 235))
    _shaded_text(draw, (10, 22), phase[0], phase[1])
    if footer:
        _shaded_text(draw, (10, image.size[1] - 20), footer, (170, 235, 190))


def _phase(
    time_s: float, contact_s: float | None, separation_s: float | None, recovery_end_s: float | None
) -> tuple[str, tuple[int, int, int]]:
    if separation_s is not None and time_s >= separation_s:
        if recovery_end_s is not None and time_s < recovery_end_s:
            return "return flight + recovery", (140, 230, 255)
        return "return flight", (140, 230, 255)
    if contact_s is not None and time_s >= contact_s:
        return "racket contact", (255, 210, 90)
    return "approach + 250 Hz tracking", (205, 205, 205)


def render_strike_gif(
    output: Path,
    *,
    width: int,
    height: int,
    sample_hz: float,
    fps: float,
    coast_s: float,
    colors: int,
) -> dict[str, Any]:
    model = make_tennis_contact_model()
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    model.vis.headlight.ambient[:] = HEADLIGHT_AMBIENT
    model.vis.headlight.diffuse[:] = HEADLIGHT_DIFFUSE
    model.vis.headlight.specular[:] = HEADLIGHT_SPECULAR

    flight = simulate_mujoco_ball_flight(
        model, CANONICAL_POSITION_M, CANONICAL_VELOCITY_M_S
    )
    plans = plan_safe_center_strikes(model, flight, seed=CANONICAL_PLAN_SEED)
    if not plans:
        raise SystemExit("canonical feed has no legal nonzero-velocity strike")

    recorder = StateRecorder(1.0 / sample_hz)
    execution = execute_strike(
        model,
        plans[0],
        CANONICAL_POSITION_M,
        CANONICAL_VELOCITY_M_S,
        config=StrikeExecutionConfig(),
        observer=recorder,
    )
    if execution.actual_contact_time_s is None or execution.separation_time_s is None:
        raise SystemExit("canonical strike did not contact and separate")

    states = list(recorder.samples)
    layout = arm_layout(model)
    ball_address = model.jnt_qposadr[model.joint("ball_free").id]
    executed_samples = len(states)

    measured_return = execution.measured_return
    if measured_return is not None:
        ready = tennis_ready_configuration(model)
        last_time_s, last_qpos = states[-1]
        step_s = 1.0 / sample_hz
        horizon_s = min(coast_s, float(measured_return.times_s[-1]))
        offset_s = last_time_s - execution.separation_time_s + step_s
        while offset_s <= horizon_s:
            qpos = last_qpos.copy()
            qpos[layout.arm_qpos] = ready
            qpos[ball_address : ball_address + 3] = [
                np.interp(offset_s, measured_return.times_s, measured_return.positions_m[:, axis])
                for axis in range(3)
            ]
            states.append((execution.separation_time_s + offset_s, qpos))
            offset_s += step_s

    ball_positions = np.array(
        [qpos[ball_address : ball_address + 3] for _, qpos in states]
    )
    recovery_end_s = (
        None
        if execution.recovery is None or execution.recovery.duration_s is None
        else execution.separation_time_s + execution.recovery.duration_s
    )
    footer = None
    if measured_return is not None and measured_return.legal_first_bounce:
        clearance = measured_return.net_clearance_m
        bounce = measured_return.first_bounce_m
        footer = (
            f"legal return  |  net +{clearance:.2f} m  |  "
            f"bounce x={bounce[0]:.1f} m y={bounce[1]:+.1f} m"
        )

    trail_stride = max(1, int(round(TRAIL_STRIDE_S * sample_hz)))
    render_data = mujoco.MjData(model)
    frames: list[Image.Image] = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for index, (time_s, qpos) in enumerate(states):
            render_data.qpos[:] = qpos
            mujoco.mj_forward(model, render_data)
            camera = _tracking_camera(
                time_s, ball_positions[index], execution.actual_contact_time_s
            )
            renderer.update_scene(render_data, camera=camera)
            _add_ball_trail(renderer.scene, ball_positions, index, trail_stride)
            frame = Image.fromarray(renderer.render())
            _caption(
                frame,
                time_s,
                _phase(
                    time_s,
                    execution.actual_contact_time_s,
                    execution.separation_time_s,
                    recovery_end_s,
                ),
                footer if time_s >= execution.separation_time_s else None,
            )
            frames.append(frame.convert("P", palette=Image.ADAPTIVE, colors=colors))

    frame_ms = int(round(1000.0 / fps))
    durations = [frame_ms] * len(frames)
    durations[-1] = max(frame_ms, 900)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )

    return {
        "output": str(output),
        "frames": len(frames),
        "executed_samples": executed_samples,
        "coast_samples": len(frames) - executed_samples,
        "size_bytes": output.stat().st_size,
        "contacted": execution.contacted,
        "actual_contact_time_s": execution.actual_contact_time_s,
        "contact_time_error_s": execution.contact_time_error_s,
        "contact_position_error_m": execution.contact_position_error_m,
        "legal_return": None if measured_return is None else measured_return.legal_first_bounce,
        "net_clearance_m": None if measured_return is None else measured_return.net_clearance_m,
        "first_bounce_m": (
            None
            if measured_return is None or measured_return.first_bounce_m is None
            else measured_return.first_bounce_m.tolist()
        ),
        "recovery_duration_s": (
            None if execution.recovery is None else execution.recovery.duration_s
        ),
        "recovered": execution.recovery is not None and execution.recovery.passed,
        "passed": execution.passed,
        "failure_reasons": list(execution.failure_reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/media/canonical_strike.gif")
    )
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--height", type=int, default=315)
    parser.add_argument("--sample-hz", type=float, default=40.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--coast-s", type=float, default=1.95)
    parser.add_argument("--colors", type=int, default=96)
    args = parser.parse_args()

    summary = render_strike_gif(
        args.output,
        width=args.width,
        height=args.height,
        sample_hz=args.sample_hz,
        fps=args.fps,
        coast_s=args.coast_s,
        colors=args.colors,
    )
    for key, value in summary.items():
        print(f"{key}={value}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
