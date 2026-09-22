"""Benchmark calibrated stereo ball tracking on held-out tennis feeds.

    scripts/run python examples/tennis_perception_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.arm import tennis_ready_configuration
from tennis_vla.environment import make_tennis_contact_model
from tennis_vla.feeder import ProgrammableFeeder
from tennis_vla.perception import (
    fit_constant_velocity_track,
    predict_x_crossing_time,
    triangulate_ball_from_images,
)


def crossing_time_s(times: np.ndarray, x_positions: np.ndarray, plane_x_m: float) -> float:
    """Interpolate the first negative-x crossing of a plane."""
    crossings = np.flatnonzero(x_positions <= plane_x_m)
    if len(crossings) == 0 or crossings[0] == 0:
        raise ValueError(f"trajectory does not cross x={plane_x_m}")
    index = int(crossings[0])
    x0, x1 = x_positions[index - 1 : index + 1]
    fraction = (plane_x_m - x0) / (x1 - x0)
    return float(times[index - 1] + fraction * (times[index] - times[index - 1]))


def _rmse(values: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def run_benchmark(
    seed_start: int = 7000,
    seeds: int = 25,
    width: int = 512,
    height: int = 384,
    observation_hz: float = 50.0,
    prediction_lead_s: float = 0.15,
    history_frames: int = 5,
    strike_plane_x_m: float = -9.25,
) -> dict[str, Any]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    if observation_hz <= 0.0 or prediction_lead_s <= 0.0:
        raise ValueError("observation_hz and prediction_lead_s must be positive")
    if history_frames < 2:
        raise ValueError("history_frames must be at least two")

    model = make_tennis_contact_model()
    data = mujoco.MjData(model)
    data.qpos[:7] = tennis_ready_configuration(model)
    data.ctrl[:] = data.qpos[:7]
    ball_qpos_address = int(model.joint("ball_free").qposadr[0])
    renderer = mujoco.Renderer(model, height=height, width=width)
    feeder = ProgrammableFeeder()

    position_errors: list[float] = []
    contact_time_errors: list[float] = []
    prediction_horizons: list[float] = []
    episode_reports: list[dict[str, Any]] = []
    total_frames = 0
    detected_frames = 0
    try:
        for seed in range(seed_start, seed_start + seeds):
            _, flight = feeder.sample_legal_feed(seed)
            true_contact_time = crossing_time_s(
                flight.times_s, flight.positions_m[:, 0], strike_plane_x_m
            )
            first_observation = true_contact_time - 0.50
            last_observation = true_contact_time - prediction_lead_s
            observation_times = np.arange(
                first_observation,
                last_observation + 1e-12,
                1.0 / observation_hz,
            )
            track_times: list[float] = []
            track_positions: list[np.ndarray] = []
            episode_position_errors: list[float] = []

            for requested_time in observation_times:
                index = int(np.argmin(np.abs(flight.times_s - requested_time)))
                timestamp = float(flight.times_s[index])
                true_position = flight.positions_m[index]
                data.qpos[ball_qpos_address : ball_qpos_address + 3] = true_position
                data.qpos[ball_qpos_address + 3 : ball_qpos_address + 7] = [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                mujoco.mj_forward(model, data)
                images = []
                for camera in ("camera1", "camera2"):
                    renderer.update_scene(data, camera=camera)
                    images.append(renderer.render().copy())

                total_frames += 1
                triangulated = triangulate_ball_from_images(
                    model, data, images[0], images[1]
                )
                if triangulated is None:
                    continue
                detected_frames += 1
                error = float(np.linalg.norm(triangulated.position_m - true_position))
                position_errors.append(error)
                episode_position_errors.append(error)
                track_times.append(timestamp)
                track_positions.append(triangulated.position_m)

            if len(track_times) < history_frames:
                predicted_contact_time = None
                contact_time_error = None
            else:
                track = fit_constant_velocity_track(
                    np.asarray(track_times[-history_frames:]),
                    np.asarray(track_positions[-history_frames:]),
                )
                predicted_contact_time = predict_x_crossing_time(
                    track, strike_plane_x_m
                )
                contact_time_error = (
                    None
                    if predicted_contact_time is None
                    else abs(predicted_contact_time - true_contact_time)
                )
                if contact_time_error is not None:
                    contact_time_errors.append(contact_time_error)
                    prediction_horizons.append(true_contact_time - track.time_s)

            episode_reports.append(
                {
                    "seed": seed,
                    "frames": int(len(observation_times)),
                    "detections": len(track_times),
                    "position_rmse_m": None
                    if not episode_position_errors
                    else _rmse(episode_position_errors),
                    "true_contact_time_s": true_contact_time,
                    "predicted_contact_time_s": predicted_contact_time,
                    "contact_time_error_s": contact_time_error,
                }
            )
    finally:
        renderer.close()

    if not position_errors or not contact_time_errors:
        raise RuntimeError("benchmark produced too few observations")
    position_rmse = _rmse(position_errors)
    contact_time_rmse = _rmse(contact_time_errors)
    return {
        "schema_version": 1,
        "benchmark": "tennis-stereo-perception-v0",
        "scene": {
            "resolution": [width, height],
            "camera_names": ["camera1", "camera2"],
            "canonical_rendering_only": True,
        },
        "evaluation": {
            "seed_start": seed_start,
            "seeds": seeds,
            "observation_hz": observation_hz,
            "prediction_lead_s": prediction_lead_s,
            "history_frames": history_frames,
            "strike_plane_x_m": strike_plane_x_m,
        },
        "metrics": {
            "frames": total_frames,
            "detected_frames": detected_frames,
            "detection_rate": detected_frames / total_frames,
            "position_rmse_m": position_rmse,
            "position_p95_error_m": float(np.quantile(position_errors, 0.95)),
            "contact_time_predictions": len(contact_time_errors),
            "minimum_prediction_horizon_s": float(min(prediction_horizons)),
            "contact_time_mae_s": float(np.mean(contact_time_errors)),
            "contact_time_rmse_s": contact_time_rmse,
            "contact_time_max_error_s": float(max(contact_time_errors)),
        },
        "m1_gate_check": {
            "position_rmse_threshold_m": 0.05,
            "position_rmse_pass": position_rmse <= 0.05,
            "contact_time_rmse_threshold_s": 0.025,
            "contact_time_rmse_pass": contact_time_rmse <= 0.025,
        },
        "episodes": episode_reports,
        "limitations": [
            "This baseline uses canonical lighting and materials without domain randomization.",
            "It does not estimate spin or model motion blur.",
            "Its temporal model assumes constant velocity and can be biased across a bounce.",
            "The M1 milestone remains open until learned and randomized held-out evaluation passes.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=7000)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tennis/perception_baseline_v0.json"),
    )
    args = parser.parse_args()
    report = run_benchmark(
        seed_start=args.seed_start,
        seeds=args.seeds,
        width=args.width,
        height=args.height,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(json.dumps(report["m1_gate_check"], indent=2))
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
