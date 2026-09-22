"""Evaluate the non-learned stereo baseline on a tennis-flight dataset."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .perception import (
    BallDetection,
    camera_ray_from_calibration,
    detect_yellow_ball,
    fit_constant_velocity_track,
    predict_x_crossing_time,
    triangulate_rays,
)


def _repository_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"git_revision": revision, "tracked_files_dirty": dirty}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _metric_summary(
    frames: int,
    position_errors: list[float],
    contact_time_errors: list[float],
    episode_count: int,
) -> dict[str, Any]:
    detections = len(position_errors)
    predictions = len(contact_time_errors)
    return {
        "frames": frames,
        "stereo_detections": detections,
        "detection_rate": 0.0 if frames == 0 else detections / frames,
        "position_rmse_m": None
        if not position_errors
        else float(np.sqrt(np.mean(np.square(position_errors)))),
        "position_p95_error_m": None
        if not position_errors
        else float(np.quantile(position_errors, 0.95)),
        "episodes": episode_count,
        "contact_time_predictions": predictions,
        "contact_time_prediction_rate": (
            0.0 if episode_count == 0 else predictions / episode_count
        ),
        "contact_time_rmse_s": None
        if not contact_time_errors
        else float(np.sqrt(np.mean(np.square(contact_time_errors)))),
        "contact_time_max_error_s": None
        if not contact_time_errors
        else float(max(contact_time_errors)),
    }


def evaluate_stereo_detector(
    dataset_dir: Path,
    detector: Callable[[np.ndarray], BallDetection | None],
    *,
    benchmark_name: str,
    detector_source: dict[str, Any],
    prediction_lead_s: float = 0.15,
    history_window_s: float = 0.10,
    maximum_ray_gap_m: float = 0.10,
) -> dict[str, Any]:
    """Score an image detector without accessing privileged state at inference."""
    dataset_dir = Path(dataset_dir)
    info = json.loads(
        (dataset_dir / "dataset_info.json").read_text(encoding="utf-8")
    )
    episodes = _load_jsonl(dataset_dir / "episodes.jsonl")
    frames = _load_jsonl(dataset_dir / "frames.jsonl")
    episode_by_id = {episode["episode_id"]: episode for episode in episodes}
    tracks: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
    position_errors: dict[str, list[float]] = defaultdict(list)
    frame_counts: dict[str, int] = defaultdict(int)

    for frame in frames:
        split = frame["split"]
        frame_counts[split] += 1
        episode = episode_by_id[frame["episode_id"]]
        rays: list[np.ndarray] = []
        for camera_name in ("camera1", "camera2"):
            image = np.asarray(
                Image.open(dataset_dir / frame["images"][camera_name]).convert("RGB")
            )
            detection = detector(image)
            if detection is None:
                break
            origin, direction = camera_ray_from_calibration(
                episode["camera_calibration"][camera_name], detection.pixel_xy
            )
            rays.extend((origin, direction))
        else:
            triangulated = triangulate_rays(*rays)
            if triangulated.ray_gap_m > maximum_ray_gap_m:
                continue
            truth = np.asarray(frame["ball"]["position_world_m"], dtype=np.float64)
            position_errors[split].append(
                float(np.linalg.norm(triangulated.position_m - truth))
            )
            tracks[frame["episode_id"]].append(
                (float(frame["timestamp_s"]), triangulated.position_m)
            )

    contact_time_errors: dict[str, list[float]] = defaultdict(list)
    episode_counts: dict[str, int] = defaultdict(int)
    for episode in episodes:
        split = episode["split"]
        episode_counts[split] += 1
        true_contact_time = float(episode["feed"]["strike_plane_time_s"])
        strike_plane_x = float(info["strike_plane_x_m"])
        eligible = [
            observation
            for observation in tracks[episode["episode_id"]]
            if observation[0] <= true_contact_time - prediction_lead_s
        ]
        if len(eligible) < 2:
            continue
        latest_track_time = eligible[-1][0]
        selected = [
            observation
            for observation in eligible
            if observation[0] >= latest_track_time - history_window_s - 1e-12
        ]
        if len(selected) < 2:
            continue
        track = fit_constant_velocity_track(
            np.asarray([item[0] for item in selected]),
            np.asarray([item[1] for item in selected]),
        )
        prediction = predict_x_crossing_time(track, strike_plane_x)
        if prediction is not None:
            contact_time_errors[split].append(abs(prediction - true_contact_time))

    split_metrics = {
        split: _metric_summary(
            frame_counts[split],
            position_errors[split],
            contact_time_errors[split],
            episode_counts[split],
        )
        for split in ("train", "validation", "test")
    }
    all_position_errors = [
        error for errors in position_errors.values() for error in errors
    ]
    all_contact_errors = [
        error for errors in contact_time_errors.values() for error in errors
    ]
    overall = _metric_summary(
        len(frames), all_position_errors, all_contact_errors, len(episodes)
    )
    minimum_detection_rate = 0.99
    minimum_prediction_rate = 0.99
    position_threshold = 0.05
    contact_time_threshold = 0.025
    gate_split = "test" if episode_counts["test"] else "overall"
    gate_metrics = split_metrics["test"] if gate_split == "test" else overall
    return {
        "schema_version": 1,
        "benchmark": benchmark_name,
        "evaluator_source": _repository_state(),
        "detector_source": detector_source,
        "dataset": {
            "name": info["dataset"],
            "source": info["source"],
            "episodes": info["episodes"],
            "frames": info["frames"],
        },
        "inference_inputs": ["camera1 RGB", "camera2 RGB", "camera calibration"],
        "privileged_state_used_for_scoring_only": True,
        "prediction_lead_s": prediction_lead_s,
        "history_window_s": history_window_s,
        "maximum_stereo_ray_gap_m": maximum_ray_gap_m,
        "metrics": overall,
        "split_metrics": split_metrics,
        "m1_gate_check": {
            "evaluated_split": gate_split,
            "minimum_detection_rate": minimum_detection_rate,
            "detection_rate_pass": gate_metrics["detection_rate"]
            >= minimum_detection_rate,
            "position_rmse_threshold_m": position_threshold,
            "position_rmse_pass": (
                gate_metrics["position_rmse_m"] is not None
                and gate_metrics["detection_rate"] >= minimum_detection_rate
                and gate_metrics["position_rmse_m"] <= position_threshold
            ),
            "minimum_contact_time_prediction_rate": minimum_prediction_rate,
            "contact_time_prediction_rate_pass": (
                gate_metrics["contact_time_prediction_rate"]
                >= minimum_prediction_rate
            ),
            "contact_time_rmse_threshold_s": contact_time_threshold,
            "contact_time_rmse_pass": (
                gate_metrics["contact_time_rmse_s"] is not None
                and gate_metrics["contact_time_prediction_rate"]
                >= minimum_prediction_rate
                and gate_metrics["contact_time_rmse_s"] <= contact_time_threshold
            ),
        },
        "interpretation": (
            "Accuracy on detected frames cannot compensate for missed stereo pairs; "
            "the randomized gate requires both coverage and error thresholds."
        ),
    }


def evaluate_color_stereo_baseline(
    dataset_dir: Path,
    *,
    prediction_lead_s: float = 0.15,
    history_window_s: float = 0.10,
    maximum_ray_gap_m: float = 0.10,
) -> dict[str, Any]:
    """Score fixed yellow segmentation without accessing state during inference."""
    return evaluate_stereo_detector(
        dataset_dir,
        detect_yellow_ball,
        benchmark_name="randomized-color-stereo-baseline-v0",
        detector_source={"type": "fixed_rgb_ratio"},
        prediction_lead_s=prediction_lead_s,
        history_window_s=history_window_s,
        maximum_ray_gap_m=maximum_ray_gap_m,
    )
