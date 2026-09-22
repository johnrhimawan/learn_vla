"""Calibrated two-camera baseline for the M1 tennis-ball tracking task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class BallDetection:
    pixel_xy: np.ndarray
    pixel_count: int


@dataclass(frozen=True)
class TriangulatedBall:
    position_m: np.ndarray
    ray_gap_m: float


@dataclass(frozen=True)
class BallTrackEstimate:
    time_s: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray


def camera_calibration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """Export the intrinsics and world pose used by :func:`camera_ray`."""
    height, width = image_shape
    camera_id = model.camera(camera_name).id
    fovy_deg = float(model.cam_fovy[camera_id])
    focal_px = float(0.5 * height / np.tan(0.5 * np.deg2rad(fovy_deg)))
    principal_point = [(width - 1) / 2.0, (height - 1) / 2.0]
    return {
        "image_size_px": [width, height],
        "fovy_deg": fovy_deg,
        "focal_length_px": [focal_px, focal_px],
        "principal_point_px": principal_point,
        "camera_origin_world_m": data.cam_xpos[camera_id].tolist(),
        "camera_to_world_rotation": data.cam_xmat[camera_id]
        .reshape(3, 3)
        .tolist(),
        "projection_convention": (
            "camera looks along -z; u=cx+f*x/(-z); v=cy-f*y/(-z)"
        ),
    }


def detect_yellow_ball(image: np.ndarray) -> BallDetection | None:
    """Find the simulated yellow ball without accessing simulator state."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    # Ratios are more stable than fixed high channel values under MuJoCo's
    # distance-dependent lighting. The simulated ball stays yellow-green while
    # the court remains dark green and the racket tip remains orange.
    red = image[..., 0].astype(np.float64)
    green = image[..., 1].astype(np.float64)
    blue = image[..., 2].astype(np.float64)
    mask = (
        (red > 45.0)
        & (green > 50.0)
        & (green > 1.06 * red)
        & (red > 1.8 * blue)
        & (green > 2.0 * blue)
    )
    rows, columns = np.nonzero(mask)
    if len(columns) == 0:
        return None
    return BallDetection(
        pixel_xy=np.array([columns.mean(), rows.mean()], dtype=np.float64),
        pixel_count=int(len(columns)),
    )


def camera_ray(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    pixel_xy: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an image pixel into a normalized world-space camera ray."""
    height, width = image_shape
    camera_id = model.camera(camera_name).id
    fovy = np.deg2rad(model.cam_fovy[camera_id])
    focal = 0.5 * height / np.tan(0.5 * fovy)
    pixel = np.asarray(pixel_xy, dtype=np.float64).reshape(2)
    camera_direction = np.array(
        [
            (pixel[0] - (width - 1) / 2.0) / focal,
            -((pixel[1] - (height - 1) / 2.0) / focal),
            -1.0,
        ]
    )
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    world_direction = rotation @ camera_direction
    world_direction /= np.linalg.norm(world_direction)
    return data.cam_xpos[camera_id].copy(), world_direction


def triangulate_rays(
    origin1: np.ndarray,
    direction1: np.ndarray,
    origin2: np.ndarray,
    direction2: np.ndarray,
) -> TriangulatedBall:
    """Return the midpoint of the closest points on two observation rays."""
    origin1 = np.asarray(origin1, dtype=np.float64).reshape(3)
    origin2 = np.asarray(origin2, dtype=np.float64).reshape(3)
    direction1 = np.asarray(direction1, dtype=np.float64).reshape(3)
    direction2 = np.asarray(direction2, dtype=np.float64).reshape(3)
    direction1 /= np.linalg.norm(direction1)
    direction2 /= np.linalg.norm(direction2)
    matrix = np.column_stack((direction1, -direction2))
    parameters, *_ = np.linalg.lstsq(matrix, origin2 - origin1, rcond=None)
    point1 = origin1 + parameters[0] * direction1
    point2 = origin2 + parameters[1] * direction2
    return TriangulatedBall(
        position_m=0.5 * (point1 + point2),
        ray_gap_m=float(np.linalg.norm(point1 - point2)),
    )


def triangulate_ball_from_images(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    image1: np.ndarray,
    image2: np.ndarray,
    camera1: str = "camera1",
    camera2: str = "camera2",
) -> TriangulatedBall | None:
    detection1 = detect_yellow_ball(image1)
    detection2 = detect_yellow_ball(image2)
    if detection1 is None or detection2 is None:
        return None
    ray1 = camera_ray(model, data, camera1, detection1.pixel_xy, image1.shape[:2])
    ray2 = camera_ray(model, data, camera2, detection2.pixel_xy, image2.shape[:2])
    return triangulate_rays(*ray1, *ray2)


def fit_constant_velocity_track(
    times_s: np.ndarray,
    positions_m: np.ndarray,
) -> BallTrackEstimate:
    """Fit a local constant-velocity model and evaluate it at the latest time."""
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.shape != (len(times), 3):
        raise ValueError("positions_m must have shape (len(times_s), 3)")
    if len(times) < 2:
        raise ValueError("at least two observations are required")
    if not np.isfinite(times).all() or not np.isfinite(positions).all():
        raise ValueError("track observations must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing")

    latest_time = float(times[-1])
    relative_times = times - latest_time
    design = np.column_stack((np.ones(len(times)), relative_times))
    coefficients, *_ = np.linalg.lstsq(design, positions, rcond=None)
    return BallTrackEstimate(
        time_s=latest_time,
        position_m=coefficients[0],
        velocity_m_s=coefficients[1],
    )


def predict_x_crossing_time(
    track: BallTrackEstimate,
    plane_x_m: float,
) -> float | None:
    """Predict the absolute time at which a local track reaches an x plane."""
    velocity_x = float(track.velocity_m_s[0])
    if abs(velocity_x) < 1e-9:
        return None
    delta_s = (float(plane_x_m) - float(track.position_m[0])) / velocity_x
    if delta_s < 0.0:
        return None
    return track.time_s + delta_s
