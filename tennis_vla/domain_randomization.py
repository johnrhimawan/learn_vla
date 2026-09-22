"""Deterministic visual and camera randomization for tennis perception data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from .environment import (
    COURT_LINE_GEOMS,
    STEREO_CAMERA_FOVY_DEG,
    STEREO_CAMERA_POSITIONS_M,
    STEREO_CAMERA_TARGET_M,
)


@dataclass(frozen=True)
class CameraDomain:
    position_m: tuple[float, float, float]
    target_m: tuple[float, float, float]
    fovy_deg: float


@dataclass(frozen=True)
class RenderDomain:
    seed: int
    court_rgb: tuple[float, float, float]
    line_rgb: tuple[float, float, float]
    ball_rgb: tuple[float, float, float]
    net_rgb: tuple[float, float, float]
    ambient_rgb: tuple[float, float, float]
    diffuse_rgb: tuple[float, float, float]
    specular_rgb: tuple[float, float, float]
    gamma: float
    sensor_noise_std: float
    exposure_s: float
    cameras: tuple[CameraDomain, CameraDomain]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple3(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in values)


def sample_render_domain(seed: int) -> RenderDomain:
    """Sample a repeatable visual domain independently for each episode."""
    rng = np.random.default_rng(seed)
    target = np.asarray(STEREO_CAMERA_TARGET_M, dtype=np.float64) + rng.uniform(
        [-0.20, -0.15, -0.10], [0.20, 0.15, 0.10]
    )
    cameras: list[CameraDomain] = []
    for base_position in STEREO_CAMERA_POSITIONS_M:
        position = np.asarray(base_position, dtype=np.float64) + rng.uniform(
            [-0.18, -0.18, -0.12], [0.18, 0.18, 0.12]
        )
        cameras.append(
            CameraDomain(
                position_m=_tuple3(position),
                target_m=_tuple3(target),
                fovy_deg=float(
                    STEREO_CAMERA_FOVY_DEG + rng.uniform(-3.0, 3.0)
                ),
            )
        )

    brightness = rng.uniform(0.75, 1.15)
    court = np.array(
        [rng.uniform(0.07, 0.22), rng.uniform(0.26, 0.50), rng.uniform(0.13, 0.32)]
    )
    ball = np.array(
        [rng.uniform(0.68, 0.94), rng.uniform(0.78, 1.0), rng.uniform(0.05, 0.28)]
    )
    return RenderDomain(
        seed=seed,
        court_rgb=_tuple3(np.clip(court * brightness, 0.0, 1.0)),
        line_rgb=_tuple3(rng.uniform(0.72, 1.0, size=3)),
        ball_rgb=_tuple3(np.clip(ball * brightness, 0.0, 1.0)),
        net_rgb=_tuple3(rng.uniform(0.68, 1.0, size=3)),
        ambient_rgb=_tuple3(rng.uniform(0.08, 0.42, size=3)),
        diffuse_rgb=_tuple3(rng.uniform(0.38, 0.90, size=3)),
        specular_rgb=_tuple3(rng.uniform(0.10, 0.50, size=3)),
        gamma=float(rng.uniform(0.80, 1.22)),
        sensor_noise_std=float(rng.uniform(0.0, 5.0)),
        exposure_s=float(rng.uniform(0.0, 0.008)),
        cameras=(cameras[0], cameras[1]),
    )


def look_at_quaternion(position_m: np.ndarray, target_m: np.ndarray) -> np.ndarray:
    """Return a MuJoCo camera quaternion whose local -z axis sees the target."""
    position = np.asarray(position_m, dtype=np.float64).reshape(3)
    target = np.asarray(target_m, dtype=np.float64).reshape(3)
    forward = target - position
    norm = float(np.linalg.norm(forward))
    if norm < 1e-9:
        raise ValueError("camera position and target must differ")
    forward /= norm
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-9:
        raise ValueError("camera view cannot be parallel to world up")
    right /= right_norm
    up = np.cross(right, forward)
    rotation = np.column_stack((right, up, -forward))
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    return quaternion


def apply_render_domain(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    domain: RenderDomain,
) -> None:
    """Apply one sampled domain to a compiled model and update derived poses."""
    model.geom("tennis_court").rgba[:3] = domain.court_rgb
    model.geom("tennis_ball_geom").rgba[:3] = domain.ball_rgb
    model.geom("tennis_net").rgba[:3] = domain.net_rgb
    for name in COURT_LINE_GEOMS:
        model.geom(name).rgba[:3] = domain.line_rgb

    model.vis.headlight.ambient[:] = domain.ambient_rgb
    model.vis.headlight.diffuse[:] = domain.diffuse_rgb
    model.vis.headlight.specular[:] = domain.specular_rgb
    if model.nlight:
        model.light_ambient[:] = 0.25 * np.asarray(domain.ambient_rgb)
        model.light_diffuse[:] = domain.diffuse_rgb
        model.light_specular[:] = domain.specular_rgb

    for name, camera in zip(("camera1", "camera2"), domain.cameras, strict=True):
        camera_id = model.camera(name).id
        model.cam_pos[camera_id] = camera.position_m
        model.cam_quat[camera_id] = look_at_quaternion(
            np.asarray(camera.position_m), np.asarray(camera.target_m)
        )
        model.cam_fovy[camera_id] = camera.fovy_deg
    mujoco.mj_forward(model, data)


def postprocess_render(
    image: np.ndarray,
    domain: RenderDomain,
    noise_seed: int,
) -> np.ndarray:
    """Apply deterministic camera gamma and sensor noise to an RGB render."""
    pixels = np.asarray(image, dtype=np.float64) / 255.0
    pixels = np.power(np.clip(pixels, 0.0, 1.0), domain.gamma)
    if domain.sensor_noise_std > 0.0:
        rng = np.random.default_rng(noise_seed)
        pixels += rng.normal(0.0, domain.sensor_noise_std / 255.0, pixels.shape)
    return np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
