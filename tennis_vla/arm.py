"""Pinned 7-DoF reference arm and racket attachment for tennis simulation."""

from __future__ import annotations

import importlib.metadata
from typing import Any

import mujoco
import mujoco_menagerie as menagerie
import numpy as np


ARM_MODEL = "rethink_robotics_sawyer"
ARM_ENTRY = "sawyer"
ARM_OID = "4b0d742b4136b8f8ec9a30af036dea239588d485"
ARM_ARCHIVE_SHA256 = "79bd43888554d98bd4e5b5fc784f4fd384653b6da160495d55cebb821faaf97e"


def _verified_robot() -> menagerie.Robot:
    robot = menagerie.get(ARM_MODEL)
    if robot.oid != ARM_OID or robot.sha256 != ARM_ARCHIVE_SHA256:
        raise RuntimeError(
            "MuJoCo Menagerie Sawyer asset changed; review and repin it before use"
        )
    return robot


def make_sawyer_racket_spec() -> mujoco.MjSpec:
    """Load the pinned Sawyer spec and attach a lightweight racket."""
    spec = _verified_robot().spec(ARM_ENTRY)
    racket = spec.body("right_l6").add_body(
        name="tennis_racket",
        pos=[0.0, 0.0, 0.0245],
    )
    racket.add_geom(
        name="racket_handle",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.0, 0.0, 0.0, 0.0, 0.0, -0.30],
        size=[0.015, 0.0, 0.0],
        density=280.0,
        rgba=[0.12, 0.12, 0.14, 1.0],
        contype=1,
        conaffinity=1,
    )
    racket.add_geom(
        name="racket_head",
        type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        pos=[0.0, 0.0, -0.48],
        # The head lies in the local x-z plane; local y is its contact normal.
        size=[0.18, 0.015, 0.135],
        density=115.0,
        rgba=[0.18, 0.22, 0.28, 0.90],
        contype=1,
        conaffinity=1,
    )
    racket.add_site(
        name="racket_center",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0.0, 0.0, -0.48],
        quat=[0.70710678, -0.70710678, 0.0, 0.0],
        size=[0.012, 0.0, 0.0],
        rgba=[0.95, 0.75, 0.12, 1.0],
    )
    racket.add_site(
        name="racket_tip",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0.0, 0.0, -0.66],
        size=[0.009, 0.0, 0.0],
        rgba=[0.95, 0.30, 0.12, 1.0],
    )
    return spec


def make_sawyer_racket_model() -> mujoco.MjModel:
    """Compile the pinned Sawyer and its generated racket."""
    return make_sawyer_racket_spec().compile()


def home_configuration(model: mujoco.MjModel) -> np.ndarray:
    if model.nkey < 1:
        raise RuntimeError("Pinned Sawyer model has no home keyframe")
    return model.key_qpos[0, :7].copy()


def audit_workspace(samples: int = 20_000, seed: int = 2026) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("samples must be positive")
    model = make_sawyer_racket_model()
    if model.nq != 7 or model.nu != 7:
        raise RuntimeError(f"Expected a 7-DoF arm, got nq={model.nq}, nu={model.nu}")
    data = mujoco.MjData(model)
    center_id = model.site("racket_center").id
    tip_id = model.site("racket_tip").id
    rng = np.random.default_rng(seed)
    lower, upper = model.jnt_range[:, 0], model.jnt_range[:, 1]
    centers = np.empty((samples, 3), dtype=np.float64)
    tips = np.empty((samples, 3), dtype=np.float64)
    for index in range(samples):
        data.qpos[:] = rng.uniform(lower, upper)
        mujoco.mj_forward(model, data)
        centers[index] = data.site_xpos[center_id]
        tips[index] = data.site_xpos[tip_id]

    data.qpos[:] = home_configuration(model)
    mujoco.mj_forward(model, data)
    home_center = data.site_xpos[center_id].copy()
    radial_reach = np.linalg.norm(centers[:, :2], axis=1)
    racket_mass = float(model.body("tennis_racket").mass[0])
    robot = _verified_robot()
    return {
        "schema_version": 1,
        "arm": {
            "model": ARM_MODEL,
            "entry": ARM_ENTRY,
            "dof": model.nq,
            "actuators": model.nu,
            "license": robot.license,
            "menagerie_version": importlib.metadata.version("mujoco-menagerie"),
            "asset_oid": robot.oid,
            "archive_sha256": robot.sha256,
        },
        "racket": {
            "body_mass_kg": racket_mass,
            "center_offset_from_wrist_m": 0.48,
            "head_semi_axes_m": [0.18, 0.015, 0.135],
        },
        "sampling": {"samples": samples, "seed": seed},
        "home_racket_center_m": home_center.tolist(),
        "racket_center_bounds_m": {
            "minimum": centers.min(axis=0).tolist(),
            "maximum": centers.max(axis=0).tolist(),
        },
        "racket_tip_bounds_m": {
            "minimum": tips.min(axis=0).tolist(),
            "maximum": tips.max(axis=0).tolist(),
        },
        "radial_reach_m": {
            "median": float(np.quantile(radial_reach, 0.50)),
            "p95": float(np.quantile(radial_reach, 0.95)),
            "maximum_sampled": float(radial_reach.max()),
        },
        "limitations": [
            "Uniform joint sampling is a kinematic audit, not a collision-free workspace.",
            "The Menagerie MJCF does not encode authoritative joint-velocity limits.",
            "This selection does not commit the project to Sawyer hardware.",
        ],
    }
