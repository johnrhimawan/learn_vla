"""Pinned 7-DoF reference arm and racket attachment for tennis simulation."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import mujoco
import mujoco_menagerie as menagerie
import numpy as np


ARM_MODEL = "rethink_robotics_sawyer"
ARM_ENTRY = "sawyer"
ARM_JOINT_NAMES: tuple[str, ...] = (
    "right_j0",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_j6",
)
# Joints belonging to a mobile base carry this prefix.  No such joint exists
# yet; the layout resolves an empty base so the fixed arm stays the default.
BASE_JOINT_PREFIX = "base_"
ARM_OID = "4b0d742b4136b8f8ec9a30af036dea239588d485"
ARM_ARCHIVE_SHA256 = "79bd43888554d98bd4e5b5fc784f4fd384653b6da160495d55cebb821faaf97e"
TENNIS_READY_QPOS_RAD = np.array(
    [
        -0.04168706145607054,
        -0.8169774398390002,
        -0.3227214808141803,
        1.3621882414605269,
        0.26116175300348615,
        1.0734506385100289,
        4.435779923828908,
    ],
    dtype=np.float64,
)


def _contiguous_slice(values: list[int], description: str) -> slice:
    """Return a slice over consecutive indices, or raise if they are not."""
    if not values:
        return slice(0, 0)
    if values != list(range(values[0], values[0] + len(values))):
        raise RuntimeError(f"{description} must occupy consecutive indices: {values}")
    return slice(values[0], values[0] + len(values))


@dataclass(frozen=True)
class EmbodimentLayout:
    """Index ranges for the arm and any mobile-base joints.

    MuJoCo keeps four index spaces that coincide only while every actuated
    joint is a one-DoF hinge with a one-to-one actuator, which is why a literal
    ``[:7]`` works today.  Adding base joints breaks that coincidence silently:
    base joints sort ahead of the arm, so ``qpos[:7]`` would still have the
    right shape while addressing the wrong degrees of freedom.  Resolving each
    space by joint name keeps the arm addressable wherever it lands.
    """

    arm_joints: slice
    arm_qpos: slice
    arm_dof: slice
    arm_actuators: slice
    base_joints: slice
    base_qpos: slice
    base_dof: slice
    base_actuators: slice

    @property
    def arm_dof_count(self) -> int:
        return self.arm_dof.stop - self.arm_dof.start

    @property
    def base_dof_count(self) -> int:
        return self.base_dof.stop - self.base_dof.start

    @property
    def has_mobile_base(self) -> bool:
        return self.base_dof_count > 0

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> EmbodimentLayout:
        """Resolve arm and base index ranges by joint name."""
        joint_names = [model.joint(index).name for index in range(model.njnt)]
        missing = [name for name in ARM_JOINT_NAMES if name not in joint_names]
        if missing:
            raise ValueError(f"model is missing arm joints: {missing}")

        arm_ids = [joint_names.index(name) for name in ARM_JOINT_NAMES]
        base_ids = [
            index
            for index, name in enumerate(joint_names)
            if name.startswith(BASE_JOINT_PREFIX)
        ]

        def ranges(joint_ids: list[int], description: str) -> tuple[slice, slice, slice]:
            joints = _contiguous_slice(joint_ids, f"{description} joints")
            qpos = _contiguous_slice(
                [int(model.jnt_qposadr[index]) for index in joint_ids],
                f"{description} qpos addresses",
            )
            dof = _contiguous_slice(
                [int(model.jnt_dofadr[index]) for index in joint_ids],
                f"{description} dof addresses",
            )
            return joints, qpos, dof

        arm_joints, arm_qpos, arm_dof = ranges(arm_ids, "arm")
        base_joints, base_qpos, base_dof = ranges(base_ids, "base")

        def actuators(joint_ids: list[int], description: str) -> slice:
            actuator_ids = [
                index
                for index in range(model.nu)
                if int(model.actuator_trntype[index]) == mujoco.mjtTrn.mjTRN_JOINT
                and int(model.actuator_trnid[index, 0]) in joint_ids
            ]
            return _contiguous_slice(actuator_ids, f"{description} actuators")

        return cls(
            arm_joints=arm_joints,
            arm_qpos=arm_qpos,
            arm_dof=arm_dof,
            arm_actuators=actuators(arm_ids, "arm"),
            base_joints=base_joints,
            base_qpos=base_qpos,
            base_dof=base_dof,
            base_actuators=actuators(base_ids, "base"),
        )


def arm_layout(model: mujoco.MjModel) -> EmbodimentLayout:
    """Convenience wrapper for the common single-call use."""
    return EmbodimentLayout.from_model(model)


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
    # The compiler pads the pinned 7-value keyframe to the model width, so the
    # arm entries must be selected by layout rather than by a leading slice.
    return model.key_qpos[0, arm_layout(model).arm_qpos].copy()


def tennis_ready_configuration(model: mujoco.MjModel) -> np.ndarray:
    """Return the collision-free phase-one ready pose for the pinned arm."""
    layout = arm_layout(model)
    lower = model.jnt_range[layout.arm_joints, 0]
    upper = model.jnt_range[layout.arm_joints, 1]
    if np.any(TENNIS_READY_QPOS_RAD < lower) or np.any(
        TENNIS_READY_QPOS_RAD > upper
    ):
        raise RuntimeError("tennis ready pose is outside the model joint limits")
    return TENNIS_READY_QPOS_RAD.copy()


def audit_workspace(samples: int = 20_000, seed: int = 2026) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("samples must be positive")
    model = make_sawyer_racket_model()
    layout = arm_layout(model)
    if layout.has_mobile_base:
        raise RuntimeError("workspace audit expects the fixed-base reference arm")
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
