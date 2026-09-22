"""Integrated MuJoCo court, ball, and racket-arm model for contact work."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from .arm import make_sawyer_racket_spec, tennis_ready_configuration
from .ballistics import BallFlightConfig
from .court import TennisCourtSpec
from .impact import RacketImpactConfig


STEREO_CAMERA_TARGET_M = (-6.5, 0.0, 1.0)
STEREO_CAMERA_POSITIONS_M = (
    (-12.5, -5.5, 3.4),
    (-12.5, 5.5, 3.4),
)
STEREO_CAMERA_FOVY_DEG = 55.0
COURT_LINE_GEOMS = (
    "near_baseline",
    "far_baseline",
    "left_doubles_sideline",
    "right_doubles_sideline",
    "left_singles_sideline",
    "right_singles_sideline",
    "near_service_line",
    "far_service_line",
    "center_service_line",
)


@dataclass(frozen=True)
class RacketContactProbe:
    contacted: bool
    contact_step: int | None
    separation_step: int | None
    incoming_normal_speed_m_s: float
    outgoing_normal_speed_m_s: float | None
    predicted_outgoing_normal_speed_m_s: float

    def metrics(self) -> dict[str, Any]:
        result = asdict(self)
        result["normal_speed_error_m_s"] = (
            None
            if self.outgoing_normal_speed_m_s is None
            else abs(
                self.outgoing_normal_speed_m_s
                - self.predicted_outgoing_normal_speed_m_s
            )
        )
        return result


def _camera_xyaxes(
    position: np.ndarray, target: np.ndarray
) -> list[float]:
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.concatenate((right, up)).tolist()


def _add_court_markings(world: mujoco.MjsBody, court: TennisCourtSpec) -> None:
    """Add visual-only regulation lines without changing ball contact."""
    half_length = court.half_length_m
    singles_half_width = court.singles_half_width_m
    doubles_half_width = court.doubles_half_width_m
    service = court.service_line_from_net_m
    line_rgba = [0.95, 0.95, 0.92, 1.0]
    line_height = 0.004
    cross_court_lines = (
        ("near_baseline", -half_length, doubles_half_width, 0.025),
        ("far_baseline", half_length, doubles_half_width, 0.025),
        ("near_service_line", -service, singles_half_width, 0.018),
        ("far_service_line", service, singles_half_width, 0.018),
    )
    for name, x_position, half_width, half_thickness in cross_court_lines:
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x_position, 0.0, line_height],
            size=[half_thickness, half_width, line_height],
            rgba=line_rgba,
            contype=0,
            conaffinity=0,
        )
    side_lines = (
        ("left_doubles_sideline", -doubles_half_width, 0.025),
        ("right_doubles_sideline", doubles_half_width, 0.025),
        ("left_singles_sideline", -singles_half_width, 0.018),
        ("right_singles_sideline", singles_half_width, 0.018),
    )
    for name, y_position, half_thickness in side_lines:
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, y_position, line_height],
            size=[half_length, half_thickness, line_height],
            rgba=line_rgba,
            contype=0,
            conaffinity=0,
        )
    world.add_geom(
        name="center_service_line",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, line_height],
        size=[service, 0.018, line_height],
        rgba=line_rgba,
        contype=0,
        conaffinity=0,
    )


def make_tennis_contact_model(
    racket_solref: tuple[float, float] = (-100_000.0, -50.0),
) -> mujoco.MjModel:
    """Build the first integrated scene with physical court/net/ball contact."""
    court = TennisCourtSpec()
    ball = BallFlightConfig()
    spec = make_sawyer_racket_spec()
    spec.modelname = "tennis_contact_v0"
    spec.option.timestep = 0.001
    spec.option.gravity = [0.0, 0.0, -ball.gravity_m_s2]

    # Put the fixed-base arm behind the near baseline, facing the court.
    spec.body("base").pos = [-10.6, 0.0, 0.0]

    world = spec.worldbody
    world.add_geom(
        name="tennis_court",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[15.0, 8.0, 0.05],
        rgba=[0.12, 0.38, 0.25, 1.0],
        friction=[0.8, 0.02, 0.002],
        contype=1,
        conaffinity=1,
    )
    _add_court_markings(world, court)
    world.add_geom(
        name="tennis_net",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, court.net_center_height_m / 2.0],
        size=[0.025, court.doubles_half_width_m, court.net_center_height_m / 2.0],
        rgba=[0.92, 0.92, 0.92, 0.72],
        friction=[0.4, 0.01, 0.001],
        contype=1,
        conaffinity=1,
    )
    ball_body = world.add_body(name="tennis_ball", pos=[0.0, 0.0, 2.0])
    ball_body.add_freejoint(name="ball_free")
    ball_body.add_geom(
        name="tennis_ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[ball.radius_m, 0.0, 0.0],
        mass=ball.mass_kg,
        rgba=[0.82, 0.95, 0.12, 1.0],
        friction=[0.7, 0.02, 0.002],
        solref=[0.006, 0.7],
        solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
        condim=4,
        contype=1,
        conaffinity=1,
    )
    spec.add_pair(
        name="ball_racket_contact",
        geomname1="tennis_ball_geom",
        geomname2="racket_head",
        condim=4,
        solref=list(racket_solref),
        solimp=[0.99, 0.999, 0.001, 0.5, 2.0],
        friction=[0.7, 0.02, 0.002, 0.0001, 0.0001],
    )
    world.add_camera(
        name="court_camera",
        pos=[-4.0, -5.0, 3.0],
        xyaxes=[0.78, -0.62, 0.0, 0.25, 0.31, 0.92],
    )
    # The stereo pair sits behind the near baseline and converges on the
    # arm's receiving half. This keeps both the bounce and strike zone in view.
    camera_target = np.asarray(STEREO_CAMERA_TARGET_M, dtype=np.float64)
    camera1_position = np.asarray(STEREO_CAMERA_POSITIONS_M[0], dtype=np.float64)
    camera2_position = np.asarray(STEREO_CAMERA_POSITIONS_M[1], dtype=np.float64)
    world.add_camera(
        name="camera1",
        pos=camera1_position.tolist(),
        xyaxes=_camera_xyaxes(camera1_position, camera_target),
        fovy=STEREO_CAMERA_FOVY_DEG,
    )
    world.add_camera(
        name="camera2",
        pos=camera2_position.tolist(),
        xyaxes=_camera_xyaxes(camera2_position, camera_target),
        fovy=STEREO_CAMERA_FOVY_DEG,
    )
    return spec.compile()


def probe_stationary_racket_contact(
    incoming_speed_m_s: float = 5.0,
    maximum_steps: int = 1000,
) -> RacketContactProbe:
    """Launch a ball normally at the held racket and measure its rebound."""
    if incoming_speed_m_s <= 0.0:
        raise ValueError("incoming_speed_m_s must be positive")
    model = make_tennis_contact_model()
    # Isolate contact response from ball drop and arm gravity compensation.
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    data.qpos[:7] = tennis_ready_configuration(model)
    data.ctrl[:] = data.qpos[:7]
    mujoco.mj_forward(model, data)

    ball_joint = model.joint("ball_free")
    qpos_address = int(ball_joint.qposadr[0])
    dof_address = int(ball_joint.dofadr[0])
    racket_site = model.site("racket_center").id
    center = data.site_xpos[racket_site].copy()
    normal = data.site_xmat[racket_site].reshape(3, 3)[:, 2].copy()
    data.qpos[qpos_address : qpos_address + 3] = center + 0.25 * normal
    data.qpos[qpos_address + 3 : qpos_address + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dof_address : dof_address + 3] = -incoming_speed_m_s * normal
    mujoco.mj_forward(model, data)

    ball_geom = model.geom("tennis_ball_geom").id
    racket_geom = model.geom("racket_head").id
    contact_step: int | None = None
    separation_step: int | None = None
    outgoing_speed: float | None = None
    for step in range(maximum_steps):
        mujoco.mj_step(model, data)
        touching = any(
            ball_geom in (data.contact[index].geom1, data.contact[index].geom2)
            and racket_geom in (data.contact[index].geom1, data.contact[index].geom2)
            for index in range(data.ncon)
        )
        if touching and contact_step is None:
            contact_step = step
        elif not touching and contact_step is not None:
            separation_step = step
            outgoing_speed = float(data.qvel[dof_address : dof_address + 3] @ normal)
            break

    impact = RacketImpactConfig()
    return RacketContactProbe(
        contacted=contact_step is not None,
        contact_step=contact_step,
        separation_step=separation_step,
        incoming_normal_speed_m_s=incoming_speed_m_s,
        outgoing_normal_speed_m_s=outgoing_speed,
        predicted_outgoing_normal_speed_m_s=(
            impact.normal_restitution * incoming_speed_m_s
        ),
    )
