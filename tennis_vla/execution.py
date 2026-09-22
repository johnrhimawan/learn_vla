"""Integrated MuJoCo execution checks for privileged tennis strikes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import mujoco
import numpy as np

from .arm import tennis_ready_configuration
from .ballistics import BallFlightConfig, BallFlightResult, simulate_ball_flight
from .court import TennisCourtSpec
from .strike import (
    DEFAULT_RECOVERY_DURATIONS_S,
    JointTrajectoryBounds,
    QuinticJointTrajectory,
    StrikePlan,
    plan_ready_recovery_trajectory,
)
from .trajectory import SimulationJointMotionLimits


@dataclass(frozen=True)
class CourtBounceAudit:
    """Agreement between analytical and MuJoCo post-bounce states."""

    analytical_time_s: float
    mujoco_separation_time_s: float
    analytical_position_m: np.ndarray
    mujoco_separation_position_m: np.ndarray
    analytical_velocity_m_s: np.ndarray
    mujoco_separation_velocity_m_s: np.ndarray
    time_error_s: float
    position_error_m: float
    velocity_error_m_s: float
    passed: bool

    def metrics(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in tuple(result.items()):
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
        return result


@dataclass(frozen=True)
class StrikeExecutionConfig:
    """Controller and acceptance bounds for the canonical M2 execution."""

    reference_rate_hz: float = 250.0
    inner_control_rate_hz: float = 1000.0
    position_feedback_nm_rad: float = 800.0
    velocity_feedback_nm_s_rad: float = 80.0
    maximum_post_contact_time_s: float = 0.05
    maximum_contact_time_error_s: float = 0.010
    maximum_contact_position_error_m: float = 0.100
    maximum_joint_tracking_error_rad: float = 0.015
    maximum_contact_joint_velocity_error_rad_s: float = 0.080
    maximum_actual_joint_speed_rad_s: float = 4.0
    maximum_actual_joint_acceleration_rad_s2: float = 15.0
    minimum_actual_joint_limit_margin_rad: float = 0.020
    minimum_measured_return_net_clearance_m: float = 0.10
    maximum_outgoing_velocity_error_m_s: float = 2.00
    recovery_duration_candidates_s: tuple[float, ...] = DEFAULT_RECOVERY_DURATIONS_S
    minimum_planned_recovery_joint_limit_margin_rad: float = 0.020
    maximum_recovery_joint_tracking_error_rad: float = 0.015
    maximum_recovery_final_joint_error_rad: float = 0.010

    def validate(self, physics_timestep_s: float) -> None:
        positive = {
            "reference_rate_hz": self.reference_rate_hz,
            "inner_control_rate_hz": self.inner_control_rate_hz,
            "position_feedback_nm_rad": self.position_feedback_nm_rad,
            "velocity_feedback_nm_s_rad": self.velocity_feedback_nm_s_rad,
            "maximum_post_contact_time_s": self.maximum_post_contact_time_s,
        }
        if any(value <= 0.0 for value in positive.values()):
            raise ValueError("controller rates, gains, and horizon must be positive")
        if not self.recovery_duration_candidates_s or any(
            duration <= 0.0 for duration in self.recovery_duration_candidates_s
        ):
            raise ValueError("recovery duration candidates must be positive")
        if any(
            value < 0.0
            for value in (
                self.maximum_contact_time_error_s,
                self.maximum_contact_position_error_m,
                self.maximum_joint_tracking_error_rad,
                self.maximum_contact_joint_velocity_error_rad_s,
                self.minimum_actual_joint_limit_margin_rad,
                self.minimum_measured_return_net_clearance_m,
                self.maximum_outgoing_velocity_error_m_s,
                self.minimum_planned_recovery_joint_limit_margin_rad,
                self.maximum_recovery_joint_tracking_error_rad,
                self.maximum_recovery_final_joint_error_rad,
            )
        ):
            raise ValueError("execution acceptance bounds cannot be negative")
        if not np.isclose(
            physics_timestep_s * self.inner_control_rate_hz,
            1.0,
            atol=1e-12,
        ):
            raise ValueError("inner control rate must equal the physics rate")
        rate_ratio = self.inner_control_rate_hz / self.reference_rate_hz
        if not np.isclose(rate_ratio, round(rate_ratio), atol=1e-12):
            raise ValueError("reference rate must divide the inner control rate")


@dataclass(frozen=True)
class StrikeRecoveryResult:
    """Planned and measured return from ball separation to the ready pose."""

    planned: bool
    duration_s: float | None
    planned_maximum_joint_speed_rad_s: float | None
    planned_maximum_joint_acceleration_rad_s2: float | None
    planned_minimum_joint_limit_margin_rad: float | None
    maximum_joint_tracking_error_rad: float | None
    final_joint_error_rad: float | None
    maximum_actual_joint_speed_rad_s: float | None
    maximum_actual_joint_acceleration_rad_s2: float | None
    maximum_applied_torque_nm: float | None
    minimum_actual_joint_limit_margin_rad: float | None
    clipped_control_steps: int
    unexpected_contact_steps: int
    passed: bool
    failure_reasons: tuple[str, ...]

    def metrics(self) -> dict[str, Any]:
        result = asdict(self)
        result["failure_reasons"] = list(self.failure_reasons)
        return result


@dataclass(frozen=True)
class StrikeExecutionResult:
    """Measured tracking, contact, and outgoing-ball state for one strike."""

    contacted: bool
    separated: bool
    planned_contact_time_s: float
    actual_contact_time_s: float | None
    separation_time_s: float | None
    actual_contact_ball_position_m: np.ndarray | None
    incoming_ball_velocity_m_s: np.ndarray | None
    actual_contact_racket_velocity_m_s: np.ndarray | None
    outgoing_ball_position_m: np.ndarray | None
    outgoing_ball_velocity_m_s: np.ndarray | None
    outgoing_velocity_error_m_s: float | None
    contact_time_error_s: float | None
    contact_position_error_m: float | None
    maximum_joint_tracking_error_rad: float
    contact_joint_position_error_rad: float | None
    contact_joint_velocity_error_rad_s: float | None
    maximum_actual_joint_speed_rad_s: float
    maximum_pre_contact_joint_acceleration_rad_s2: float
    maximum_contact_phase_joint_acceleration_rad_s2: float
    maximum_applied_torque_nm: float
    minimum_actual_joint_limit_margin_rad: float
    clipped_control_steps: int
    unexpected_contact_steps: int
    measured_return: BallFlightResult | None
    recovery: StrikeRecoveryResult | None
    passed: bool
    failure_reasons: tuple[str, ...]

    def metrics(self) -> dict[str, Any]:
        return {
            "contacted": self.contacted,
            "separated": self.separated,
            "planned_contact_time_s": self.planned_contact_time_s,
            "actual_contact_time_s": self.actual_contact_time_s,
            "separation_time_s": self.separation_time_s,
            "actual_contact_ball_position_m": _optional_list(
                self.actual_contact_ball_position_m
            ),
            "incoming_ball_velocity_m_s": _optional_list(
                self.incoming_ball_velocity_m_s
            ),
            "actual_contact_racket_velocity_m_s": _optional_list(
                self.actual_contact_racket_velocity_m_s
            ),
            "outgoing_ball_position_m": _optional_list(
                self.outgoing_ball_position_m
            ),
            "outgoing_ball_velocity_m_s": _optional_list(
                self.outgoing_ball_velocity_m_s
            ),
            "outgoing_velocity_error_m_s": self.outgoing_velocity_error_m_s,
            "contact_time_error_s": self.contact_time_error_s,
            "contact_position_error_m": self.contact_position_error_m,
            "maximum_joint_tracking_error_rad": (
                self.maximum_joint_tracking_error_rad
            ),
            "contact_joint_position_error_rad": (
                self.contact_joint_position_error_rad
            ),
            "contact_joint_velocity_error_rad_s": (
                self.contact_joint_velocity_error_rad_s
            ),
            "maximum_actual_joint_speed_rad_s": (
                self.maximum_actual_joint_speed_rad_s
            ),
            "maximum_pre_contact_joint_acceleration_rad_s2": (
                self.maximum_pre_contact_joint_acceleration_rad_s2
            ),
            "maximum_contact_phase_joint_acceleration_rad_s2": (
                self.maximum_contact_phase_joint_acceleration_rad_s2
            ),
            "maximum_applied_torque_nm": self.maximum_applied_torque_nm,
            "minimum_actual_joint_limit_margin_rad": (
                self.minimum_actual_joint_limit_margin_rad
            ),
            "clipped_control_steps": self.clipped_control_steps,
            "unexpected_contact_steps": self.unexpected_contact_steps,
            "measured_return": (
                None if self.measured_return is None else self.measured_return.metrics()
            ),
            "recovery": None if self.recovery is None else self.recovery.metrics(),
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }


def _optional_list(value: np.ndarray | None) -> list[float] | None:
    return None if value is None else value.tolist()


def _ball_addresses(model: mujoco.MjModel) -> tuple[int, int]:
    joint = model.joint("ball_free")
    return int(joint.qposadr[0]), int(joint.dofadr[0])


def apply_ball_drag(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: BallFlightConfig | None = None,
) -> None:
    """Write the analytical quadratic-drag force to the ball free joint."""
    config = config or BallFlightConfig()
    _, dof_address = _ball_addresses(model)
    velocity = data.qvel[dof_address : dof_address + 3]
    drag_scale = (
        0.5
        * config.air_density_kg_m3
        * config.drag_coefficient
        * np.pi
        * config.radius_m**2
    )
    data.qfrc_applied[dof_address : dof_address + 3] = (
        -drag_scale * np.linalg.norm(velocity) * velocity
    )


def simulate_mujoco_ball_flight(
    model: mujoco.MjModel,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    *,
    config: BallFlightConfig | None = None,
) -> BallFlightResult:
    """Roll out the calibrated MuJoCo ball with robot collisions isolated.

    Collision masks are restored before return, so the same model can then be
    used for strike planning and execution.  This privileged M2 predictor uses
    the simulator's compliant court bounce instead of treating the independent
    analytical flight model as exact after impact with the court.
    """
    config = config or BallFlightConfig()
    if not np.isclose(model.opt.timestep, config.dt_s, atol=1e-12):
        raise ValueError("flight configuration timestep must match MuJoCo")
    court = TennisCourtSpec()
    data = mujoco.MjData(model)
    qpos_address, dof_address = _ball_addresses(model)
    ready = tennis_ready_configuration(model)
    data.qpos[:7] = ready
    data.ctrl[:7] = ready
    initial_position = np.asarray(position_m, dtype=np.float64).reshape(3)
    initial_velocity = np.asarray(velocity_m_s, dtype=np.float64).reshape(3)
    data.qpos[qpos_address : qpos_address + 7] = [
        *initial_position,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    data.qvel[dof_address : dof_address + 3] = initial_velocity

    ball_geom = model.geom("tennis_ball_geom").id
    court_geom = model.geom("tennis_court").id
    net_geom = model.geom("tennis_net").id
    collision_geoms = {ball_geom, court_geom, net_geom}
    saved_contype = model.geom_contype.copy()
    saved_conaffinity = model.geom_conaffinity.copy()
    saved_pair_signature = model.pair_signature.copy()
    for geom_id in range(model.ngeom):
        if geom_id not in collision_geoms:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
    for pair_id in range(model.npair):
        if model.pair(pair_id).name == "ball_racket_contact":
            model.pair_signature[pair_id] = -1

    steps = int(round(config.duration_s / config.dt_s))
    times = np.arange(steps + 1, dtype=np.float64) * config.dt_s
    positions = np.empty((steps + 1, 3), dtype=np.float64)
    velocities = np.empty((steps + 1, 3), dtype=np.float64)
    positions[0] = initial_position
    velocities[0] = initial_velocity
    hit_net = False
    try:
        mujoco.mj_forward(model, data)
        for step in range(1, steps + 1):
            apply_ball_drag(model, data, config)
            mujoco.mj_step(model, data)
            positions[step] = data.qpos[
                qpos_address : qpos_address + 3
            ]
            velocities[step] = data.qvel[
                dof_address : dof_address + 3
            ]
            hit_net = hit_net or _geoms_touching(data, ball_geom, net_geom)
    finally:
        model.geom_contype[:] = saved_contype
        model.geom_conaffinity[:] = saved_conaffinity
        model.pair_signature[:] = saved_pair_signature

    crossing_indices = np.flatnonzero(
        positions[:-1, 0] * positions[1:, 0] <= 0.0
    )
    net_crossing: np.ndarray | None = None
    net_clearance: float | None = None
    if len(crossing_indices):
        index = int(crossing_indices[0])
        delta_x = positions[index + 1, 0] - positions[index, 0]
        fraction = (
            0.0
            if abs(delta_x) < 1e-12
            else -positions[index, 0] / delta_x
        )
        net_crossing = positions[index] + fraction * (
            positions[index + 1] - positions[index]
        )
        if abs(net_crossing[1]) <= court.doubles_half_width_m:
            net_clearance = float(
                net_crossing[2]
                - config.radius_m
                - court.net_height_m(float(net_crossing[1]))
            )

    bounce_indices = np.flatnonzero(
        (velocities[:-1, 2] < 0.0)
        & (velocities[1:, 2] > 0.0)
        & (positions[1:, 2] < 0.10)
    )
    first_bounce = (
        None
        if not len(bounce_indices)
        else positions[int(bounce_indices[0] + 1)].copy()
    )
    initial_side = 1.0 if initial_position[0] >= 0.0 else -1.0
    legal_first_bounce = bool(
        not hit_net
        and first_bounce is not None
        and first_bounce[0] * initial_side <= 0.0
        and court.contains_singles_bounce(first_bounce, config.radius_m)
    )
    if hit_net:
        outcome = "hit_net"
    elif first_bounce is None:
        outcome = "airborne"
    elif first_bounce[0] * initial_side > 0.0:
        outcome = "same_side_bounce"
    elif legal_first_bounce:
        outcome = "legal_first_bounce"
    else:
        outcome = "out"
    return BallFlightResult(
        times_s=times,
        positions_m=positions,
        velocities_m_s=velocities,
        net_crossing_m=net_crossing,
        first_bounce_m=first_bounce,
        bounce_count=int(len(bounce_indices)),
        hit_net=hit_net,
        net_clearance_m=net_clearance,
        legal_first_bounce=legal_first_bounce,
        outcome=outcome,
    )


def audit_court_bounce(
    model: mujoco.MjModel,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    *,
    flight_config: BallFlightConfig | None = None,
    maximum_time_error_s: float = 0.005,
    maximum_position_error_m: float = 0.050,
    maximum_velocity_error_m_s: float = 0.250,
) -> CourtBounceAudit:
    """Compare the first MuJoCo bounce separation with the analytical model."""
    config = flight_config or BallFlightConfig()
    expected = simulate_ball_flight(position_m, velocity_m_s, config)
    bounce_indices = np.flatnonzero(
        (expected.velocities_m_s[:-1, 2] < 0.0)
        & (expected.velocities_m_s[1:, 2] > 0.0)
    )
    if not len(bounce_indices):
        raise ValueError("analytical flight has no bounce in its duration")
    expected_index = int(bounce_indices[0] + 1)

    data = mujoco.MjData(model)
    qpos_address, dof_address = _ball_addresses(model)
    ready = tennis_ready_configuration(model)
    data.qpos[:7] = ready
    data.ctrl[:7] = ready
    data.qpos[qpos_address : qpos_address + 7] = [
        *np.asarray(position_m, dtype=np.float64).reshape(3),
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    data.qvel[dof_address : dof_address + 3] = np.asarray(
        velocity_m_s, dtype=np.float64
    ).reshape(3)
    mujoco.mj_forward(model, data)
    ball_geom = model.geom("tennis_ball_geom").id
    court_geom = model.geom("tennis_court").id
    touched = False
    actual_time: float | None = None
    actual_position: np.ndarray | None = None
    actual_velocity: np.ndarray | None = None
    maximum_steps = int(np.ceil(config.duration_s / model.opt.timestep))
    for _ in range(maximum_steps):
        apply_ball_drag(model, data, config)
        mujoco.mj_step(model, data)
        touching = _geoms_touching(data, ball_geom, court_geom)
        if touching:
            touched = True
        elif touched:
            actual_time = float(data.time)
            actual_position = data.qpos[qpos_address : qpos_address + 3].copy()
            actual_velocity = data.qvel[dof_address : dof_address + 3].copy()
            break
    if actual_time is None or actual_position is None or actual_velocity is None:
        raise RuntimeError("MuJoCo ball did not separate after its first court bounce")

    expected_time = float(expected.times_s[expected_index])
    expected_position = expected.positions_m[expected_index].copy()
    expected_velocity = expected.velocities_m_s[expected_index].copy()
    time_error = abs(actual_time - expected_time)
    position_error = float(np.linalg.norm(actual_position - expected_position))
    velocity_error = float(np.linalg.norm(actual_velocity - expected_velocity))
    return CourtBounceAudit(
        analytical_time_s=expected_time,
        mujoco_separation_time_s=actual_time,
        analytical_position_m=expected_position,
        mujoco_separation_position_m=actual_position,
        analytical_velocity_m_s=expected_velocity,
        mujoco_separation_velocity_m_s=actual_velocity,
        time_error_s=time_error,
        position_error_m=position_error,
        velocity_error_m_s=velocity_error,
        passed=(
            time_error <= maximum_time_error_s
            and position_error <= maximum_position_error_m
            and velocity_error <= maximum_velocity_error_m_s
        ),
    )


def _geoms_touching(data: mujoco.MjData, first: int, second: int) -> bool:
    pair = {first, second}
    return any(
        {data.contact[index].geom1, data.contact[index].geom2} == pair
        for index in range(data.ncon)
    )


def _interpolated_reference(
    plan: StrikePlan,
    elapsed_s: float,
    period_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate between adjacent 250 Hz trajectory references."""

    def sample(time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if time_s <= plan.trajectory.duration_s:
            return plan.trajectory.sample(time_s)
        post_contact = time_s - plan.trajectory.duration_s
        return (
            plan.candidate.solution.joint_positions_rad
            + plan.contact_joint_velocities_rad_s * post_contact,
            plan.contact_joint_velocities_rad_s.copy(),
            np.zeros(7, dtype=np.float64),
        )

    lower_time = np.floor(elapsed_s / period_s) * period_s
    upper_time = lower_time + period_s
    fraction = (elapsed_s - lower_time) / period_s
    lower = sample(lower_time)
    upper = sample(upper_time)
    return tuple(
        (1.0 - fraction) * lower[index] + fraction * upper[index]
        for index in range(3)
    )


def _racket_linear_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, site_id: int
) -> np.ndarray:
    jacobian = np.zeros((3, model.nv), dtype=np.float64)
    rotation = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacobian, rotation, site_id)
    return jacobian @ data.qvel


def _interpolated_trajectory_reference(
    trajectory: QuinticJointTrajectory,
    elapsed_s: float,
    period_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a joint trajectory between adjacent servo references."""
    lower_time = min(
        np.floor(elapsed_s / period_s) * period_s,
        trajectory.duration_s,
    )
    upper_time = min(lower_time + period_s, trajectory.duration_s)
    if upper_time <= lower_time:
        return trajectory.sample(trajectory.duration_s)
    fraction = (elapsed_s - lower_time) / (upper_time - lower_time)
    lower = trajectory.sample(float(lower_time))
    upper = trajectory.sample(float(upper_time))
    return tuple(
        (1.0 - fraction) * lower[index] + fraction * upper[index]
        for index in range(3)
    )


def _plan_recovery_trajectory(
    model: mujoco.MjModel,
    start_position_rad: np.ndarray,
    start_velocity_rad_s: np.ndarray,
    config: StrikeExecutionConfig,
) -> tuple[QuinticJointTrajectory, JointTrajectoryBounds] | None:
    """Find the shortest screened path from the measured state to ready."""
    return plan_ready_recovery_trajectory(
        model,
        start_position_rad,
        start_velocity_rad_s,
        duration_candidates_s=config.recovery_duration_candidates_s,
        limits=SimulationJointMotionLimits(
            maximum_speed_rad_s=config.maximum_actual_joint_speed_rad_s,
            maximum_acceleration_rad_s2=(
                config.maximum_actual_joint_acceleration_rad_s2
            ),
            minimum_joint_limit_margin_rad=(
                config.minimum_planned_recovery_joint_limit_margin_rad
            ),
        ),
    )


def _execute_recovery(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    inverse_data: mujoco.MjData,
    *,
    config: StrikeExecutionConfig,
    flight_config: BallFlightConfig,
) -> StrikeRecoveryResult:
    """Plan and execute a bounded recovery on the live post-strike state."""
    planned = _plan_recovery_trajectory(
        model,
        data.qpos[:7].copy(),
        data.qvel[:7].copy(),
        config,
    )
    if planned is None:
        return StrikeRecoveryResult(
            planned=False,
            duration_s=None,
            planned_maximum_joint_speed_rad_s=None,
            planned_maximum_joint_acceleration_rad_s2=None,
            planned_minimum_joint_limit_margin_rad=None,
            maximum_joint_tracking_error_rad=None,
            final_joint_error_rad=None,
            maximum_actual_joint_speed_rad_s=None,
            maximum_actual_joint_acceleration_rad_s2=None,
            maximum_applied_torque_nm=None,
            minimum_actual_joint_limit_margin_rad=None,
            clipped_control_steps=0,
            unexpected_contact_steps=0,
            passed=False,
            failure_reasons=("no_feasible_plan",),
        )

    trajectory, bounds = planned
    physics_dt = float(model.opt.timestep)
    reference_period = 1.0 / config.reference_rate_hz
    actuator_gain = model.actuator_gainprm[:7, 0]
    actuator_damping = -model.actuator_biasprm[:7, 2]
    control_lower = model.actuator_ctrlrange[:7, 0]
    control_upper = model.actuator_ctrlrange[:7, 1]
    joint_lower = model.jnt_range[:7, 0]
    joint_upper = model.jnt_range[:7, 1]
    inverse_torque_all = np.zeros(model.nv, dtype=np.float64)
    ball_geom = model.geom("tennis_ball_geom").id
    court_geom = model.geom("tennis_court").id
    net_geom = model.geom("tennis_net").id
    allowed_pairs = ({ball_geom, court_geom}, {ball_geom, net_geom})

    maximum_tracking_error = 0.0
    maximum_speed = 0.0
    maximum_acceleration = 0.0
    maximum_torque = 0.0
    minimum_margin = float("inf")
    clipped_steps = 0
    unexpected_contact_steps = 0
    maximum_steps = int(np.ceil(trajectory.duration_s / physics_dt))

    for step in range(maximum_steps):
        elapsed_s = min(step * physics_dt, trajectory.duration_s)
        desired_position, desired_velocity, desired_acceleration = (
            _interpolated_trajectory_reference(
                trajectory,
                elapsed_s,
                reference_period,
            )
        )
        inverse_data.qpos[:7] = desired_position
        inverse_data.qvel[:7] = desired_velocity
        mujoco.mj_forward(model, inverse_data)
        inverse_data.qacc[:] = 0.0
        inverse_data.qacc[:7] = desired_acceleration
        mujoco.mj_rne(model, inverse_data, 1, inverse_torque_all)
        applied_torque = (
            inverse_torque_all[:7]
            - inverse_data.qfrc_passive[:7]
            + config.position_feedback_nm_rad
            * (desired_position - data.qpos[:7])
            + config.velocity_feedback_nm_s_rad
            * (desired_velocity - data.qvel[:7])
        )
        position_command = desired_position + (
            actuator_damping * desired_velocity / actuator_gain
        )
        clipped_command = np.clip(position_command, control_lower, control_upper)
        if not np.array_equal(clipped_command, position_command):
            clipped_steps += 1
        data.ctrl[:7] = clipped_command
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[:7] = applied_torque
        apply_ball_drag(model, data, flight_config)
        mujoco.mj_step(model, data)

        expected_position, _, _ = _interpolated_trajectory_reference(
            trajectory,
            min((step + 1) * physics_dt, trajectory.duration_s),
            reference_period,
        )
        maximum_tracking_error = max(
            maximum_tracking_error,
            float(np.max(np.abs(data.qpos[:7] - expected_position))),
        )
        maximum_speed = max(maximum_speed, float(np.max(np.abs(data.qvel[:7]))))
        maximum_acceleration = max(
            maximum_acceleration,
            float(np.max(np.abs(data.qacc[:7]))),
        )
        maximum_torque = max(maximum_torque, float(np.max(np.abs(applied_torque))))
        minimum_margin = min(
            minimum_margin,
            float(
                np.min(
                    np.minimum(
                        data.qpos[:7] - joint_lower,
                        joint_upper - data.qpos[:7],
                    )
                )
            ),
        )
        if any(
            {data.contact[index].geom1, data.contact[index].geom2}
            not in allowed_pairs
            for index in range(data.ncon)
        ):
            unexpected_contact_steps += 1

    final_error = float(
        np.max(np.abs(data.qpos[:7] - tennis_ready_configuration(model)))
    )
    failures = []
    if maximum_tracking_error > config.maximum_recovery_joint_tracking_error_rad:
        failures.append("joint_tracking")
    if final_error > config.maximum_recovery_final_joint_error_rad:
        failures.append("final_joint_error")
    if maximum_speed > config.maximum_actual_joint_speed_rad_s:
        failures.append("actual_joint_speed")
    if maximum_acceleration > config.maximum_actual_joint_acceleration_rad_s2:
        failures.append("actual_joint_acceleration")
    if minimum_margin < config.minimum_actual_joint_limit_margin_rad:
        failures.append("joint_limit_margin")
    if clipped_steps:
        failures.append("clipped_control")
    if unexpected_contact_steps:
        failures.append("unexpected_contact")
    return StrikeRecoveryResult(
        planned=True,
        duration_s=trajectory.duration_s,
        planned_maximum_joint_speed_rad_s=bounds.maximum_joint_speed_rad_s,
        planned_maximum_joint_acceleration_rad_s2=(
            bounds.maximum_joint_acceleration_rad_s2
        ),
        planned_minimum_joint_limit_margin_rad=(
            bounds.minimum_joint_limit_margin_rad
        ),
        maximum_joint_tracking_error_rad=maximum_tracking_error,
        final_joint_error_rad=final_error,
        maximum_actual_joint_speed_rad_s=maximum_speed,
        maximum_actual_joint_acceleration_rad_s2=maximum_acceleration,
        maximum_applied_torque_nm=maximum_torque,
        minimum_actual_joint_limit_margin_rad=minimum_margin,
        clipped_control_steps=clipped_steps,
        unexpected_contact_steps=unexpected_contact_steps,
        passed=not failures,
        failure_reasons=tuple(failures),
    )


def execute_strike(
    model: mujoco.MjModel,
    plan: StrikePlan,
    initial_ball_position_m: np.ndarray,
    initial_ball_velocity_m_s: np.ndarray,
    *,
    config: StrikeExecutionConfig | None = None,
    flight_config: BallFlightConfig | None = None,
) -> StrikeExecutionResult:
    """Track one planned strike through contact and a bounded recovery.

    The trajectory generator emits 250 Hz references.  A 1 kHz inner loop
    interpolates them and applies rigid-body inverse dynamics plus feedback.
    After measuring ball separation, a separately screened trajectory returns
    the arm from its measured joint state to the ready pose.
    """
    config = config or StrikeExecutionConfig()
    flight_config = flight_config or BallFlightConfig()
    physics_dt = float(model.opt.timestep)
    config.validate(physics_dt)
    reference_period = 1.0 / config.reference_rate_hz

    data = mujoco.MjData(model)
    inverse_data = mujoco.MjData(model)
    qpos_address, dof_address = _ball_addresses(model)
    start_position, _, _ = plan.trajectory.sample(0.0)
    data.qpos[:7] = start_position
    data.ctrl[:7] = start_position
    data.qpos[qpos_address : qpos_address + 7] = [
        *np.asarray(initial_ball_position_m, dtype=np.float64).reshape(3),
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    data.qvel[dof_address : dof_address + 3] = np.asarray(
        initial_ball_velocity_m_s, dtype=np.float64
    ).reshape(3)
    inverse_data.qpos[qpos_address : qpos_address + 7] = [
        10.0,
        0.0,
        4.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    mujoco.mj_forward(model, data)

    actuator_gain = model.actuator_gainprm[:7, 0]
    actuator_damping = -model.actuator_biasprm[:7, 2]
    control_lower = model.actuator_ctrlrange[:7, 0]
    control_upper = model.actuator_ctrlrange[:7, 1]
    joint_lower = model.jnt_range[:7, 0]
    joint_upper = model.jnt_range[:7, 1]
    inverse_torque_all = np.zeros(model.nv, dtype=np.float64)
    ball_geom = model.geom("tennis_ball_geom").id
    racket_geom = model.geom("racket_head").id
    court_geom = model.geom("tennis_court").id
    site_id = model.site("racket_center").id

    actual_contact_time: float | None = None
    separation_time: float | None = None
    contact_ball_position: np.ndarray | None = None
    incoming_ball_velocity: np.ndarray | None = None
    contact_racket_velocity: np.ndarray | None = None
    outgoing_ball_position: np.ndarray | None = None
    outgoing_ball_velocity: np.ndarray | None = None
    contact_joint_position_error: float | None = None
    contact_joint_velocity_error: float | None = None
    maximum_tracking_error = 0.0
    maximum_speed = 0.0
    maximum_pre_contact_acceleration = 0.0
    maximum_contact_phase_acceleration = 0.0
    maximum_torque = 0.0
    minimum_margin = float("inf")
    clipped_steps = 0
    unexpected_contact_steps = 0
    maximum_time = plan.candidate.time_s + config.maximum_post_contact_time_s
    maximum_steps = int(np.ceil(maximum_time / physics_dt)) + 1

    for step in range(maximum_steps):
        elapsed = step * physics_dt
        desired_position, desired_velocity, desired_acceleration = (
            _interpolated_reference(plan, elapsed, reference_period)
        )
        inverse_data.qpos[:7] = desired_position
        inverse_data.qvel[:7] = desired_velocity
        mujoco.mj_forward(model, inverse_data)
        inverse_data.qacc[:] = 0.0
        inverse_data.qacc[:7] = desired_acceleration
        mujoco.mj_rne(model, inverse_data, 1, inverse_torque_all)
        applied_torque = (
            inverse_torque_all[:7]
            - inverse_data.qfrc_passive[:7]
            + config.position_feedback_nm_rad
            * (desired_position - data.qpos[:7])
            + config.velocity_feedback_nm_s_rad
            * (desired_velocity - data.qvel[:7])
        )
        position_command = desired_position + (
            actuator_damping * desired_velocity / actuator_gain
        )
        clipped_command = np.clip(position_command, control_lower, control_upper)
        if not np.array_equal(clipped_command, position_command):
            clipped_steps += 1
        data.ctrl[:7] = clipped_command
        data.qfrc_applied[:7] = applied_torque
        apply_ball_drag(model, data, flight_config)
        pre_step_ball_velocity = data.qvel[
            dof_address : dof_address + 3
        ].copy()
        mujoco.mj_step(model, data)

        expected_position, _, _ = _interpolated_reference(
            plan, float(data.time), reference_period
        )
        maximum_tracking_error = max(
            maximum_tracking_error,
            float(np.max(np.abs(data.qpos[:7] - expected_position))),
        )
        maximum_speed = max(maximum_speed, float(np.max(np.abs(data.qvel[:7]))))
        maximum_torque = max(maximum_torque, float(np.max(np.abs(applied_torque))))
        minimum_margin = min(
            minimum_margin,
            float(
                np.min(
                    np.minimum(
                        data.qpos[:7] - joint_lower,
                        joint_upper - data.qpos[:7],
                    )
                )
            ),
        )

        touching_racket = _geoms_touching(data, ball_geom, racket_geom)
        actual_acceleration = float(np.max(np.abs(data.qacc[:7])))
        if actual_contact_time is None and not touching_racket:
            maximum_pre_contact_acceleration = max(
                maximum_pre_contact_acceleration, actual_acceleration
            )
        else:
            maximum_contact_phase_acceleration = max(
                maximum_contact_phase_acceleration, actual_acceleration
            )
        allowed_pairs = (
            {ball_geom, racket_geom},
            {ball_geom, court_geom},
        )
        if any(
            {data.contact[index].geom1, data.contact[index].geom2}
            not in allowed_pairs
            for index in range(data.ncon)
        ):
            unexpected_contact_steps += 1

        if touching_racket and actual_contact_time is None:
            actual_contact_time = float(data.time)
            contact_ball_position = data.qpos[
                qpos_address : qpos_address + 3
            ].copy()
            incoming_ball_velocity = pre_step_ball_velocity
            contact_racket_velocity = _racket_linear_velocity(model, data, site_id)
            contact_joint_position_error = float(
                np.max(
                    np.abs(
                        data.qpos[:7]
                        - plan.candidate.solution.joint_positions_rad
                    )
                )
            )
            contact_joint_velocity_error = float(
                np.max(
                    np.abs(
                        data.qvel[:7] - plan.contact_joint_velocities_rad_s
                    )
                )
            )
        elif not touching_racket and actual_contact_time is not None:
            separation_time = float(data.time)
            outgoing_ball_position = data.qpos[
                qpos_address : qpos_address + 3
            ].copy()
            outgoing_ball_velocity = data.qvel[
                dof_address : dof_address + 3
            ].copy()
            break

    contact_time_error = (
        None
        if actual_contact_time is None
        else abs(actual_contact_time - plan.candidate.time_s)
    )
    contact_position_error = (
        None
        if contact_ball_position is None
        else float(
            np.linalg.norm(contact_ball_position - plan.candidate.ball_position_m)
        )
    )
    outgoing_velocity_error = (
        None
        if outgoing_ball_velocity is None
        else float(
            np.linalg.norm(
                outgoing_ball_velocity - plan.outgoing_ball_velocity_m_s
            )
        )
    )
    measured_return = (
        None
        if outgoing_ball_position is None or outgoing_ball_velocity is None
        else simulate_ball_flight(
            outgoing_ball_position,
            outgoing_ball_velocity,
            replace(
                flight_config,
                duration_s=max(4.0, flight_config.duration_s),
            ),
        )
    )
    recovery = (
        None
        if separation_time is None
        else _execute_recovery(
            model,
            data,
            inverse_data,
            config=config,
            flight_config=flight_config,
        )
    )

    failures = []
    if actual_contact_time is None:
        failures.append("no_ball_racket_contact")
    if separation_time is None:
        failures.append("no_ball_racket_separation")
    if (
        contact_time_error is not None
        and contact_time_error > config.maximum_contact_time_error_s
    ):
        failures.append("contact_time")
    if (
        contact_position_error is not None
        and contact_position_error > config.maximum_contact_position_error_m
    ):
        failures.append("contact_position")
    if maximum_tracking_error > config.maximum_joint_tracking_error_rad:
        failures.append("joint_tracking")
    if (
        contact_joint_velocity_error is not None
        and contact_joint_velocity_error
        > config.maximum_contact_joint_velocity_error_rad_s
    ):
        failures.append("contact_joint_velocity")
    if maximum_speed > config.maximum_actual_joint_speed_rad_s:
        failures.append("actual_joint_speed")
    if (
        maximum_pre_contact_acceleration
        > config.maximum_actual_joint_acceleration_rad_s2
    ):
        failures.append("actual_joint_acceleration")
    if minimum_margin < config.minimum_actual_joint_limit_margin_rad:
        failures.append("joint_limit_margin")
    if clipped_steps:
        failures.append("clipped_control")
    if unexpected_contact_steps:
        failures.append("unexpected_contact")
    if (
        outgoing_velocity_error is not None
        and outgoing_velocity_error > config.maximum_outgoing_velocity_error_m_s
    ):
        failures.append("outgoing_velocity_model_error")
    if measured_return is not None:
        if not measured_return.legal_first_bounce:
            failures.append("illegal_measured_return")
        elif (
            measured_return.net_clearance_m is None
            or measured_return.net_clearance_m
            < config.minimum_measured_return_net_clearance_m
        ):
            failures.append("measured_return_net_clearance")
    if recovery is not None and not recovery.passed:
        failures.extend(
            f"recovery_{reason}" for reason in recovery.failure_reasons
        )

    return StrikeExecutionResult(
        contacted=actual_contact_time is not None,
        separated=separation_time is not None,
        planned_contact_time_s=plan.candidate.time_s,
        actual_contact_time_s=actual_contact_time,
        separation_time_s=separation_time,
        actual_contact_ball_position_m=contact_ball_position,
        incoming_ball_velocity_m_s=incoming_ball_velocity,
        actual_contact_racket_velocity_m_s=contact_racket_velocity,
        outgoing_ball_position_m=outgoing_ball_position,
        outgoing_ball_velocity_m_s=outgoing_ball_velocity,
        outgoing_velocity_error_m_s=outgoing_velocity_error,
        contact_time_error_s=contact_time_error,
        contact_position_error_m=contact_position_error,
        maximum_joint_tracking_error_rad=maximum_tracking_error,
        contact_joint_position_error_rad=contact_joint_position_error,
        contact_joint_velocity_error_rad_s=contact_joint_velocity_error,
        maximum_actual_joint_speed_rad_s=maximum_speed,
        maximum_pre_contact_joint_acceleration_rad_s2=(
            maximum_pre_contact_acceleration
        ),
        maximum_contact_phase_joint_acceleration_rad_s2=(
            maximum_contact_phase_acceleration
        ),
        maximum_applied_torque_nm=maximum_torque,
        minimum_actual_joint_limit_margin_rad=minimum_margin,
        clipped_control_steps=clipped_steps,
        unexpected_contact_steps=unexpected_contact_steps,
        measured_return=measured_return,
        recovery=recovery,
        passed=not failures,
        failure_reasons=tuple(failures),
    )
