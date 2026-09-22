"""Integrated MuJoCo execution checks for privileged tennis strikes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from .arm import tennis_ready_configuration
from .ballistics import BallFlightConfig, BallFlightResult, simulate_ball_flight
from .strike import StrikePlan


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
    maximum_contact_position_error_m: float = 0.050
    maximum_joint_tracking_error_rad: float = 0.015
    maximum_contact_joint_velocity_error_rad_s: float = 0.080
    maximum_actual_joint_speed_rad_s: float = 4.0
    maximum_actual_joint_acceleration_rad_s2: float = 15.0
    minimum_actual_joint_limit_margin_rad: float = 0.020
    minimum_measured_return_net_clearance_m: float = 0.10
    maximum_outgoing_velocity_error_m_s: float = 1.50

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


def execute_strike(
    model: mujoco.MjModel,
    plan: StrikePlan,
    initial_ball_position_m: np.ndarray,
    initial_ball_velocity_m_s: np.ndarray,
    *,
    config: StrikeExecutionConfig | None = None,
    flight_config: BallFlightConfig | None = None,
) -> StrikeExecutionResult:
    """Track one planned strike through MuJoCo ball-racket separation.

    The trajectory generator emits 250 Hz references.  A 1 kHz inner loop
    interpolates them and applies rigid-body inverse dynamics plus feedback.
    The short post-contact continuation is only long enough to measure ball
    separation; a recovery or follow-through plan remains a later milestone.
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
            flight_config,
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
        passed=not failures,
        failure_reasons=tuple(failures),
    )
