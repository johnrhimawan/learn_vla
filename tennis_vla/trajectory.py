"""Joint-space arrival trajectories for privileged tennis intercepts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .arm import tennis_ready_configuration
from .ballistics import BallFlightResult
from .intercept import (
    InterceptCandidate,
    RacketIKConfig,
    find_kinematic_intercepts,
)


MINIMUM_JERK_PEAK_SPEED = 15.0 / 8.0
MINIMUM_JERK_PEAK_ACCELERATION = 10.0 / np.sqrt(3.0)


@dataclass(frozen=True)
class SimulationJointMotionLimits:
    """Project limits for simulation studies, not Sawyer hardware ratings."""

    maximum_speed_rad_s: float = 4.0
    maximum_acceleration_rad_s2: float = 15.0
    minimum_joint_limit_margin_rad: float = 0.03

    def validate(self) -> None:
        if self.maximum_speed_rad_s <= 0.0:
            raise ValueError("maximum_speed_rad_s must be positive")
        if self.maximum_acceleration_rad_s2 <= 0.0:
            raise ValueError("maximum_acceleration_rad_s2 must be positive")
        if self.minimum_joint_limit_margin_rad < 0.0:
            raise ValueError("minimum_joint_limit_margin_rad cannot be negative")


@dataclass(frozen=True)
class ArrivalTrackingConfig:
    """Acceptance thresholds for the simulated 250 Hz arrival controller."""

    servo_rate_hz: float = 250.0
    maximum_final_joint_error_rad: float = 0.01
    maximum_racket_position_tracking_error_m: float = 0.01
    maximum_racket_normal_tracking_error_deg: float = 1.0
    minimum_actual_joint_limit_margin_rad: float = 0.02
    maximum_actual_joint_speed_rad_s: float = 4.0
    maximum_actual_joint_acceleration_rad_s2: float = 15.0

    def validate(self) -> None:
        if self.servo_rate_hz <= 0.0:
            raise ValueError("servo_rate_hz must be positive")
        if self.maximum_final_joint_error_rad <= 0.0:
            raise ValueError("maximum_final_joint_error_rad must be positive")
        if self.maximum_racket_position_tracking_error_m <= 0.0:
            raise ValueError(
                "maximum_racket_position_tracking_error_m must be positive"
            )
        if self.maximum_racket_normal_tracking_error_deg <= 0.0:
            raise ValueError(
                "maximum_racket_normal_tracking_error_deg must be positive"
            )
        if self.minimum_actual_joint_limit_margin_rad < 0.0:
            raise ValueError(
                "minimum_actual_joint_limit_margin_rad cannot be negative"
            )
        if self.maximum_actual_joint_speed_rad_s <= 0.0:
            raise ValueError("maximum_actual_joint_speed_rad_s must be positive")
        if self.maximum_actual_joint_acceleration_rad_s2 <= 0.0:
            raise ValueError(
                "maximum_actual_joint_acceleration_rad_s2 must be positive"
            )


@dataclass(frozen=True)
class InterceptArrivalPlan:
    """A zero-velocity minimum-jerk arrival at one intercept pose."""

    candidate: InterceptCandidate
    planning_start_time_s: float
    duration_s: float
    start_joint_positions_rad: np.ndarray
    target_joint_positions_rad: np.ndarray
    peak_joint_speeds_rad_s: np.ndarray
    peak_joint_accelerations_rad_s2: np.ndarray
    minimum_joint_limit_margin_rad: float
    feasible: bool
    failure_reasons: tuple[str, ...]

    @property
    def maximum_joint_speed_rad_s(self) -> float:
        return float(np.max(self.peak_joint_speeds_rad_s))

    @property
    def maximum_joint_acceleration_rad_s2(self) -> float:
        return float(np.max(self.peak_joint_accelerations_rad_s2))

    def metrics(self, limits: SimulationJointMotionLimits) -> dict[str, Any]:
        return {
            "intercept": self.candidate.metrics(),
            "planning_start_time_s": self.planning_start_time_s,
            "duration_s": self.duration_s,
            "start_joint_positions_rad": self.start_joint_positions_rad.tolist(),
            "target_joint_positions_rad": self.target_joint_positions_rad.tolist(),
            "peak_joint_speeds_rad_s": self.peak_joint_speeds_rad_s.tolist(),
            "peak_joint_accelerations_rad_s2": (
                self.peak_joint_accelerations_rad_s2.tolist()
            ),
            "maximum_speed_ratio": (
                self.maximum_joint_speed_rad_s / limits.maximum_speed_rad_s
            ),
            "maximum_acceleration_ratio": (
                self.maximum_joint_acceleration_rad_s2
                / limits.maximum_acceleration_rad_s2
            ),
            "minimum_joint_limit_margin_rad": (
                self.minimum_joint_limit_margin_rad
            ),
            "feasible": self.feasible,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class ArrivalTrackingResult:
    """MuJoCo tracking metrics for one privileged arrival plan."""

    duration_s: float
    servo_steps: int
    physics_steps: int
    maximum_joint_tracking_error_rad: float
    p95_joint_tracking_error_rad: float
    final_joint_tracking_error_rad: float
    maximum_actual_joint_speed_rad_s: float
    maximum_actual_joint_acceleration_rad_s2: float
    maximum_inverse_dynamics_torque_nm: float
    minimum_actual_joint_limit_margin_rad: float
    clipped_servo_commands: int
    unexpected_contact_steps: int
    maximum_unexpected_penetration_m: float
    racket_position_tracking_error_m: float
    racket_normal_tracking_error_deg: float
    final_contact_position_error_m: float
    final_contact_normal_error_deg: float
    passed: bool
    failure_reasons: tuple[str, ...]

    def metrics(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "servo_steps": self.servo_steps,
            "physics_steps": self.physics_steps,
            "maximum_joint_tracking_error_rad": (
                self.maximum_joint_tracking_error_rad
            ),
            "p95_joint_tracking_error_rad": self.p95_joint_tracking_error_rad,
            "final_joint_tracking_error_rad": self.final_joint_tracking_error_rad,
            "maximum_actual_joint_speed_rad_s": (
                self.maximum_actual_joint_speed_rad_s
            ),
            "maximum_actual_joint_acceleration_rad_s2": (
                self.maximum_actual_joint_acceleration_rad_s2
            ),
            "maximum_inverse_dynamics_torque_nm": (
                self.maximum_inverse_dynamics_torque_nm
            ),
            "minimum_actual_joint_limit_margin_rad": (
                self.minimum_actual_joint_limit_margin_rad
            ),
            "clipped_servo_commands": self.clipped_servo_commands,
            "unexpected_contact_steps": self.unexpected_contact_steps,
            "maximum_unexpected_penetration_m": (
                self.maximum_unexpected_penetration_m
            ),
            "racket_position_tracking_error_m": (
                self.racket_position_tracking_error_m
            ),
            "racket_normal_tracking_error_deg": (
                self.racket_normal_tracking_error_deg
            ),
            "final_contact_position_error_m": self.final_contact_position_error_m,
            "final_contact_normal_error_deg": self.final_contact_normal_error_deg,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }


def sample_minimum_jerk(
    start: np.ndarray,
    target: np.ndarray,
    elapsed_s: float,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return position, velocity, and acceleration for a quintic trajectory."""
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    start_array = np.asarray(start, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if start_array.shape != target_array.shape:
        raise ValueError("start and target must have the same shape")
    phase = float(np.clip(elapsed_s / duration_s, 0.0, 1.0))
    delta = target_array - start_array
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    blend_rate = (
        30.0 * phase**2 - 60.0 * phase**3 + 30.0 * phase**4
    ) / duration_s
    blend_acceleration = (
        60.0 * phase - 180.0 * phase**2 + 120.0 * phase**3
    ) / duration_s**2
    return (
        start_array + blend * delta,
        blend_rate * delta,
        blend_acceleration * delta,
    )


def assess_intercept_arrival(
    model: mujoco.MjModel,
    candidate: InterceptCandidate,
    *,
    planning_start_time_s: float = 0.0,
    start_joint_positions_rad: np.ndarray | None = None,
    limits: SimulationJointMotionLimits | None = None,
) -> InterceptArrivalPlan:
    """Check an analytical minimum-jerk arrival against simulation limits."""
    limits = limits or SimulationJointMotionLimits()
    limits.validate()
    start = np.asarray(
        tennis_ready_configuration(model)
        if start_joint_positions_rad is None
        else start_joint_positions_rad,
        dtype=np.float64,
    ).reshape(7)
    target = candidate.solution.joint_positions_rad.copy()
    duration = candidate.time_s - planning_start_time_s
    if duration <= 0.0:
        raise ValueError("planning_start_time_s must precede the intercept")

    delta = np.abs(target - start)
    peak_speeds = MINIMUM_JERK_PEAK_SPEED * delta / duration
    peak_accelerations = (
        MINIMUM_JERK_PEAK_ACCELERATION * delta / duration**2
    )
    joint_lower = model.jnt_range[:7, 0]
    joint_upper = model.jnt_range[:7, 1]
    endpoint_margins = np.minimum(
        np.minimum(start - joint_lower, joint_upper - start),
        np.minimum(target - joint_lower, joint_upper - target),
    )
    minimum_margin = float(endpoint_margins.min())
    failures = []
    if np.any(peak_speeds > limits.maximum_speed_rad_s + 1e-12):
        failures.append("joint_speed")
    if np.any(
        peak_accelerations > limits.maximum_acceleration_rad_s2 + 1e-12
    ):
        failures.append("joint_acceleration")
    if minimum_margin + 1e-12 < limits.minimum_joint_limit_margin_rad:
        failures.append("joint_limit_margin")
    return InterceptArrivalPlan(
        candidate=candidate,
        planning_start_time_s=planning_start_time_s,
        duration_s=duration,
        start_joint_positions_rad=start,
        target_joint_positions_rad=target,
        peak_joint_speeds_rad_s=peak_speeds,
        peak_joint_accelerations_rad_s2=peak_accelerations,
        minimum_joint_limit_margin_rad=minimum_margin,
        feasible=not failures,
        failure_reasons=tuple(failures),
    )


def plan_intercept_arrivals(
    model: mujoco.MjModel,
    flight: BallFlightResult,
    *,
    planning_start_time_s: float = 0.0,
    start_joint_positions_rad: np.ndarray | None = None,
    limits: SimulationJointMotionLimits | None = None,
    ik_config: RacketIKConfig | None = None,
    maximum_candidates: int = 8,
    seed: int = 2026,
) -> list[InterceptArrivalPlan]:
    """Assess all sampled kinematic intercepts in chronological order."""
    limits = limits or SimulationJointMotionLimits()
    candidates = find_kinematic_intercepts(
        model,
        flight,
        maximum_candidates=maximum_candidates,
        ik_config=ik_config,
        seed=seed,
    )
    return [
        assess_intercept_arrival(
            model,
            candidate,
            planning_start_time_s=planning_start_time_s,
            start_joint_positions_rad=start_joint_positions_rad,
            limits=limits,
        )
        for candidate in candidates
        if candidate.time_s > planning_start_time_s
    ]


def earliest_feasible_arrival(
    plans: list[InterceptArrivalPlan],
) -> InterceptArrivalPlan | None:
    """Select the earliest feasible arrival from chronological plans."""
    return next((plan for plan in plans if plan.feasible), None)


def _normal_error_deg(current: np.ndarray, target: np.ndarray) -> float:
    cosine = float(np.clip(current @ target, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def track_intercept_arrival(
    model: mujoco.MjModel,
    plan: InterceptArrivalPlan,
    *,
    config: ArrivalTrackingConfig | None = None,
) -> ArrivalTrackingResult:
    """Track a plan with position servos and inverse-dynamics feedforward.

    The inverse-dynamics torque is privileged information used by the M2 oracle.
    A position command compensates the Menagerie actuator's velocity damping,
    while ``qfrc_applied`` supplies the modeled inertia, Coriolis, and gravity
    terms. Commands update at the configured servo rate and are held between
    MuJoCo physics steps.
    """
    if not plan.feasible:
        raise ValueError("cannot track an infeasible arrival plan")
    config = config or ArrivalTrackingConfig()
    config.validate()
    physics_dt = float(model.opt.timestep)
    servo_period = 1.0 / config.servo_rate_hz
    physics_steps_per_servo = int(round(servo_period / physics_dt))
    if not np.isclose(
        physics_steps_per_servo * physics_dt,
        servo_period,
        atol=1e-12,
    ):
        raise ValueError("servo period must be an integer multiple of physics dt")
    physics_steps = int(round(plan.duration_s / physics_dt))
    if not np.isclose(
        physics_steps * physics_dt,
        plan.duration_s,
        atol=1e-9,
    ):
        raise ValueError("arrival duration must be an integer multiple of physics dt")

    data = mujoco.MjData(model)
    inverse_data = mujoco.MjData(model)
    data.qpos[:7] = plan.start_joint_positions_rad
    data.ctrl[:7] = plan.start_joint_positions_rad
    ball_joint = model.joint("ball_free")
    ball_qpos_address = int(ball_joint.qposadr[0])
    parked_ball_qpos = np.array([10.0, 0.0, 4.0, 1.0, 0.0, 0.0, 0.0])
    data.qpos[ball_qpos_address : ball_qpos_address + 7] = parked_ball_qpos
    inverse_data.qpos[ball_qpos_address : ball_qpos_address + 7] = (
        parked_ball_qpos
    )
    mujoco.mj_forward(model, data)

    actuator_gain = model.actuator_gainprm[:7, 0]
    actuator_damping = -model.actuator_biasprm[:7, 2]
    control_lower = model.actuator_ctrlrange[:7, 0]
    control_upper = model.actuator_ctrlrange[:7, 1]
    joint_lower = model.jnt_range[:7, 0]
    joint_upper = model.jnt_range[:7, 1]
    tracking_errors = []
    maximum_actual_speed = 0.0
    maximum_actual_acceleration = 0.0
    maximum_inverse_torque = 0.0
    minimum_actual_margin = float("inf")
    clipped_commands = 0
    unexpected_contact_steps = 0
    maximum_unexpected_penetration = 0.0
    servo_steps = 0
    inverse_torque_all = np.zeros(model.nv, dtype=np.float64)
    ball_geom_id = model.geom("tennis_ball_geom").id

    for step in range(physics_steps):
        if step % physics_steps_per_servo == 0:
            command_time = step * physics_dt
            desired_position, desired_velocity, desired_acceleration = (
                sample_minimum_jerk(
                    plan.start_joint_positions_rad,
                    plan.target_joint_positions_rad,
                    command_time,
                    plan.duration_s,
                )
            )
            inverse_data.qpos[:7] = desired_position
            inverse_data.qvel[:7] = desired_velocity
            # Refresh kinematics, then use unconstrained rigid-body inverse
            # dynamics. Contact forces are audited separately and must not be
            # injected into the feedforward torque.
            mujoco.mj_forward(model, inverse_data)
            inverse_data.qacc[:] = 0.0
            inverse_data.qacc[:7] = desired_acceleration
            mujoco.mj_rne(model, inverse_data, 1, inverse_torque_all)
            inverse_torque = inverse_torque_all[:7]
            position_command = desired_position + (
                actuator_damping * desired_velocity / actuator_gain
            )
            clipped = np.clip(position_command, control_lower, control_upper)
            if not np.array_equal(clipped, position_command):
                clipped_commands += 1
            data.ctrl[:7] = clipped
            data.qfrc_applied[:7] = inverse_torque
            maximum_inverse_torque = max(
                maximum_inverse_torque,
                float(np.max(np.abs(inverse_torque))),
            )
            servo_steps += 1

        mujoco.mj_step(model, data)
        desired_position, _, _ = sample_minimum_jerk(
            plan.start_joint_positions_rad,
            plan.target_joint_positions_rad,
            float(data.time),
            plan.duration_s,
        )
        tracking_errors.append(
            float(np.max(np.abs(data.qpos[:7] - desired_position)))
        )
        maximum_actual_speed = max(
            maximum_actual_speed,
            float(np.max(np.abs(data.qvel[:7]))),
        )
        maximum_actual_acceleration = max(
            maximum_actual_acceleration,
            float(np.max(np.abs(data.qacc[:7]))),
        )
        actual_margins = np.minimum(
            data.qpos[:7] - joint_lower,
            joint_upper - data.qpos[:7],
        )
        minimum_actual_margin = min(
            minimum_actual_margin,
            float(actual_margins.min()),
        )
        unexpected_contacts = [
            data.contact[index]
            for index in range(data.ncon)
            if ball_geom_id
            not in (data.contact[index].geom1, data.contact[index].geom2)
        ]
        if unexpected_contacts:
            unexpected_contact_steps += 1
            maximum_unexpected_penetration = max(
                maximum_unexpected_penetration,
                max(max(0.0, -float(contact.dist)) for contact in unexpected_contacts),
            )

    mujoco.mj_forward(model, data)
    final_joint_error = float(
        np.max(np.abs(data.qpos[:7] - plan.target_joint_positions_rad))
    )
    site_id = model.site("racket_center").id
    actual_racket_position = data.site_xpos[site_id].copy()
    actual_racket_normal = data.site_xmat[site_id].reshape(3, 3)[:, 2].copy()
    tracking_position_error = float(
        np.linalg.norm(
            actual_racket_position - plan.candidate.solution.racket_position_m
        )
    )
    tracking_normal_error = _normal_error_deg(
        actual_racket_normal,
        plan.candidate.solution.racket_normal,
    )
    contact_normal = (
        plan.candidate.ball_position_m - plan.candidate.racket_center_target_m
    )
    contact_normal /= np.linalg.norm(contact_normal)
    contact_position_error = float(
        np.linalg.norm(
            actual_racket_position - plan.candidate.racket_center_target_m
        )
    )
    contact_normal_error = _normal_error_deg(actual_racket_normal, contact_normal)
    failures = []
    if final_joint_error > config.maximum_final_joint_error_rad:
        failures.append("final_joint_tracking")
    if tracking_position_error > config.maximum_racket_position_tracking_error_m:
        failures.append("racket_position_tracking")
    if tracking_normal_error > config.maximum_racket_normal_tracking_error_deg:
        failures.append("racket_normal_tracking")
    if minimum_actual_margin < config.minimum_actual_joint_limit_margin_rad:
        failures.append("joint_limit_margin")
    if maximum_actual_speed > config.maximum_actual_joint_speed_rad_s:
        failures.append("actual_joint_speed")
    if (
        maximum_actual_acceleration
        > config.maximum_actual_joint_acceleration_rad_s2
    ):
        failures.append("actual_joint_acceleration")
    if clipped_commands:
        failures.append("clipped_servo_command")
    if unexpected_contact_steps:
        failures.append("unexpected_contact")

    errors = np.asarray(tracking_errors, dtype=np.float64)
    return ArrivalTrackingResult(
        duration_s=plan.duration_s,
        servo_steps=servo_steps,
        physics_steps=physics_steps,
        maximum_joint_tracking_error_rad=float(errors.max()),
        p95_joint_tracking_error_rad=float(np.quantile(errors, 0.95)),
        final_joint_tracking_error_rad=final_joint_error,
        maximum_actual_joint_speed_rad_s=maximum_actual_speed,
        maximum_actual_joint_acceleration_rad_s2=maximum_actual_acceleration,
        maximum_inverse_dynamics_torque_nm=maximum_inverse_torque,
        minimum_actual_joint_limit_margin_rad=minimum_actual_margin,
        clipped_servo_commands=clipped_commands,
        unexpected_contact_steps=unexpected_contact_steps,
        maximum_unexpected_penetration_m=maximum_unexpected_penetration,
        racket_position_tracking_error_m=tracking_position_error,
        racket_normal_tracking_error_deg=tracking_normal_error,
        final_contact_position_error_m=contact_position_error,
        final_contact_normal_error_deg=contact_normal_error,
        passed=not failures,
        failure_reasons=tuple(failures),
    )
