"""Privileged nonzero-velocity strike planning for tennis returns."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .arm import tennis_ready_configuration
from .ballistics import BallFlightConfig, BallFlightResult, simulate_ball_flight
from .impact import apply_racket_impact
from .intercept import InterceptCandidate, RacketIKConfig, find_kinematic_intercepts
from .trajectory import SimulationJointMotionLimits


DEFAULT_RECOVERY_DURATIONS_S = (
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.25,
    2.50,
    2.75,
    3.00,
    3.25,
    3.50,
    3.75,
    4.00,
)


@dataclass(frozen=True)
class StrikeSearchConfig:
    """Discrete phase-one search over safe-center stroke templates."""

    face_pitch_degrees: tuple[float, ...] = (
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        22.0,
        24.0,
        26.0,
        28.0,
        30.0,
        32.0,
        34.0,
        36.0,
    )
    racket_normal_speeds_m_s: tuple[float, ...] = (
        3.0,
        2.9,
        2.8,
        2.7,
        2.6,
        2.5,
        2.4,
        2.3,
        2.2,
        2.1,
        2.0,
        1.9,
        1.8,
        1.7,
        1.6,
        1.5,
        1.4,
        1.3,
        1.2,
        1.1,
        1.0,
    )
    racket_tangent_ratios: tuple[float, ...] = (
        0.0,
        -0.5,
        -0.75,
        -1.0,
        -0.25,
    )
    alternative_ik_solutions: int = 4
    planning_ball_timestep_s: float = 0.005
    maximum_returned_plans: int = 8
    trajectory_safety_sample_period_s: float = 0.01
    post_contact_safety_horizon_s: float = 0.05
    predicted_recovery_start_delay_s: float = 0.012
    recovery_duration_candidates_s: tuple[float, ...] = DEFAULT_RECOVERY_DURATIONS_S
    maximum_planned_recovery_acceleration_rad_s2: float = 13.6
    preferred_motion_limit_utilization: float = 0.95
    landing_target_xy_m: tuple[float, float] = (4.0, 0.0)
    minimum_net_clearance_m: float = 0.10
    fixed_wrist_roll_velocity_rad_s: float = 0.0

    def validate(self) -> None:
        if not self.face_pitch_degrees:
            raise ValueError("face_pitch_degrees cannot be empty")
        if not self.racket_normal_speeds_m_s:
            raise ValueError("racket_normal_speeds_m_s cannot be empty")
        if any(speed <= 0.0 for speed in self.racket_normal_speeds_m_s):
            raise ValueError("racket normal speeds must be positive")
        if not self.racket_tangent_ratios:
            raise ValueError("racket_tangent_ratios cannot be empty")
        if self.alternative_ik_solutions < 1:
            raise ValueError("alternative_ik_solutions must be at least one")
        if self.planning_ball_timestep_s <= 0.0:
            raise ValueError("planning_ball_timestep_s must be positive")
        if self.maximum_returned_plans < 1:
            raise ValueError("maximum_returned_plans must be at least one")
        if self.trajectory_safety_sample_period_s <= 0.0:
            raise ValueError("trajectory safety sample period must be positive")
        if self.post_contact_safety_horizon_s < 0.0:
            raise ValueError("post-contact safety horizon cannot be negative")
        if self.predicted_recovery_start_delay_s < 0.0:
            raise ValueError("predicted recovery start delay cannot be negative")
        if not self.recovery_duration_candidates_s or any(
            duration <= 0.0 for duration in self.recovery_duration_candidates_s
        ):
            raise ValueError("recovery duration candidates must be positive")
        if self.maximum_planned_recovery_acceleration_rad_s2 <= 0.0:
            raise ValueError(
                "maximum planned recovery acceleration must be positive"
            )
        if not 0.0 < self.preferred_motion_limit_utilization <= 1.0:
            raise ValueError(
                "preferred motion limit utilization must be in (0, 1]"
            )
        if self.minimum_net_clearance_m < 0.0:
            raise ValueError("minimum_net_clearance_m cannot be negative")


@dataclass(frozen=True)
class JointTrajectoryBounds:
    peak_joint_speeds_rad_s: np.ndarray
    peak_joint_accelerations_rad_s2: np.ndarray
    minimum_joint_limit_margin_rad: float

    @property
    def maximum_joint_speed_rad_s(self) -> float:
        return float(np.max(self.peak_joint_speeds_rad_s))

    @property
    def maximum_joint_acceleration_rad_s2(self) -> float:
        return float(np.max(self.peak_joint_accelerations_rad_s2))


@dataclass(frozen=True)
class QuinticJointTrajectory:
    """Quintic joint trajectory with position, velocity, and acceleration ends."""

    duration_s: float
    coefficients: np.ndarray

    @classmethod
    def from_boundary_conditions(
        cls,
        start_position_rad: np.ndarray,
        target_position_rad: np.ndarray,
        *,
        duration_s: float,
        start_velocity_rad_s: np.ndarray | None = None,
        target_velocity_rad_s: np.ndarray | None = None,
        start_acceleration_rad_s2: np.ndarray | None = None,
        target_acceleration_rad_s2: np.ndarray | None = None,
    ) -> QuinticJointTrajectory:
        if duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        start = np.asarray(start_position_rad, dtype=np.float64)
        target = np.asarray(target_position_rad, dtype=np.float64)
        if start.shape != target.shape:
            raise ValueError("start and target positions must have the same shape")

        def optional(value: np.ndarray | None) -> np.ndarray:
            if value is None:
                return np.zeros_like(start)
            array = np.asarray(value, dtype=np.float64)
            if array.shape != start.shape:
                raise ValueError("trajectory boundary arrays must share a shape")
            return array

        start_velocity = optional(start_velocity_rad_s)
        target_velocity = optional(target_velocity_rad_s)
        start_acceleration = optional(start_acceleration_rad_s2)
        target_acceleration = optional(target_acceleration_rad_s2)
        duration = float(duration_s)
        coefficients = np.empty((start.size, 6), dtype=np.float64)
        coefficients[:, 0] = start
        coefficients[:, 1] = start_velocity * duration
        coefficients[:, 2] = 0.5 * start_acceleration * duration**2
        residual_position = target - coefficients[:, :3].sum(axis=1)
        residual_velocity = (
            target_velocity * duration
            - coefficients[:, 1]
            - 2.0 * coefficients[:, 2]
        )
        residual_acceleration = (
            target_acceleration * duration**2 - 2.0 * coefficients[:, 2]
        )
        boundary_matrix = np.array(
            [
                [1.0, 1.0, 1.0],
                [3.0, 4.0, 5.0],
                [6.0, 12.0, 20.0],
            ]
        )
        coefficients[:, 3:] = np.linalg.solve(
            boundary_matrix,
            np.vstack(
                (
                    residual_position,
                    residual_velocity,
                    residual_acceleration,
                )
            ),
        ).T
        return cls(duration_s=duration, coefficients=coefficients)

    def sample(
        self, elapsed_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phase = float(np.clip(elapsed_s / self.duration_s, 0.0, 1.0))
        powers = np.array([1.0, phase, phase**2, phase**3, phase**4, phase**5])
        position = self.coefficients @ powers
        velocity_phase = np.array(
            [0.0, 1.0, 2.0 * phase, 3.0 * phase**2, 4.0 * phase**3, 5.0 * phase**4]
        )
        acceleration_phase = np.array(
            [0.0, 0.0, 2.0, 6.0 * phase, 12.0 * phase**2, 20.0 * phase**3]
        )
        velocity = self.coefficients @ velocity_phase / self.duration_s
        acceleration = (
            self.coefficients @ acceleration_phase / self.duration_s**2
        )
        return position, velocity, acceleration

    def bounds(self, model: mujoco.MjModel) -> JointTrajectoryBounds:
        """Compute exact polynomial extrema over the closed trajectory interval."""
        peak_speeds = np.zeros(self.coefficients.shape[0], dtype=np.float64)
        peak_accelerations = np.zeros_like(peak_speeds)
        minimum_margin = float("inf")
        joint_lower = model.jnt_range[: len(peak_speeds), 0]
        joint_upper = model.jnt_range[: len(peak_speeds), 1]

        for joint, coefficients in enumerate(self.coefficients):
            c0, c1, c2, c3, c4, c5 = coefficients
            velocity_coefficients = np.array([5 * c5, 4 * c4, 3 * c3, 2 * c2, c1])
            acceleration_coefficients = np.array([20 * c5, 12 * c4, 6 * c3, 2 * c2])
            jerk_coefficients = np.array([60 * c5, 24 * c4, 6 * c3])

            position_phases = [0.0, 1.0, *_unit_interval_roots(velocity_coefficients)]
            speed_phases = [0.0, 1.0, *_unit_interval_roots(acceleration_coefficients)]
            acceleration_phases = [0.0, 1.0, *_unit_interval_roots(jerk_coefficients)]
            positions = [
                self.sample(phase * self.duration_s)[0][joint]
                for phase in position_phases
            ]
            speeds = [
                self.sample(phase * self.duration_s)[1][joint]
                for phase in speed_phases
            ]
            accelerations = [
                self.sample(phase * self.duration_s)[2][joint]
                for phase in acceleration_phases
            ]
            peak_speeds[joint] = max(abs(value) for value in speeds)
            peak_accelerations[joint] = max(abs(value) for value in accelerations)
            minimum_margin = min(
                minimum_margin,
                min(
                    min(value - joint_lower[joint], joint_upper[joint] - value)
                    for value in positions
                ),
            )
        return JointTrajectoryBounds(
            peak_joint_speeds_rad_s=peak_speeds,
            peak_joint_accelerations_rad_s2=peak_accelerations,
            minimum_joint_limit_margin_rad=minimum_margin,
        )


@dataclass(frozen=True)
class StrikePlan:
    candidate: InterceptCandidate
    face_pitch_degrees: float
    requested_racket_normal_speed_m_s: float
    requested_racket_tangent_ratio: float
    requested_racket_tangent_speed_m_s: float
    contact_joint_velocities_rad_s: np.ndarray
    contact_racket_velocity_m_s: np.ndarray
    outgoing_ball_velocity_m_s: np.ndarray
    predicted_return: BallFlightResult
    trajectory: QuinticJointTrajectory
    trajectory_bounds: JointTrajectoryBounds
    landing_target_xy_m: np.ndarray
    landing_error_m: float

    def metrics(self) -> dict[str, Any]:
        bounce = self.predicted_return.first_bounce_m
        return {
            "intercept": self.candidate.metrics(),
            "face_pitch_degrees": self.face_pitch_degrees,
            "requested_racket_normal_speed_m_s": (
                self.requested_racket_normal_speed_m_s
            ),
            "requested_racket_tangent_ratio": (
                self.requested_racket_tangent_ratio
            ),
            "requested_racket_tangent_speed_m_s": (
                self.requested_racket_tangent_speed_m_s
            ),
            "contact_joint_velocities_rad_s": (
                self.contact_joint_velocities_rad_s.tolist()
            ),
            "contact_racket_velocity_m_s": self.contact_racket_velocity_m_s.tolist(),
            "outgoing_ball_velocity_m_s": self.outgoing_ball_velocity_m_s.tolist(),
            "predicted_net_clearance_m": self.predicted_return.net_clearance_m,
            "predicted_first_bounce_m": None if bounce is None else bounce.tolist(),
            "predicted_legal_return": self.predicted_return.legal_first_bounce,
            "landing_target_xy_m": self.landing_target_xy_m.tolist(),
            "landing_error_m": self.landing_error_m,
            "trajectory": {
                "duration_s": self.trajectory.duration_s,
                "start_joint_positions_rad": (
                    self.trajectory.coefficients[:, 0].tolist()
                ),
                "contact_joint_positions_rad": (
                    self.candidate.solution.joint_positions_rad.tolist()
                ),
                "peak_joint_speeds_rad_s": (
                    self.trajectory_bounds.peak_joint_speeds_rad_s.tolist()
                ),
                "peak_joint_accelerations_rad_s2": (
                    self.trajectory_bounds.peak_joint_accelerations_rad_s2.tolist()
                ),
                "maximum_joint_speed_rad_s": (
                    self.trajectory_bounds.maximum_joint_speed_rad_s
                ),
                "maximum_joint_acceleration_rad_s2": (
                    self.trajectory_bounds.maximum_joint_acceleration_rad_s2
                ),
                "minimum_joint_limit_margin_rad": (
                    self.trajectory_bounds.minimum_joint_limit_margin_rad
                ),
            },
        }


def strike_ik_config() -> RacketIKConfig:
    return RacketIKConfig(
        position_tolerance_m=0.005,
        normal_tolerance_deg=0.5,
        damping=0.02,
        maximum_iterations=350,
        restarts=24,
        initial_joint_guesses_rad=(
            (
                -1.42072953,
                -0.62989468,
                -2.50767152,
                -0.42313329,
                -2.47537407,
                2.62599651,
                4.38343366,
            ),
        ),
    )


def _unit_interval_roots(coefficients: np.ndarray) -> list[float]:
    trimmed = np.trim_zeros(np.asarray(coefficients, dtype=np.float64), "f")
    if not len(trimmed):
        return []
    roots = np.roots(trimmed)
    return sorted(
        float(root.real)
        for root in roots
        if abs(float(root.imag)) < 1e-9 and 0.0 < float(root.real) < 1.0
    )


def minimum_infinity_joint_velocity(
    jacobian: np.ndarray,
    target_velocity: np.ndarray,
    *,
    fixed_joint_velocities: dict[int, float] | None = None,
) -> np.ndarray:
    """Solve ``J qdot = velocity`` while minimizing the largest free-joint speed.

    The small tennis system admits an exact deterministic vertex enumeration,
    avoiding a heavyweight optimization dependency for a 3x7 problem.
    """
    matrix = np.asarray(jacobian, dtype=np.float64)
    target = np.asarray(target_velocity, dtype=np.float64).reshape(matrix.shape[0])
    fixed = fixed_joint_velocities or {}
    if matrix.ndim != 2:
        raise ValueError("jacobian must be a matrix")
    if any(index < 0 or index >= matrix.shape[1] for index in fixed):
        raise ValueError("fixed joint index is outside the Jacobian")
    free = [index for index in range(matrix.shape[1]) if index not in fixed]
    fixed_vector = np.zeros(matrix.shape[1], dtype=np.float64)
    for index, value in fixed.items():
        fixed_vector[index] = value
    reduced_target = target - matrix @ fixed_vector
    reduced = matrix[:, free]
    rows, columns = reduced.shape
    if np.linalg.matrix_rank(reduced) < rows:
        raise ValueError("free-joint Jacobian does not span the target velocity")
    active_count = columns + 1 - rows
    if active_count < 0:
        raise ValueError("velocity system has more constraints than variables")

    best: tuple[float, float, np.ndarray] | None = None
    for indices in itertools.combinations(range(columns), active_count):
        for signs in itertools.product((-1.0, 1.0), repeat=active_count):
            system = np.zeros((columns + 1, columns + 1), dtype=np.float64)
            right_hand_side = np.zeros(columns + 1, dtype=np.float64)
            system[:rows, :columns] = reduced
            right_hand_side[:rows] = reduced_target
            for row, (index, sign) in enumerate(zip(indices, signs), start=rows):
                system[row, index] = 1.0
                system[row, columns] = -sign
            try:
                solution = np.linalg.solve(system, right_hand_side)
            except np.linalg.LinAlgError:
                continue
            velocity = solution[:columns]
            infinity_norm = float(solution[columns])
            if infinity_norm < -1e-9:
                continue
            if np.max(np.abs(velocity)) > infinity_norm + 1e-7:
                continue
            if np.linalg.norm(reduced @ velocity - reduced_target) > 1e-7:
                continue
            score = (infinity_norm, float(np.linalg.norm(velocity)), velocity)
            if best is None or score[:2] < best[:2]:
                best = score
    if best is None:
        raise ValueError("no bounded joint-velocity solution was found")
    result = fixed_vector.copy()
    result[free] = best[2]
    return result


def trajectory_is_execution_safe(
    model: mujoco.MjModel,
    trajectory: QuinticJointTrajectory,
    *,
    sample_period_s: float = 0.01,
    post_contact_time_s: float = 0.0,
) -> bool:
    """Reject sampled collisions and velocity-compensation command clipping."""
    if sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be positive")
    if post_contact_time_s < 0.0:
        raise ValueError("post_contact_time_s cannot be negative")
    data = mujoco.MjData(model)
    ball_joint = model.joint("ball_free")
    ball_qpos_address = int(ball_joint.qposadr[0])
    data.qpos[ball_qpos_address : ball_qpos_address + 7] = [
        10.0,
        0.0,
        4.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    actuator_gain = model.actuator_gainprm[:7, 0]
    actuator_damping = -model.actuator_biasprm[:7, 2]
    control_lower = model.actuator_ctrlrange[:7, 0]
    control_upper = model.actuator_ctrlrange[:7, 1]
    total_duration = trajectory.duration_s + post_contact_time_s
    sample_count = int(np.ceil(total_duration / sample_period_s))
    terminal_position, terminal_velocity, _ = trajectory.sample(
        trajectory.duration_s
    )
    for elapsed_s in np.linspace(0.0, total_duration, sample_count + 1):
        if elapsed_s <= trajectory.duration_s:
            position, velocity, _ = trajectory.sample(float(elapsed_s))
        else:
            position = terminal_position + terminal_velocity * (
                elapsed_s - trajectory.duration_s
            )
            velocity = terminal_velocity
        position_command = position + actuator_damping * velocity / actuator_gain
        if np.any(position_command < control_lower) or np.any(
            position_command > control_upper
        ):
            return False
        data.qpos[:7] = position
        mujoco.mj_forward(model, data)
        if data.ncon:
            return False
    return True


def plan_ready_recovery_trajectory(
    model: mujoco.MjModel,
    start_position_rad: np.ndarray,
    start_velocity_rad_s: np.ndarray,
    *,
    duration_candidates_s: tuple[float, ...] = DEFAULT_RECOVERY_DURATIONS_S,
    limits: SimulationJointMotionLimits | None = None,
    sample_period_s: float = 0.01,
) -> tuple[QuinticJointTrajectory, JointTrajectoryBounds] | None:
    """Find the shortest bounded, collision-free path to the ready pose."""
    if not duration_candidates_s or any(
        duration <= 0.0 for duration in duration_candidates_s
    ):
        raise ValueError("recovery duration candidates must be positive")
    limits = limits or SimulationJointMotionLimits()
    limits.validate()
    ready = tennis_ready_configuration(model)
    zeros = np.zeros(7, dtype=np.float64)
    for duration_s in duration_candidates_s:
        trajectory = QuinticJointTrajectory.from_boundary_conditions(
            start_position_rad,
            ready,
            duration_s=duration_s,
            start_velocity_rad_s=start_velocity_rad_s,
            target_velocity_rad_s=zeros,
            start_acceleration_rad_s2=zeros,
            target_acceleration_rad_s2=zeros,
        )
        bounds = trajectory.bounds(model)
        if (
            bounds.maximum_joint_speed_rad_s > limits.maximum_speed_rad_s
            or bounds.maximum_joint_acceleration_rad_s2
            > limits.maximum_acceleration_rad_s2
            or bounds.minimum_joint_limit_margin_rad
            < limits.minimum_joint_limit_margin_rad
        ):
            continue
        if trajectory_is_execution_safe(
            model,
            trajectory,
            sample_period_s=sample_period_s,
        ):
            return trajectory, bounds
    return None


def plan_safe_center_strikes(
    model: mujoco.MjModel,
    flight: BallFlightResult,
    *,
    search: StrikeSearchConfig | None = None,
    limits: SimulationJointMotionLimits | None = None,
    ik_config: RacketIKConfig | None = None,
    seed: int = 2026,
) -> list[StrikePlan]:
    """Find legal nonzero-velocity returns ranked by landing-target error."""
    search = search or StrikeSearchConfig()
    search.validate()
    limits = limits or SimulationJointMotionLimits()
    limits.validate()
    ik_config = ik_config or strike_ik_config()
    start = tennis_ready_configuration(model)
    landing_target = np.asarray(search.landing_target_xy_m, dtype=np.float64)
    site_id = model.site("racket_center").id
    def search_with_ik_branches(solutions_per_pose: int) -> list[StrikePlan]:
        coarse_records = []
        planning_flight = BallFlightConfig(
            dt_s=search.planning_ball_timestep_s
        )
        for pitch_degrees in search.face_pitch_degrees:
            pitch = np.deg2rad(pitch_degrees)
            target_normal = np.array([np.cos(pitch), 0.0, np.sin(pitch)])
            vertical_tangent = np.array(
                [-np.sin(pitch), 0.0, np.cos(pitch)]
            )
            candidates = find_kinematic_intercepts(
                model,
                flight,
                racket_normal=target_normal,
                maximum_candidates=8 * solutions_per_pose,
                solutions_per_pose=solutions_per_pose,
                ik_config=ik_config,
                seed=seed,
            )
            for candidate in candidates:
                data = mujoco.MjData(model)
                data.qpos[:7] = candidate.solution.joint_positions_rad
                mujoco.mj_forward(model, data)
                jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
                jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
                mujoco.mj_jacSite(
                    model,
                    data,
                    jacobian_position,
                    jacobian_rotation,
                    site_id,
                )
                for tangent_ratio in search.racket_tangent_ratios:
                    unit_racket_velocity = (
                        target_normal + tangent_ratio * vertical_tangent
                    )
                    try:
                        unit_joint_velocity = minimum_infinity_joint_velocity(
                            jacobian_position[:, :7],
                            unit_racket_velocity,
                            fixed_joint_velocities={
                                6: search.fixed_wrist_roll_velocity_rad_s
                            },
                        )
                    except ValueError:
                        continue

                    for racket_speed in search.racket_normal_speeds_m_s:
                        contact_joint_velocity = (
                            unit_joint_velocity * racket_speed
                        )
                        racket_velocity = (
                            jacobian_position[:, :7] @ contact_joint_velocity
                        )
                        try:
                            outgoing_velocity = apply_racket_impact(
                                candidate.ball_velocity_m_s,
                                racket_velocity,
                                candidate.solution.racket_normal,
                            )
                        except ValueError:
                            continue
                        trajectory = (
                            QuinticJointTrajectory.from_boundary_conditions(
                                start,
                                candidate.solution.joint_positions_rad,
                                duration_s=candidate.time_s,
                                target_velocity_rad_s=contact_joint_velocity,
                            )
                        )
                        bounds = trajectory.bounds(model)
                        if (
                            bounds.maximum_joint_speed_rad_s
                            > limits.maximum_speed_rad_s + 1e-9
                            or bounds.maximum_joint_acceleration_rad_s2
                            > limits.maximum_acceleration_rad_s2 + 1e-9
                            or bounds.minimum_joint_limit_margin_rad
                            < limits.minimum_joint_limit_margin_rad - 1e-9
                        ):
                            continue
                        if not trajectory_is_execution_safe(
                            model,
                            trajectory,
                            sample_period_s=(
                                search.trajectory_safety_sample_period_s
                            ),
                            post_contact_time_s=(
                                search.post_contact_safety_horizon_s
                            ),
                        ):
                            continue
                        predicted_recovery_position = (
                            candidate.solution.joint_positions_rad
                            + contact_joint_velocity
                            * search.predicted_recovery_start_delay_s
                        )
                        if plan_ready_recovery_trajectory(
                            model,
                            predicted_recovery_position,
                            contact_joint_velocity,
                            duration_candidates_s=(
                                search.recovery_duration_candidates_s
                            ),
                            limits=SimulationJointMotionLimits(
                                maximum_speed_rad_s=limits.maximum_speed_rad_s,
                                maximum_acceleration_rad_s2=min(
                                    limits.maximum_acceleration_rad_s2,
                                    search.maximum_planned_recovery_acceleration_rad_s2,
                                ),
                                minimum_joint_limit_margin_rad=(
                                    limits.minimum_joint_limit_margin_rad
                                ),
                            ),
                            sample_period_s=(
                                search.trajectory_safety_sample_period_s
                            ),
                        ) is None:
                            continue
                        predicted_return = simulate_ball_flight(
                            candidate.ball_position_m,
                            outgoing_velocity,
                            planning_flight,
                        )
                        bounce = predicted_return.first_bounce_m
                        if (
                            not predicted_return.legal_first_bounce
                            or predicted_return.net_clearance_m is None
                            or predicted_return.net_clearance_m
                            < search.minimum_net_clearance_m
                            or bounce is None
                        ):
                            continue
                        landing_error = float(
                            np.linalg.norm(bounce[:2] - landing_target)
                        )
                        coarse_records.append(
                            (
                                landing_error,
                                candidate,
                                pitch_degrees,
                                racket_speed,
                                tangent_ratio,
                                contact_joint_velocity,
                                racket_velocity,
                                outgoing_velocity,
                                trajectory,
                                bounds,
                            )
                        )
        plans = []
        for record in sorted(coarse_records, key=lambda item: item[0]):
            (
                _,
                candidate,
                pitch_degrees,
                racket_speed,
                tangent_ratio,
                contact_joint_velocity,
                racket_velocity,
                outgoing_velocity,
                trajectory,
                bounds,
            ) = record
            predicted_return = simulate_ball_flight(
                candidate.ball_position_m,
                outgoing_velocity,
            )
            bounce = predicted_return.first_bounce_m
            if (
                not predicted_return.legal_first_bounce
                or predicted_return.net_clearance_m is None
                or predicted_return.net_clearance_m
                < search.minimum_net_clearance_m
                or bounce is None
            ):
                continue
            landing_error = float(np.linalg.norm(bounce[:2] - landing_target))
            plans.append(
                StrikePlan(
                    candidate=candidate,
                    face_pitch_degrees=pitch_degrees,
                    requested_racket_normal_speed_m_s=racket_speed,
                    requested_racket_tangent_ratio=tangent_ratio,
                    requested_racket_tangent_speed_m_s=(
                        tangent_ratio * racket_speed
                    ),
                    contact_joint_velocities_rad_s=contact_joint_velocity,
                    contact_racket_velocity_m_s=racket_velocity,
                    outgoing_ball_velocity_m_s=outgoing_velocity,
                    predicted_return=predicted_return,
                    trajectory=trajectory,
                    trajectory_bounds=bounds,
                    landing_target_xy_m=landing_target.copy(),
                    landing_error_m=landing_error,
                )
            )
            if len(plans) >= search.maximum_returned_plans:
                break
        return plans

    plans = search_with_ik_branches(1)
    if not plans and search.alternative_ik_solutions > 1:
        plans = search_with_ik_branches(search.alternative_ik_solutions)

    return sorted(
        plans,
        key=lambda plan: (
            max(
                plan.trajectory_bounds.maximum_joint_speed_rad_s
                / limits.maximum_speed_rad_s,
                plan.trajectory_bounds.maximum_joint_acceleration_rad_s2
                / limits.maximum_acceleration_rad_s2,
            )
            > search.preferred_motion_limit_utilization,
            plan.landing_error_m,
            max(
                plan.trajectory_bounds.maximum_joint_speed_rad_s
                / limits.maximum_speed_rad_s,
                plan.trajectory_bounds.maximum_joint_acceleration_rad_s2
                / limits.maximum_acceleration_rad_s2,
            ),
            -float(plan.predicted_return.net_clearance_m or 0.0),
        ),
    )
