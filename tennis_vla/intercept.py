"""Kinematic racket-pose solver and privileged ball-intercept search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from .arm import EmbodimentLayout, arm_layout, tennis_ready_configuration
from .ballistics import BallFlightConfig, BallFlightResult


@dataclass(frozen=True)
class RacketIKConfig:
    position_tolerance_m: float = 0.015
    normal_tolerance_deg: float = 5.0
    damping: float = 0.04
    normal_weight: float = 0.25
    joint_limit_margin_rad: float = 0.03
    maximum_step_rad: float = 0.20
    maximum_iterations: int = 250
    restarts: int = 8
    local_restarts: int = 0
    local_restart_standard_deviation_rad: float = 0.6
    posture_weight: float = 0.0
    posture_tolerance_rad: float = 0.01
    initial_joint_guesses_rad: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class RacketIKSolution:
    converged: bool
    joint_positions_rad: np.ndarray
    racket_position_m: np.ndarray
    racket_normal: np.ndarray
    position_error_m: float
    normal_error_deg: float
    iterations: int
    restart: int

    def metrics(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "joint_positions_rad",
            "racket_position_m",
            "racket_normal",
        ):
            result[key] = result[key].tolist()
        return result


@dataclass(frozen=True)
class InterceptCandidate:
    time_s: float
    ball_position_m: np.ndarray
    ball_velocity_m_s: np.ndarray
    racket_center_target_m: np.ndarray
    solution: RacketIKSolution

    def metrics(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "ball_position_m": self.ball_position_m.tolist(),
            "ball_velocity_m_s": self.ball_velocity_m_s.tolist(),
            "racket_center_target_m": self.racket_center_target_m.tolist(),
            "ik": self.solution.metrics(),
        }


def _normal_error_deg(current: np.ndarray, target: np.ndarray) -> float:
    cosine = float(np.clip(current @ target, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def _ik_initial_guesses(
    model: mujoco.MjModel,
    config: RacketIKConfig,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    seed: int,
    layout: EmbodimentLayout,
) -> list[np.ndarray]:
    if config.restarts < 1:
        raise ValueError("restarts must be at least one")
    fixed_guess_count = 1 + len(config.initial_joint_guesses_rad)
    if not 0 <= config.local_restarts <= config.restarts - fixed_guess_count:
        raise ValueError("configured IK guesses exceed the restart count")
    if config.local_restart_standard_deviation_rad <= 0.0:
        raise ValueError("local restart deviation must be positive")
    rng = np.random.default_rng(seed)
    local_rng = np.random.default_rng(seed ^ 0x5EED5EED)
    ready = tennis_ready_configuration(model)
    guesses = [ready]
    for values in config.initial_joint_guesses_rad:
        guess = np.asarray(values, dtype=np.float64)
        if guess.shape != (layout.arm_dof_count,):
            raise ValueError(
                f"each initial joint guess must contain {layout.arm_dof_count} values"
            )
        guesses.append(np.clip(guess, joint_lower, joint_upper))
    guesses.extend(
        np.clip(
            ready
            + local_rng.normal(
                0.0,
                config.local_restart_standard_deviation_rad,
                size=layout.arm_dof_count,
            ),
            joint_lower,
            joint_upper,
        )
        for _ in range(config.local_restarts)
    )
    guesses.extend(
        rng.uniform(joint_lower, joint_upper)
        for _ in range(
            config.restarts - fixed_guess_count - config.local_restarts
        )
    )
    return guesses


def _solve_racket_pose_attempt(
    model: mujoco.MjModel,
    target_position: np.ndarray,
    target_normal: np.ndarray,
    config: RacketIKConfig,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    initial: np.ndarray,
    restart: int,
    layout: EmbodimentLayout,
) -> RacketIKSolution:
    site_id = model.site("racket_center").id
    data = mujoco.MjData(model)
    data.qpos[layout.arm_qpos] = initial
    preferred_position = tennis_ready_configuration(model)
    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
    iterations = 0
    for iterations in range(1, config.maximum_iterations + 1):
        mujoco.mj_forward(model, data)
        position = data.site_xpos[site_id].copy()
        normal = data.site_xmat[site_id].reshape(3, 3)[:, 2].copy()
        position_error = target_position - position
        normal_rotation_error = np.cross(normal, target_normal)
        position_norm = float(np.linalg.norm(position_error))
        normal_degrees = _normal_error_deg(normal, target_normal)
        task_converged = (
            position_norm <= config.position_tolerance_m
            and normal_degrees <= config.normal_tolerance_deg
        )
        if task_converged and config.posture_weight == 0.0:
            break

        mujoco.mj_jacSite(
            model,
            data,
            jacobian_position,
            jacobian_rotation,
            site_id,
        )
        jacobian = np.vstack(
            (
                jacobian_position[:, layout.arm_dof],
                config.normal_weight * jacobian_rotation[:, layout.arm_dof],
            )
        )
        error = np.concatenate(
            (position_error, config.normal_weight * normal_rotation_error)
        )
        regularized = (
            jacobian @ jacobian.T
            + config.damping**2 * np.eye(jacobian.shape[0])
        )
        damped_inverse = jacobian.T @ np.linalg.solve(
            regularized, np.eye(jacobian.shape[0])
        )
        joint_step = damped_inverse @ error
        if config.posture_weight > 0.0:
            nullspace = np.eye(layout.arm_dof_count) - damped_inverse @ jacobian
            posture_step = (
                config.posture_weight
                * nullspace
                @ (preferred_position - data.qpos[layout.arm_qpos])
            )
            if task_converged and np.linalg.norm(posture_step) <= (
                config.posture_tolerance_rad
            ):
                break
            joint_step += posture_step
        step_norm = float(np.linalg.norm(joint_step))
        if step_norm > config.maximum_step_rad:
            joint_step *= config.maximum_step_rad / step_norm
        data.qpos[layout.arm_qpos] = np.clip(
            data.qpos[layout.arm_qpos] + joint_step, joint_lower, joint_upper
        )

    mujoco.mj_forward(model, data)
    position = data.site_xpos[site_id].copy()
    normal = data.site_xmat[site_id].reshape(3, 3)[:, 2].copy()
    position_norm = float(np.linalg.norm(target_position - position))
    normal_degrees = _normal_error_deg(normal, target_normal)
    return RacketIKSolution(
        converged=(
            position_norm <= config.position_tolerance_m
            and normal_degrees <= config.normal_tolerance_deg
        ),
        joint_positions_rad=data.qpos[layout.arm_qpos].copy(),
        racket_position_m=position,
        racket_normal=normal,
        position_error_m=position_norm,
        normal_error_deg=normal_degrees,
        iterations=iterations,
        restart=restart,
    )


def _prepare_ik(
    model: mujoco.MjModel,
    target_position_m: np.ndarray,
    target_normal: np.ndarray,
    config: RacketIKConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, EmbodimentLayout]:
    if config.posture_weight < 0.0:
        raise ValueError("posture_weight cannot be negative")
    if config.posture_tolerance_rad <= 0.0:
        raise ValueError("posture_tolerance_rad must be positive")
    target_position = np.asarray(target_position_m, dtype=np.float64).reshape(3)
    normal = np.asarray(target_normal, dtype=np.float64).reshape(3).copy()
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9:
        raise ValueError("target_normal must be nonzero")
    normal /= normal_norm
    layout = arm_layout(model)
    joint_lower = model.jnt_range[layout.arm_joints, 0] + config.joint_limit_margin_rad
    joint_upper = model.jnt_range[layout.arm_joints, 1] - config.joint_limit_margin_rad
    if np.any(joint_lower >= joint_upper):
        raise ValueError("joint_limit_margin_rad leaves an empty joint range")
    return target_position, normal, joint_lower, joint_upper, layout


def solve_racket_pose(
    model: mujoco.MjModel,
    target_position_m: np.ndarray,
    target_normal: np.ndarray,
    *,
    config: RacketIKConfig | None = None,
    seed: int = 2026,
) -> RacketIKSolution:
    """Solve racket-center position and face-normal alignment with restarts."""
    config = config or RacketIKConfig()
    target_position, target_normal, joint_lower, joint_upper, layout = _prepare_ik(
        model, target_position_m, target_normal, config
    )
    initial_guesses = _ik_initial_guesses(
        model, config, joint_lower, joint_upper, seed, layout
    )
    best: RacketIKSolution | None = None

    for restart, initial in enumerate(initial_guesses):
        solution = _solve_racket_pose_attempt(
            model,
            target_position,
            target_normal,
            config,
            joint_lower,
            joint_upper,
            initial,
            restart,
            layout,
        )
        score = solution.position_error_m + np.deg2rad(
            solution.normal_error_deg
        ) * config.normal_weight
        if best is None:
            best = solution
        else:
            best_score = best.position_error_m + np.deg2rad(
                best.normal_error_deg
            ) * config.normal_weight
            if score < best_score:
                best = solution
        if solution.converged:
            return solution

    if best is None:
        raise RuntimeError("IK did not execute any restart")
    return best


def solve_racket_pose_candidates(
    model: mujoco.MjModel,
    target_position_m: np.ndarray,
    target_normal: np.ndarray,
    *,
    config: RacketIKConfig | None = None,
    seed: int = 2026,
    maximum_solutions: int = 4,
) -> list[RacketIKSolution]:
    """Return distinct converged IK branches for motion-aware strike planning."""
    if maximum_solutions < 1:
        raise ValueError("maximum_solutions must be at least one")
    config = config or RacketIKConfig()
    target_position, normal, joint_lower, joint_upper, layout = _prepare_ik(
        model, target_position_m, target_normal, config
    )
    guesses = _ik_initial_guesses(
        model, config, joint_lower, joint_upper, seed, layout
    )
    solutions = []
    for restart, initial in enumerate(guesses):
        solution = _solve_racket_pose_attempt(
            model,
            target_position,
            normal,
            config,
            joint_lower,
            joint_upper,
            initial,
            restart,
            layout,
        )
        if not solution.converged:
            continue
        if any(
            np.max(
                np.abs(
                    solution.joint_positions_rad - previous.joint_positions_rad
                )
            )
            < 1e-3
            for previous in solutions
        ):
            continue
        solutions.append(solution)
        if len(solutions) >= maximum_solutions:
            break
    return solutions


def _first_bounce_time(flight: BallFlightResult, ball_radius_m: float) -> float:
    candidates = np.flatnonzero(
        (flight.positions_m[:, 2] <= ball_radius_m + 1e-9)
        & (flight.velocities_m_s[:, 2] > 0.0)
    )
    if not len(candidates):
        raise ValueError("flight has no bounce")
    return float(flight.times_s[int(candidates[0])])


def find_kinematic_intercepts(
    model: mujoco.MjModel,
    flight: BallFlightResult,
    *,
    racket_normal: np.ndarray | None = None,
    sample_period_s: float = 0.02,
    maximum_candidates: int = 8,
    solutions_per_pose: int = 1,
    contact_x_bounds_m: tuple[float, float] = (-10.1, -9.45),
    maximum_abs_contact_y_m: float = 1.1,
    contact_z_bounds_m: tuple[float, float] = (0.60, 1.35),
    ik_config: RacketIKConfig | None = None,
    seed: int = 2026,
) -> list[InterceptCandidate]:
    """Find post-bounce samples that admit a kinematic racket contact pose."""
    if solutions_per_pose < 1:
        raise ValueError("solutions_per_pose must be at least one")
    if contact_x_bounds_m[0] >= contact_x_bounds_m[1]:
        raise ValueError("contact x bounds must be increasing")
    if contact_z_bounds_m[0] >= contact_z_bounds_m[1]:
        raise ValueError("contact z bounds must be increasing")
    if maximum_abs_contact_y_m <= 0.0:
        raise ValueError("maximum absolute contact y must be positive")
    ball = BallFlightConfig()
    bounce_time = _first_bounce_time(flight, ball.radius_m)
    first_index = int(np.searchsorted(flight.times_s, bounce_time))
    stride = max(1, int(round(sample_period_s / ball.dt_s)))
    normal = np.asarray(
        [1.0, 0.0, 0.0] if racket_normal is None else racket_normal,
        dtype=np.float64,
    ).reshape(3)
    normal /= np.linalg.norm(normal)
    contact_offset_m = ball.radius_m + 0.015
    candidates: list[InterceptCandidate] = []

    for index in range(first_index, len(flight.times_s), stride):
        ball_position = flight.positions_m[index]
        if not (
            contact_x_bounds_m[0]
            <= ball_position[0]
            <= contact_x_bounds_m[1]
            and abs(ball_position[1]) <= maximum_abs_contact_y_m
            and contact_z_bounds_m[0]
            <= ball_position[2]
            <= contact_z_bounds_m[1]
        ):
            continue
        racket_target = ball_position - contact_offset_m * normal
        if solutions_per_pose == 1:
            solution = solve_racket_pose(
                model,
                racket_target,
                normal,
                config=ik_config,
                seed=seed + index,
            )
            solutions = [solution] if solution.converged else []
        else:
            solutions = solve_racket_pose_candidates(
                model,
                racket_target,
                normal,
                config=ik_config,
                seed=seed + index,
                maximum_solutions=solutions_per_pose,
            )
        for solution in solutions:
            candidates.append(
                InterceptCandidate(
                    time_s=float(flight.times_s[index]),
                    ball_position_m=ball_position.copy(),
                    ball_velocity_m_s=flight.velocities_m_s[index].copy(),
                    racket_center_target_m=racket_target,
                    solution=solution,
                )
            )
            if len(candidates) >= maximum_candidates:
                return candidates
    return candidates
