"""Kinematic racket-pose solver and privileged ball-intercept search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from .arm import home_configuration
from .ballistics import BallFlightConfig, BallFlightResult


@dataclass(frozen=True)
class RacketIKConfig:
    position_tolerance_m: float = 0.015
    normal_tolerance_deg: float = 5.0
    damping: float = 0.04
    normal_weight: float = 0.25
    maximum_step_rad: float = 0.20
    maximum_iterations: int = 250
    restarts: int = 8


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
    target_position = np.asarray(target_position_m, dtype=np.float64).reshape(3)
    target_normal = np.asarray(target_normal, dtype=np.float64).reshape(3).copy()
    normal_norm = float(np.linalg.norm(target_normal))
    if normal_norm < 1e-9:
        raise ValueError("target_normal must be nonzero")
    target_normal /= normal_norm
    if model.nq < 7 or model.nv < 7:
        raise ValueError("model must contain the seven arm joints first")

    joint_lower = model.jnt_range[:7, 0]
    joint_upper = model.jnt_range[:7, 1]
    site_id = model.site("racket_center").id
    rng = np.random.default_rng(seed)
    initial_guesses = [home_configuration(model)]
    initial_guesses.extend(
        rng.uniform(joint_lower, joint_upper) for _ in range(config.restarts - 1)
    )
    best: RacketIKSolution | None = None

    for restart, initial in enumerate(initial_guesses):
        data = mujoco.MjData(model)
        data.qpos[:7] = initial
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
            if (
                position_norm <= config.position_tolerance_m
                and normal_degrees <= config.normal_tolerance_deg
            ):
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
                    jacobian_position[:, :7],
                    config.normal_weight * jacobian_rotation[:, :7],
                )
            )
            error = np.concatenate(
                (position_error, config.normal_weight * normal_rotation_error)
            )
            regularized = (
                jacobian @ jacobian.T
                + config.damping**2 * np.eye(jacobian.shape[0])
            )
            joint_step = jacobian.T @ np.linalg.solve(regularized, error)
            step_norm = float(np.linalg.norm(joint_step))
            if step_norm > config.maximum_step_rad:
                joint_step *= config.maximum_step_rad / step_norm
            data.qpos[:7] = np.clip(
                data.qpos[:7] + joint_step, joint_lower, joint_upper
            )

        mujoco.mj_forward(model, data)
        position = data.site_xpos[site_id].copy()
        normal = data.site_xmat[site_id].reshape(3, 3)[:, 2].copy()
        position_norm = float(np.linalg.norm(target_position - position))
        normal_degrees = _normal_error_deg(normal, target_normal)
        solution = RacketIKSolution(
            converged=(
                position_norm <= config.position_tolerance_m
                and normal_degrees <= config.normal_tolerance_deg
            ),
            joint_positions_rad=data.qpos[:7].copy(),
            racket_position_m=position,
            racket_normal=normal,
            position_error_m=position_norm,
            normal_error_deg=normal_degrees,
            iterations=iterations,
            restart=restart,
        )
        score = position_norm + np.deg2rad(normal_degrees) * config.normal_weight
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
    ik_config: RacketIKConfig | None = None,
    seed: int = 2026,
) -> list[InterceptCandidate]:
    """Find post-bounce samples that admit a kinematic racket contact pose."""
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
            -10.1 <= ball_position[0] <= -9.45
            and abs(ball_position[1]) <= 1.1
            and 0.15 <= ball_position[2] <= 1.35
        ):
            continue
        racket_target = ball_position - contact_offset_m * normal
        solution = solve_racket_pose(
            model,
            racket_target,
            normal,
            config=ik_config,
            seed=seed + index,
        )
        if solution.converged:
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
                break
    return candidates
