"""Joint-space arrival trajectories for privileged tennis intercepts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .arm import home_configuration
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
        home_configuration(model)
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
