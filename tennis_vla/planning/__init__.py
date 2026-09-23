"""Privileged planning: racket-pose IK, joint trajectories, and strike search."""

from __future__ import annotations

from .intercept import (
    InterceptCandidate,
    RacketIKConfig,
    RacketIKSolution,
    find_kinematic_intercepts,
    solve_racket_pose,
    solve_racket_pose_candidates,
)
from .trajectory import (
    ArrivalTrackingConfig,
    ArrivalTrackingResult,
    InterceptArrivalPlan,
    MINIMUM_JERK_PEAK_ACCELERATION,
    MINIMUM_JERK_PEAK_SPEED,
    SimulationJointMotionLimits,
    assess_intercept_arrival,
    earliest_feasible_arrival,
    plan_intercept_arrivals,
    sample_minimum_jerk,
    track_intercept_arrival,
)
from .strike import (
    DEFAULT_RECOVERY_DURATIONS_S,
    JointTrajectoryBounds,
    QuinticJointTrajectory,
    StrikePlan,
    StrikeSearchConfig,
    minimum_infinity_joint_velocity,
    plan_ready_recovery_trajectory,
    plan_safe_center_strikes,
    strike_ik_config,
    trajectory_is_execution_safe,
)

__all__ = [
    "ArrivalTrackingConfig",
    "ArrivalTrackingResult",
    "DEFAULT_RECOVERY_DURATIONS_S",
    "InterceptArrivalPlan",
    "InterceptCandidate",
    "JointTrajectoryBounds",
    "MINIMUM_JERK_PEAK_ACCELERATION",
    "MINIMUM_JERK_PEAK_SPEED",
    "QuinticJointTrajectory",
    "RacketIKConfig",
    "RacketIKSolution",
    "SimulationJointMotionLimits",
    "StrikePlan",
    "StrikeSearchConfig",
    "assess_intercept_arrival",
    "earliest_feasible_arrival",
    "find_kinematic_intercepts",
    "minimum_infinity_joint_velocity",
    "plan_intercept_arrivals",
    "plan_ready_recovery_trajectory",
    "plan_safe_center_strikes",
    "sample_minimum_jerk",
    "solve_racket_pose",
    "solve_racket_pose_candidates",
    "strike_ik_config",
    "track_intercept_arrival",
    "trajectory_is_execution_safe",
]
