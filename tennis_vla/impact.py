"""First-order moving-racket impact model for oracle and unit validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RacketImpactConfig:
    normal_restitution: float = 0.78
    tangential_speed_retention: float = 0.82


def apply_racket_impact(
    ball_velocity_m_s: np.ndarray,
    racket_velocity_m_s: np.ndarray,
    racket_normal: np.ndarray,
    config: RacketImpactConfig | None = None,
) -> np.ndarray:
    """Resolve a ball impact against an effectively massive moving racket.

    The normal points from the racket toward the desired outgoing half-court.
    Spin and finite racket inertia enter in a later calibrated model.
    """
    config = config or RacketImpactConfig()
    ball_velocity = np.asarray(ball_velocity_m_s, dtype=np.float64).reshape(3)
    racket_velocity = np.asarray(racket_velocity_m_s, dtype=np.float64).reshape(3)
    normal = np.asarray(racket_normal, dtype=np.float64).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9:
        raise ValueError("racket_normal must be non-zero")
    normal = normal / normal_norm

    relative = ball_velocity - racket_velocity
    incoming_normal_speed = float(relative @ normal)
    if incoming_normal_speed >= 0.0:
        raise ValueError("ball is not approaching the racket's front face")
    tangential = relative - incoming_normal_speed * normal
    outgoing_relative = (
        config.tangential_speed_retention * tangential
        - config.normal_restitution * incoming_normal_speed * normal
    )
    return racket_velocity + outgoing_relative
