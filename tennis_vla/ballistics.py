"""Deterministic tennis-ball flight used to validate the simulator foundation.

This model deliberately starts with gravity, quadratic drag, and a simple court
bounce. Spin-dependent lift and racket-string deformation are later calibration
milestones; their parameters should not be guessed into the training simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BallFlightConfig:
    # Midpoints of the 2026 ITF Type-2 ranges: 56.0-59.4 g and 6.54-6.86 cm.
    mass_kg: float = 0.0577
    radius_m: float = 0.0335
    air_density_kg_m3: float = 1.225
    drag_coefficient: float = 0.55
    gravity_m_s2: float = 9.81
    vertical_restitution: float = 0.74
    horizontal_speed_retention: float = 0.82
    dt_s: float = 0.001
    duration_s: float = 2.0


@dataclass(frozen=True)
class BallFlightResult:
    times_s: np.ndarray
    positions_m: np.ndarray
    velocities_m_s: np.ndarray
    net_crossing_m: np.ndarray | None
    first_bounce_m: np.ndarray | None
    bounce_count: int

    def metrics(self) -> dict[str, Any]:
        return {
            "duration_s": round(float(self.times_s[-1]), 6),
            "samples": int(len(self.times_s)),
            "net_crossing_m": None
            if self.net_crossing_m is None
            else [round(float(value), 6) for value in self.net_crossing_m],
            "first_bounce_m": None
            if self.first_bounce_m is None
            else [round(float(value), 6) for value in self.first_bounce_m],
            "bounce_count": self.bounce_count,
            "maximum_height_m": round(float(self.positions_m[:, 2].max()), 6),
            "final_speed_m_s": round(
                float(np.linalg.norm(self.velocities_m_s[-1])), 6
            ),
        }


def _acceleration(velocity: np.ndarray, config: BallFlightConfig) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    area = np.pi * config.radius_m**2
    drag_scale = (
        0.5
        * config.air_density_kg_m3
        * config.drag_coefficient
        * area
        / config.mass_kg
    )
    acceleration = -drag_scale * speed * velocity
    acceleration[2] -= config.gravity_m_s2
    return acceleration


def _rk4_step(
    position: np.ndarray,
    velocity: np.ndarray,
    dt: float,
    config: BallFlightConfig,
) -> tuple[np.ndarray, np.ndarray]:
    def derivative(state: np.ndarray) -> np.ndarray:
        return np.concatenate((state[3:], _acceleration(state[3:], config)))

    state = np.concatenate((position, velocity))
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    next_state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return next_state[:3], next_state[3:]


def simulate_ball_flight(
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    config: BallFlightConfig | None = None,
) -> BallFlightResult:
    """Simulate a ball and report its net crossing and first court bounce.

    Coordinates follow the project convention: court length is the x-axis, the
    net is x=0, court width is y, and height is z.
    """
    config = config or BallFlightConfig()
    position = np.asarray(position_m, dtype=np.float64).reshape(3).copy()
    velocity = np.asarray(velocity_m_s, dtype=np.float64).reshape(3).copy()
    steps = int(round(config.duration_s / config.dt_s))
    times = np.linspace(0.0, config.duration_s, steps + 1)
    positions = np.empty((steps + 1, 3), dtype=np.float64)
    velocities = np.empty((steps + 1, 3), dtype=np.float64)
    positions[0], velocities[0] = position, velocity
    net_crossing: np.ndarray | None = None
    first_bounce: np.ndarray | None = None
    bounce_count = 0

    for index in range(1, steps + 1):
        previous_position = position.copy()
        position, velocity = _rk4_step(position, velocity, config.dt_s, config)

        if net_crossing is None and previous_position[0] * position[0] <= 0.0:
            dx = position[0] - previous_position[0]
            fraction = 0.0 if abs(dx) < 1e-12 else -previous_position[0] / dx
            net_crossing = previous_position + fraction * (position - previous_position)

        if position[2] <= config.radius_m and velocity[2] < 0.0:
            position[2] = config.radius_m
            velocity[2] *= -config.vertical_restitution
            velocity[:2] *= config.horizontal_speed_retention
            bounce_count += 1
            if first_bounce is None:
                first_bounce = position.copy()

        positions[index], velocities[index] = position, velocity

    return BallFlightResult(
        times_s=times,
        positions_m=positions,
        velocities_m_s=velocities,
        net_crossing_m=net_crossing,
        first_bounce_m=first_bounce,
        bounce_count=bounce_count,
    )
