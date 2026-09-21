"""Seeded programmable-ball-machine feeds for curriculum generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ballistics import BallFlightConfig, BallFlightResult, simulate_ball_flight


@dataclass(frozen=True)
class Feed:
    seed: int
    attempt: int
    position_m: np.ndarray
    velocity_m_s: np.ndarray


@dataclass(frozen=True)
class FeedEnvelope:
    source_x_m: tuple[float, float] = (9.5, 11.0)
    source_y_m: tuple[float, float] = (-1.0, 1.0)
    source_z_m: tuple[float, float] = (1.2, 1.8)
    forward_speed_m_s: tuple[float, float] = (14.0, 20.0)
    lateral_speed_m_s: tuple[float, float] = (-2.0, 2.0)
    vertical_speed_m_s: tuple[float, float] = (3.5, 5.5)


class ProgrammableFeeder:
    def __init__(
        self,
        envelope: FeedEnvelope | None = None,
        flight_config: BallFlightConfig | None = None,
    ) -> None:
        self.envelope = envelope or FeedEnvelope()
        self.flight_config = flight_config or BallFlightConfig()

    def sample_legal_feed(
        self, seed: int, max_attempts: int = 128
    ) -> tuple[Feed, BallFlightResult]:
        """Rejection-sample a deterministic feed with a legal first bounce."""
        rng = np.random.default_rng(seed)
        for attempt in range(max_attempts):
            position = np.array(
                [
                    rng.uniform(*self.envelope.source_x_m),
                    rng.uniform(*self.envelope.source_y_m),
                    rng.uniform(*self.envelope.source_z_m),
                ]
            )
            velocity = np.array(
                [
                    -rng.uniform(*self.envelope.forward_speed_m_s),
                    rng.uniform(*self.envelope.lateral_speed_m_s),
                    rng.uniform(*self.envelope.vertical_speed_m_s),
                ]
            )
            result = simulate_ball_flight(position, velocity, self.flight_config)
            if result.legal_first_bounce:
                return Feed(seed, attempt, position, velocity), result
        raise RuntimeError(
            f"Could not produce a legal feed for seed {seed} in {max_attempts} attempts"
        )
