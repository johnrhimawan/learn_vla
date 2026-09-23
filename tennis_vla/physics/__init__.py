"""Ball flight, court geometry, and racket impact.

No dependency on the robot or the scene: this layer is the reference physics
the rest of the project is calibrated against."""

from __future__ import annotations

from .court import (
    TennisCourtSpec,
    court_scene_xml,
)
from .ballistics import (
    BallFlightConfig,
    BallFlightResult,
    simulate_ball_flight,
)
from .impact import (
    RacketImpactConfig,
    apply_racket_impact,
)

__all__ = [
    "BallFlightConfig",
    "BallFlightResult",
    "RacketImpactConfig",
    "TennisCourtSpec",
    "apply_racket_impact",
    "court_scene_xml",
    "simulate_ball_flight",
]
