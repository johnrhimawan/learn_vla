"""Simulation and learning components for the tennis VLA project.

The package is layered, and imports run in one direction only:

    physics, robot  ->  environment  ->  planning  ->  control
                                     \\-> perception

- ``physics``     ball flight, court geometry, racket impact; no MuJoCo scene
- ``robot``       the pinned Sawyer arm, its racket, and the optional base
- ``environment`` the MuJoCo scene and the programmable ball machine
- ``planning``    racket-pose IK, joint trajectories, strike search
- ``control``     execution of a planned strike against MuJoCo physics
- ``perception``  ball detection, dataset generation, evaluation

Import from the subpackage that owns the concept, for example
``from tennis_vla.planning import plan_safe_center_strikes``.  The names
re-exported here are only the small physics surface used across the project.
"""

from __future__ import annotations

from .environment import FeedEnvelope, ProgrammableFeeder
from .physics import (
    BallFlightConfig,
    BallFlightResult,
    RacketImpactConfig,
    TennisCourtSpec,
    apply_racket_impact,
    simulate_ball_flight,
)

__all__ = [
    "BallFlightConfig",
    "BallFlightResult",
    "FeedEnvelope",
    "ProgrammableFeeder",
    "RacketImpactConfig",
    "TennisCourtSpec",
    "apply_racket_impact",
    "simulate_ball_flight",
]
