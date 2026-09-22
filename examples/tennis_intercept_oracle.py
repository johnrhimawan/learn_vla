"""Find privileged post-bounce racket intercepts for one seeded tennis feed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.environment import make_tennis_contact_model
from tennis_vla.feeder import ProgrammableFeeder
from tennis_vla.intercept import find_kinematic_intercepts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    feed, flight = ProgrammableFeeder().sample_legal_feed(args.seed)
    candidates = find_kinematic_intercepts(make_tennis_contact_model(), flight)
    report = {
        "schema_version": 1,
        "feed_seed": feed.seed,
        "feed_initial_position_m": feed.position_m.tolist(),
        "feed_initial_velocity_m_s": feed.velocity_m_s.tolist(),
        "candidate_count": len(candidates),
        "candidates": [candidate.metrics() for candidate in candidates],
        "limitations": [
            "This is a kinematic pose oracle; joint speed and acceleration are not yet checked.",
            "The desired racket normal is fixed toward the far side.",
            "The ball and racket centers include a first-order contact offset.",
        ],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not candidates:
        raise SystemExit("no kinematic intercept found")


if __name__ == "__main__":
    main()
