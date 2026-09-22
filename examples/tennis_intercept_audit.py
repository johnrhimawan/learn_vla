"""Audit privileged kinematic intercept coverage over seeded tennis feeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.environment import make_tennis_contact_model
from tennis_vla.feeder import ProgrammableFeeder
from tennis_vla.intercept import RacketIKConfig, find_kinematic_intercepts


def repository_state() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"git_revision": revision, "tracked_files_dirty": dirty}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tennis/intercept_kinematic_audit_v0.json"),
    )
    args = parser.parse_args()
    if args.seeds < 1:
        raise SystemExit("--seeds must be positive")

    model = make_tennis_contact_model()
    feeder = ProgrammableFeeder()
    ik_config = RacketIKConfig(restarts=4, maximum_iterations=180)
    episodes = []
    reachable_positions = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        feed, flight = feeder.sample_legal_feed(seed)
        candidates = find_kinematic_intercepts(
            model,
            flight,
            maximum_candidates=1,
            ik_config=ik_config,
            seed=10_000 + seed,
        )
        candidate = candidates[0] if candidates else None
        if candidate is not None:
            reachable_positions.append(candidate.ball_position_m)
        episodes.append(
            {
                "seed": seed,
                "feed_attempt": feed.attempt,
                "reachable": candidate is not None,
                "first_intercept": None
                if candidate is None
                else candidate.metrics(),
            }
        )

    positions = np.asarray(reachable_positions)
    reachable_count = len(reachable_positions)
    report = {
        "schema_version": 1,
        "audit": "privileged-kinematic-intercept-v0",
        "source": repository_state(),
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "reachable_feeds": reachable_count,
        "reachable_fraction": reachable_count / args.seeds,
        "first_intercept_bounds_m": None
        if not reachable_count
        else {
            "minimum": positions.min(axis=0).tolist(),
            "maximum": positions.max(axis=0).tolist(),
        },
        "perception_reference_plane_x_m": -9.25,
        "search_region": {
            "x_m": [-10.1, -9.45],
            "absolute_y_max_m": 1.1,
            "z_m": [0.15, 1.35],
            "post_bounce_only": True,
        },
        "ik_config": {
            "position_tolerance_m": ik_config.position_tolerance_m,
            "normal_tolerance_deg": ik_config.normal_tolerance_deg,
            "joint_limit_margin_rad": ik_config.joint_limit_margin_rad,
            "restarts": ik_config.restarts,
            "maximum_iterations": ik_config.maximum_iterations,
        },
        "episodes": episodes,
        "limitations": [
            "Reachability is kinematic and does not include joint speed or acceleration.",
            "The broad perception feeder is not yet restricted to the arm workspace.",
            "Collision-free swing and outgoing-ball placement are not evaluated.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "reachable_feeds": reachable_count,
                "seeds": args.seeds,
                "reachable_fraction": report["reachable_fraction"],
                "first_intercept_bounds_m": report["first_intercept_bounds_m"],
            },
            indent=2,
        )
    )
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
