"""Verify kinematic eligibility of the phase-one contact feeder curriculum."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.environment import make_tennis_contact_model
from tennis_vla.feeder import PHASE_ONE_CONTACT_ENVELOPE, ProgrammableFeeder
from tennis_vla.intercept import RacketIKConfig, find_kinematic_intercepts


SPLITS = {"train": (0, 200), "validation": (8000, 200), "test": (9000, 200)}


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
    model = make_tennis_contact_model()
    feeder = ProgrammableFeeder(PHASE_ONE_CONTACT_ENVELOPE)
    ik_config = RacketIKConfig(restarts=4, maximum_iterations=180)
    split_reports = {}
    all_positions = []
    total_reachable = 0
    total_feeds = 0

    for split, (seed_start, count) in SPLITS.items():
        failed_seeds = []
        positions = []
        for seed in range(seed_start, seed_start + count):
            _, flight = feeder.sample_legal_feed(seed)
            candidates = find_kinematic_intercepts(
                model,
                flight,
                maximum_candidates=1,
                ik_config=ik_config,
                seed=40_000 + seed,
            )
            if candidates:
                positions.append(candidates[0].ball_position_m)
                all_positions.append(candidates[0].ball_position_m)
            else:
                failed_seeds.append(seed)
        reachable = count - len(failed_seeds)
        total_reachable += reachable
        total_feeds += count
        position_array = np.asarray(positions)
        split_reports[split] = {
            "seed_start": seed_start,
            "feeds": count,
            "reachable": reachable,
            "reachable_fraction": reachable / count,
            "failed_seeds": failed_seeds,
            "first_intercept_bounds_m": {
                "minimum": position_array.min(axis=0).tolist(),
                "maximum": position_array.max(axis=0).tolist(),
            },
        }

    all_position_array = np.asarray(all_positions)
    report = {
        "schema_version": 1,
        "audit": "phase-one-contact-curriculum-kinematic-v0",
        "source": repository_state(),
        "envelope": {
            key: list(value)
            for key, value in PHASE_ONE_CONTACT_ENVELOPE.__dict__.items()
        },
        "ik_config": {
            "position_tolerance_m": ik_config.position_tolerance_m,
            "normal_tolerance_deg": ik_config.normal_tolerance_deg,
            "joint_limit_margin_rad": ik_config.joint_limit_margin_rad,
            "restarts": ik_config.restarts,
            "maximum_iterations": ik_config.maximum_iterations,
        },
        "splits": split_reports,
        "total_feeds": total_feeds,
        "total_reachable": total_reachable,
        "reachable_fraction": total_reachable / total_feeds,
        "first_intercept_bounds_m": {
            "minimum": all_position_array.min(axis=0).tolist(),
            "maximum": all_position_array.max(axis=0).tolist(),
        },
        "gate": {
            "minimum_test_reachable_fraction": 0.95,
            "test_pass": split_reports["test"]["reachable_fraction"] >= 0.95,
        },
        "limitations": [
            "This gate measures kinematic eligibility, not executed racket contact.",
            "Joint speed, acceleration, and collision constraints are not yet applied.",
            "The envelope is a phase-one curriculum and does not cover full-court feeds.",
        ],
    }
    output = Path("results/tennis/contact_curriculum_kinematic_audit_v0.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "splits": {
                    name: {
                        "reachable": values["reachable"],
                        "feeds": values["feeds"],
                        "reachable_fraction": values["reachable_fraction"],
                    }
                    for name, values in split_reports.items()
                },
                "gate": report["gate"],
            },
            indent=2,
        )
    )
    print(f"saved={output.resolve()}")


if __name__ == "__main__":
    main()
