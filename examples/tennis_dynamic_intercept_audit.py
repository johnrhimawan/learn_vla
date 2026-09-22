"""Audit minimum-jerk racket arrivals over the contact curriculum."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.arm import tennis_ready_configuration
from tennis_vla.environment import make_tennis_contact_model
from tennis_vla.feeder import PHASE_ONE_CONTACT_ENVELOPE, ProgrammableFeeder
from tennis_vla.intercept import RacketIKConfig
from tennis_vla.trajectory import (
    SimulationJointMotionLimits,
    earliest_feasible_arrival,
    plan_intercept_arrivals,
)


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


def distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def main() -> None:
    model = make_tennis_contact_model()
    feeder = ProgrammableFeeder(PHASE_ONE_CONTACT_ENVELOPE)
    ik_config = RacketIKConfig(restarts=4, maximum_iterations=180)
    limits = SimulationJointMotionLimits()
    split_reports = {}

    for split, (seed_start, count) in SPLITS.items():
        kinematic_seeds = []
        feasible_seeds = []
        positions = []
        durations = []
        speeds = []
        accelerations = []
        margins = []
        candidate_failures: Counter[str] = Counter()
        for seed in range(seed_start, seed_start + count):
            _, flight = feeder.sample_legal_feed(seed)
            plans = plan_intercept_arrivals(
                model,
                flight,
                # The feeder trigger defines time zero for this privileged oracle.
                planning_start_time_s=0.0,
                limits=limits,
                ik_config=ik_config,
                seed=40_000 + seed,
            )
            if plans:
                kinematic_seeds.append(seed)
            selected = earliest_feasible_arrival(plans)
            if selected is None:
                for plan in plans:
                    candidate_failures.update(plan.failure_reasons)
                continue
            feasible_seeds.append(seed)
            positions.append(selected.candidate.ball_position_m)
            durations.append(selected.duration_s)
            speeds.append(selected.maximum_joint_speed_rad_s)
            accelerations.append(selected.maximum_joint_acceleration_rad_s2)
            margins.append(selected.minimum_joint_limit_margin_rad)

        position_array = np.asarray(positions)
        split_reports[split] = {
            "seed_start": seed_start,
            "feeds": count,
            "kinematically_reachable": len(kinematic_seeds),
            "kinematically_reachable_fraction": len(kinematic_seeds) / count,
            "dynamically_feasible": len(feasible_seeds),
            "dynamically_feasible_fraction": len(feasible_seeds) / count,
            "kinematically_unreachable_seeds": sorted(
                set(range(seed_start, seed_start + count)) - set(kinematic_seeds)
            ),
            "dynamically_infeasible_seeds": sorted(
                set(kinematic_seeds) - set(feasible_seeds)
            ),
            "infeasible_candidate_reason_counts": dict(candidate_failures),
            "selected_intercept_bounds_m": None
            if not len(positions)
            else {
                "minimum": position_array.min(axis=0).tolist(),
                "maximum": position_array.max(axis=0).tolist(),
            },
            "arrival_duration_s": distribution(durations),
            "maximum_joint_speed_rad_s": distribution(speeds),
            "maximum_joint_acceleration_rad_s2": distribution(accelerations),
            "minimum_joint_limit_margin_rad": distribution(margins),
        }

    report = {
        "schema_version": 1,
        "audit": "phase-one-contact-minimum-jerk-arrival-v0",
        "source": repository_state(),
        "planning_start": {
            "time_s": 0.0,
            "event": "programmable feeder launch trigger",
        },
        "start_joint_positions_rad": tennis_ready_configuration(model).tolist(),
        "trajectory": {
            "type": "quintic minimum jerk",
            "initial_velocity_rad_s": 0.0,
            "terminal_velocity_rad_s": 0.0,
            "initial_acceleration_rad_s2": 0.0,
            "terminal_acceleration_rad_s2": 0.0,
        },
        "simulation_joint_motion_limits": {
            "maximum_speed_rad_s": limits.maximum_speed_rad_s,
            "maximum_acceleration_rad_s2": limits.maximum_acceleration_rad_s2,
            "minimum_joint_limit_margin_rad": (
                limits.minimum_joint_limit_margin_rad
            ),
            "authority": "project simulation contract; not hardware ratings",
        },
        "ik_config": {
            "position_tolerance_m": ik_config.position_tolerance_m,
            "normal_tolerance_deg": ik_config.normal_tolerance_deg,
            "joint_limit_margin_rad": ik_config.joint_limit_margin_rad,
            "restarts": ik_config.restarts,
            "maximum_iterations": ik_config.maximum_iterations,
        },
        "splits": split_reports,
        "gate": {
            "minimum_test_dynamically_feasible_fraction": 0.95,
            "test_pass": (
                split_reports["test"]["dynamically_feasible_fraction"] >= 0.95
            ),
        },
        "limitations": [
            "The oracle plans from the known feeder trigger with privileged ball state.",
            "The trajectory arrives with zero racket velocity; an active strike is not yet planned.",
            "Analytical speed and acceleration checks do not prove actuator tracking.",
            "Self-collision, ball contact, and outgoing-ball placement are not evaluated.",
            "The simulation limits must be replaced by authoritative hardware limits before transfer.",
        ],
    }
    output = Path("results/tennis/dynamic_intercept_audit_v0.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "splits": {
                    name: {
                        "kinematic": values["kinematically_reachable_fraction"],
                        "dynamic": values["dynamically_feasible_fraction"],
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
