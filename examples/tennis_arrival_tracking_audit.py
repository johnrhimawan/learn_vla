"""Audit simulated 250 Hz tracking of privileged tennis arrivals."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.arm import tennis_ready_configuration
from tennis_vla.environment import make_tennis_contact_model
from tennis_vla.feeder import PHASE_ONE_CONTACT_ENVELOPE, ProgrammableFeeder
from tennis_vla.intercept import RacketIKConfig
from tennis_vla.trajectory import (
    ArrivalTrackingConfig,
    SimulationJointMotionLimits,
    earliest_feasible_arrival,
    plan_intercept_arrivals,
    track_intercept_arrival,
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
    motion_limits = SimulationJointMotionLimits()
    tracking_config = ArrivalTrackingConfig()
    split_reports = {}

    for split, (seed_start, count) in SPLITS.items():
        kinematic_seeds = []
        dynamic_seeds = []
        tracking_pass_seeds = []
        failure_reasons: Counter[str] = Counter()
        metrics: dict[str, list[float]] = {
            "maximum_joint_tracking_error_rad": [],
            "p95_joint_tracking_error_rad": [],
            "final_joint_tracking_error_rad": [],
            "maximum_actual_joint_speed_rad_s": [],
            "maximum_actual_joint_acceleration_rad_s2": [],
            "maximum_inverse_dynamics_torque_nm": [],
            "minimum_actual_joint_limit_margin_rad": [],
            "racket_position_tracking_error_m": [],
            "racket_normal_tracking_error_deg": [],
            "final_contact_position_error_m": [],
            "final_contact_normal_error_deg": [],
        }
        total_unexpected_contact_steps = 0
        total_clipped_servo_commands = 0

        for seed in range(seed_start, seed_start + count):
            _, flight = feeder.sample_legal_feed(seed)
            plans = plan_intercept_arrivals(
                model,
                flight,
                planning_start_time_s=0.0,
                limits=motion_limits,
                ik_config=ik_config,
                seed=40_000 + seed,
            )
            if plans:
                kinematic_seeds.append(seed)
            selected = earliest_feasible_arrival(plans)
            if selected is None:
                continue
            dynamic_seeds.append(seed)
            tracking = track_intercept_arrival(
                model,
                selected,
                config=tracking_config,
            )
            failure_reasons.update(tracking.failure_reasons)
            total_unexpected_contact_steps += tracking.unexpected_contact_steps
            total_clipped_servo_commands += tracking.clipped_servo_commands
            for name in metrics:
                metrics[name].append(float(getattr(tracking, name)))
            if tracking.passed:
                tracking_pass_seeds.append(seed)

        all_seeds = set(range(seed_start, seed_start + count))
        kinematic_set = set(kinematic_seeds)
        dynamic_set = set(dynamic_seeds)
        tracking_set = set(tracking_pass_seeds)
        split_reports[split] = {
            "seed_start": seed_start,
            "feeds": count,
            "kinematically_reachable": len(kinematic_seeds),
            "kinematically_reachable_fraction": len(kinematic_seeds) / count,
            "dynamically_feasible": len(dynamic_seeds),
            "dynamically_feasible_fraction": len(dynamic_seeds) / count,
            "tracking_passed": len(tracking_pass_seeds),
            "tracking_pass_fraction": len(tracking_pass_seeds) / count,
            "kinematically_unreachable_seeds": sorted(all_seeds - kinematic_set),
            "dynamically_infeasible_seeds": sorted(
                kinematic_set - dynamic_set
            ),
            "tracking_failed_seeds": sorted(dynamic_set - tracking_set),
            "tracking_failure_reason_counts": dict(failure_reasons),
            "total_unexpected_contact_steps": total_unexpected_contact_steps,
            "total_clipped_servo_commands": total_clipped_servo_commands,
            "tracking_metrics": {
                name: distribution(values) for name, values in metrics.items()
            },
        }

    report = {
        "schema_version": 1,
        "audit": "phase-one-contact-arrival-tracking-v0",
        "source": repository_state(),
        "planning_start": {
            "time_s": 0.0,
            "event": "programmable feeder launch trigger",
        },
        "start_joint_positions_rad": tennis_ready_configuration(model).tolist(),
        "contact_search": {
            "x_m": [-10.1, -9.45],
            "absolute_y_max_m": 1.1,
            "z_m": [0.60, 1.35],
        },
        "simulation_joint_motion_limits": asdict(motion_limits),
        "tracking_config": asdict(tracking_config),
        "controller": {
            "physics_rate_hz": 1.0 / float(model.opt.timestep),
            "servo_rate_hz": tracking_config.servo_rate_hz,
            "position_command": "minimum jerk plus actuator damping compensation",
            "feedforward": "unconstrained MuJoCo rigid-body inverse dynamics",
        },
        "splits": split_reports,
        "gate": {
            "minimum_test_tracking_pass_fraction": 0.95,
            "test_pass": split_reports["test"]["tracking_pass_fraction"] >= 0.95,
        },
        "limitations": [
            "The privileged oracle plans from the known feeder trigger and exact ball state.",
            "The ball is parked away from the arm, so this audit does not execute contact.",
            "The arrival ends with zero racket velocity and is not an active strike.",
            "No authoritative actuator torque limit is available for the reference model.",
            "Simulation motion and tracking thresholds must be replaced before hardware transfer.",
        ],
    }
    output = Path("results/tennis/arrival_tracking_audit_v0.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "splits": {
                    name: {
                        "kinematic": values["kinematically_reachable_fraction"],
                        "dynamic": values["dynamically_feasible_fraction"],
                        "tracking": values["tracking_pass_fraction"],
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
