"""Audit planned and executed active strikes on held-out feeder seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.environment import make_tennis_contact_model
from tennis_vla.execution import (
    StrikeExecutionConfig,
    execute_strike,
    simulate_mujoco_ball_flight,
)
from tennis_vla.feeder import PHASE_ONE_CONTACT_ENVELOPE, ProgrammableFeeder
from tennis_vla.strike import plan_safe_center_strikes


SAFETY_FAILURES = {
    "joint_tracking",
    "contact_joint_velocity",
    "actual_joint_speed",
    "actual_joint_acceleration",
    "joint_limit_margin",
    "clipped_control",
    "unexpected_contact",
    "recovery_no_feasible_plan",
    "recovery_joint_tracking",
    "recovery_final_joint_error",
    "recovery_actual_joint_speed",
    "recovery_actual_joint_acceleration",
    "recovery_joint_limit_margin",
    "recovery_clipped_control",
    "recovery_unexpected_contact",
}


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


def audit_seed(seed: int) -> dict[str, Any]:
    model = make_tennis_contact_model()
    feed, analytical_flight = ProgrammableFeeder(
        PHASE_ONE_CONTACT_ENVELOPE
    ).sample_legal_feed(seed)
    flight = simulate_mujoco_ball_flight(
        model,
        feed.position_m,
        feed.velocity_m_s,
    )
    plans = plan_safe_center_strikes(model, flight, seed=40_000 + seed)
    result: dict[str, Any] = {
        "seed": seed,
        "feed": {
            "attempt": feed.attempt,
            "position_m": feed.position_m.tolist(),
            "velocity_m_s": feed.velocity_m_s.tolist(),
            "analytical_first_bounce_m": (
                analytical_flight.first_bounce_m.tolist()
            ),
            "mujoco_first_bounce_m": flight.first_bounce_m.tolist(),
        },
        "planned": bool(plans),
        "feasible_plan_count": len(plans),
        "selected_plan": None,
        "execution": None,
        "contacted": False,
        "legal_return": False,
        "recovered": False,
        "controller_safe": True,
        "strict_pass": False,
    }
    if not plans:
        return result

    plan = plans[0]
    execution = execute_strike(
        model,
        plan,
        feed.position_m,
        feed.velocity_m_s,
    )
    execution_metrics = execution.metrics()
    result["selected_plan"] = {
        "intercept_time_s": plan.candidate.time_s,
        "contact_position_m": plan.candidate.ball_position_m.tolist(),
        "face_pitch_degrees": plan.face_pitch_degrees,
        "racket_normal_speed_m_s": plan.requested_racket_normal_speed_m_s,
        "racket_tangent_ratio": plan.requested_racket_tangent_ratio,
        "racket_tangent_speed_m_s": plan.requested_racket_tangent_speed_m_s,
        "predicted_first_bounce_m": (
            plan.predicted_return.first_bounce_m.tolist()
        ),
        "predicted_net_clearance_m": plan.predicted_return.net_clearance_m,
        "landing_error_m": plan.landing_error_m,
        "maximum_joint_speed_rad_s": (
            plan.trajectory_bounds.maximum_joint_speed_rad_s
        ),
        "maximum_joint_acceleration_rad_s2": (
            plan.trajectory_bounds.maximum_joint_acceleration_rad_s2
        ),
        "minimum_joint_limit_margin_rad": (
            plan.trajectory_bounds.minimum_joint_limit_margin_rad
        ),
    }
    result["execution"] = execution_metrics
    result["contacted"] = execution.contacted
    result["legal_return"] = bool(
        execution.measured_return is not None
        and execution.measured_return.legal_first_bounce
    )
    result["recovered"] = bool(
        execution.recovery is not None and execution.recovery.passed
    )
    result["controller_safe"] = not bool(
        SAFETY_FAILURES.intersection(execution.failure_reasons)
    )
    result["strict_pass"] = execution.passed
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("development", "expansion-development", "final-heldout"),
        default="development",
    )
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profiles = {
        "development": {
            "seed_start": 9_000,
            "count": 20,
            "output": "results/tennis/active_strike_development_v1.json",
            "audit": "phase-one-active-strike-development-v1",
            "split_name": "test-development-prefix",
            "gate_name": "development_gate",
            "limitation": (
                "This development prefix is smaller than the 200-feed "
                "held-out gate."
            ),
        },
        "expansion-development": {
            "seed_start": 10_000,
            "count": 200,
            "output": (
                "results/tennis/active_strike_v1_development_baseline_v0.json"
            ),
            "audit": "phase-one-active-strike-v1-development-baseline-v0",
            "split_name": "v1-expansion-development",
            "gate_name": "development_gate",
            "limitation": (
                "This 200-feed v1 development split may be used for tuning."
            ),
        },
        "final-heldout": {
            "seed_start": 12_000,
            "count": 200,
            "output": "results/tennis/active_strike_heldout_v0.json",
            "audit": "phase-one-active-strike-heldout-v0",
            "split_name": "final-heldout",
            "gate_name": "heldout_gate",
            "limitation": "This is the reserved 200-feed final simulation split.",
        },
    }
    profile = profiles[args.split]
    if args.seed_start is None:
        args.seed_start = profile["seed_start"]
    if args.count is None:
        args.count = profile["count"]
    if args.output is None:
        args.output = Path(profile["output"])
    if args.count < 1 or args.workers < 1:
        raise SystemExit("count and workers must be positive")
    audit_name = profile["audit"]
    split_name = profile["split_name"]
    gate_name = profile["gate_name"]
    split_limitation = profile["limitation"]

    source = repository_state()
    seeds = range(args.seed_start, args.seed_start + args.count)
    episodes = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(audit_seed, seed): seed for seed in seeds}
        for future in as_completed(futures):
            episode = future.result()
            episodes.append(episode)
            print(
                json.dumps(
                    {
                        "seed": episode["seed"],
                        "planned": episode["planned"],
                        "contacted": episode["contacted"],
                        "legal_return": episode["legal_return"],
                        "recovered": episode["recovered"],
                        "controller_safe": episode["controller_safe"],
                        "strict_pass": episode["strict_pass"],
                    }
                ),
                flush=True,
            )
    episodes.sort(key=lambda episode: episode["seed"])

    planned = sum(episode["planned"] for episode in episodes)
    contacted = sum(episode["contacted"] for episode in episodes)
    legal_returns = sum(episode["legal_return"] for episode in episodes)
    recovered = sum(episode["recovered"] for episode in episodes)
    controller_safe = sum(episode["controller_safe"] for episode in episodes)
    strict_passes = sum(episode["strict_pass"] for episode in episodes)
    execution_failures: Counter[str] = Counter()
    for episode in episodes:
        if episode["execution"] is not None:
            execution_failures.update(episode["execution"]["failure_reasons"])

    def execution_values(key: str) -> list[float]:
        return [
            float(episode["execution"][key])
            for episode in episodes
            if episode["execution"] is not None
            and episode["execution"][key] is not None
        ]

    report = {
        "schema_version": 2,
        "audit": audit_name,
        "source": source,
        "split": {
            "name": split_name,
            "seed_start": args.seed_start,
            "feeds": args.count,
            "workers": args.workers,
        },
        "envelope": {
            key: list(value)
            for key, value in PHASE_ONE_CONTACT_ENVELOPE.__dict__.items()
        },
        "execution_thresholds": StrikeExecutionConfig().__dict__,
        "summary": {
            "planned": planned,
            "planned_fraction": planned / args.count,
            "contacted": contacted,
            "contact_fraction": contacted / args.count,
            "legal_returns": legal_returns,
            "legal_return_fraction": legal_returns / args.count,
            "recovered": recovered,
            "recovery_fraction": recovered / args.count,
            "controller_safe": controller_safe,
            "controller_safe_fraction": controller_safe / args.count,
            "strict_passes": strict_passes,
            "strict_pass_fraction": strict_passes / args.count,
            "execution_failure_reason_counts": dict(execution_failures),
            "contact_time_error_s": distribution(
                execution_values("contact_time_error_s")
            ),
            "contact_position_error_m": distribution(
                execution_values("contact_position_error_m")
            ),
            "outgoing_velocity_error_m_s": distribution(
                execution_values("outgoing_velocity_error_m_s")
            ),
            "maximum_joint_tracking_error_rad": distribution(
                execution_values("maximum_joint_tracking_error_rad")
            ),
            "recovery_duration_s": distribution(
                [
                    float(episode["execution"]["recovery"]["duration_s"])
                    for episode in episodes
                    if episode["execution"] is not None
                    and episode["execution"]["recovery"] is not None
                    and episode["execution"]["recovery"]["duration_s"]
                    is not None
                ]
            ),
            "recovery_final_joint_error_rad": distribution(
                [
                    float(
                        episode["execution"]["recovery"][
                            "final_joint_error_rad"
                        ]
                    )
                    for episode in episodes
                    if episode["execution"] is not None
                    and episode["execution"]["recovery"] is not None
                    and episode["execution"]["recovery"][
                        "final_joint_error_rad"
                    ]
                    is not None
                ]
            ),
            "recovery_maximum_actual_joint_acceleration_rad_s2": distribution(
                [
                    float(
                        episode["execution"]["recovery"][
                            "maximum_actual_joint_acceleration_rad_s2"
                        ]
                    )
                    for episode in episodes
                    if episode["execution"] is not None
                    and episode["execution"]["recovery"] is not None
                    and episode["execution"]["recovery"][
                        "maximum_actual_joint_acceleration_rad_s2"
                    ]
                    is not None
                ]
            ),
            "measured_net_clearance_m": distribution(
                [
                    float(episode["execution"]["measured_return"]["net_clearance_m"])
                    for episode in episodes
                    if episode["execution"] is not None
                    and episode["execution"]["measured_return"] is not None
                    and episode["execution"]["measured_return"]["net_clearance_m"]
                    is not None
                ]
            ),
        },
        gate_name: {
            "minimum_contact_fraction": 0.95,
            "minimum_legal_return_fraction": 0.95,
            "minimum_recovery_fraction": 0.95,
            "require_zero_controller_safety_failures": True,
            "passed": (
                contacted / args.count >= 0.95
                and legal_returns / args.count >= 0.95
                and recovered / args.count >= 0.95
                and controller_safe == args.count
            ),
        },
        "episodes": episodes,
        "limitations": [
            split_limitation,
            (
                "Planning and control use exact MuJoCo ball state and inverse "
                "dynamics."
            ),
            (
                "The first-order impact-model check is reported separately "
                "from legal return."
            ),
            (
                "Recovery uses privileged joint state and returns to one fixed "
                "ready pose."
            ),
            "Spin and calibrated string-bed response are not modeled.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "gate": report[gate_name],
            },
            indent=2,
        )
    )
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
