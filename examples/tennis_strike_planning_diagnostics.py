"""Classify planner rejection stages on failed development feeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.environment import make_tennis_contact_model
from tennis_vla.execution import simulate_mujoco_ball_flight
from tennis_vla.feeder import PHASE_ONE_CONTACT_ENVELOPE, ProgrammableFeeder
from tennis_vla.strike import plan_safe_center_strikes


PLANNING_STAGES = (
    "kinematic_candidates",
    "joint_velocity_directions",
    "impact_approach_candidates",
    "approach_motion_feasible",
    "approach_execution_safe",
    "recovery_feasible",
    "coarse_legal_returns",
    "full_legal_returns",
)


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


def terminal_rejection_stage(diagnostics: dict[str, int | bool]) -> str:
    for stage in PLANNING_STAGES:
        if int(diagnostics[stage]) == 0:
            return stage
    return "plan_returned"


def diagnose_seed(seed: int) -> dict[str, Any]:
    model = make_tennis_contact_model()
    feed, _ = ProgrammableFeeder(PHASE_ONE_CONTACT_ENVELOPE).sample_legal_feed(seed)
    flight = simulate_mujoco_ball_flight(
        model,
        feed.position_m,
        feed.velocity_m_s,
    )
    diagnostics: dict[str, int | bool] = {}
    started = time.perf_counter()
    plans = plan_safe_center_strikes(
        model,
        flight,
        seed=40_000 + seed,
        diagnostics=diagnostics,
    )
    duration_s = time.perf_counter() - started
    return {
        "seed": seed,
        "feed_position_m": feed.position_m.tolist(),
        "feed_velocity_m_s": feed.velocity_m_s.tolist(),
        "plans": len(plans),
        "terminal_rejection_stage": terminal_rejection_stage(diagnostics),
        "planning_duration_s": duration_s,
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/tennis/active_strike_v1_development_baseline_v0.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/tennis/active_strike_v1_no_plan_diagnostics_v0.json"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")

    baseline = json.loads(args.input.read_text(encoding="utf-8"))
    seeds = [
        int(episode["seed"])
        for episode in baseline["episodes"]
        if not episode["planned"]
    ]
    episodes = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(diagnose_seed, seed): seed for seed in seeds}
        for future in as_completed(futures):
            episode = future.result()
            episodes.append(episode)
            print(
                json.dumps(
                    {
                        "seed": episode["seed"],
                        "stage": episode["terminal_rejection_stage"],
                        "duration_s": episode["planning_duration_s"],
                    }
                ),
                flush=True,
            )
    episodes.sort(key=lambda episode: episode["seed"])
    stages = Counter(
        episode["terminal_rejection_stage"] for episode in episodes
    )
    durations = [float(episode["planning_duration_s"]) for episode in episodes]
    report = {
        "schema_version": 1,
        "audit": "active-strike-v1-no-plan-diagnostics-v0",
        "source": repository_state(),
        "baseline": str(args.input),
        "feeds": len(episodes),
        "terminal_rejection_stage_counts": dict(stages),
        "planning_duration_s": {
            "minimum": min(durations),
            "mean": sum(durations) / len(durations),
            "maximum": max(durations),
        },
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report | {"episodes": f"{len(episodes)} omitted"}, indent=2))
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
