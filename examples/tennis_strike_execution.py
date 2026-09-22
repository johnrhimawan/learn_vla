"""Execute and audit the canonical nonzero-velocity tennis strike."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.ballistics import simulate_ball_flight
from tennis_vla.environment import make_tennis_contact_model
from tennis_vla.execution import (
    StrikeExecutionConfig,
    audit_court_bounce,
    execute_strike,
    simulate_mujoco_ball_flight,
)
from tennis_vla.strike import plan_safe_center_strikes


CANONICAL_POSITION_M = np.array([10.5, 0.0, 1.4])
CANONICAL_VELOCITY_M_S = np.array([-17.75, 0.0, 4.6])


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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tennis/canonical_strike_execution_v0.json"),
    )
    args = parser.parse_args()

    model = make_tennis_contact_model()
    analytical_flight = simulate_ball_flight(
        CANONICAL_POSITION_M,
        CANONICAL_VELOCITY_M_S,
    )
    flight = simulate_mujoco_ball_flight(
        model,
        CANONICAL_POSITION_M,
        CANONICAL_VELOCITY_M_S,
    )
    plans = plan_safe_center_strikes(model, flight, seed=40_001)
    if not plans:
        raise SystemExit("canonical feed has no legal nonzero-velocity strike")
    plan = plans[0]
    bounce = audit_court_bounce(
        model,
        CANONICAL_POSITION_M,
        CANONICAL_VELOCITY_M_S,
    )
    execution_config = StrikeExecutionConfig()
    execution = execute_strike(
        model,
        plan,
        CANONICAL_POSITION_M,
        CANONICAL_VELOCITY_M_S,
        config=execution_config,
    )
    report = {
        "schema_version": 1,
        "audit": "canonical-active-strike-execution-v0",
        "source": repository_state(),
        "incoming_feed": {
            "position_m": CANONICAL_POSITION_M.tolist(),
            "velocity_m_s": CANONICAL_VELOCITY_M_S.tolist(),
            "analytical_first_bounce_m": (
                analytical_flight.first_bounce_m.tolist()
            ),
            "mujoco_first_bounce_m": flight.first_bounce_m.tolist(),
        },
        "controller": asdict(execution_config),
        "court_bounce_calibration": bounce.metrics(),
        "analytical_strike_plan": plan.metrics(),
        "mujoco_execution": execution.metrics(),
        "passed": bounce.passed and execution.passed,
        "limitations": [
            "This is one canonical feed, not a held-out feed-envelope audit.",
            "The controller uses exact ball state and privileged inverse dynamics.",
            "The measured outgoing state is continued with the analytical flight model.",
            "The contact trajectory stops after separation and has no recovery motion.",
            (
                f"The {execution.maximum_contact_phase_joint_acceleration_rad_s2:.1f} "
                "rad/s^2 contact-phase peak has no hardware gate."
            ),
            "Spin and calibrated string-bed response are not modeled.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "bounce": bounce.metrics(),
                "execution": execution.metrics(),
            },
            indent=2,
        )
    )
    print(f"saved={args.output.resolve()}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
