"""Plan a nonzero-velocity legal return for the canonical tennis feed."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.ballistics import simulate_ball_flight
from tennis_vla.environment import make_tennis_contact_model
from tennis_vla.strike import StrikeSearchConfig, plan_safe_center_strikes


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
        default=Path("results/tennis/canonical_strike_plan_v0.json"),
    )
    args = parser.parse_args()
    flight = simulate_ball_flight(CANONICAL_POSITION_M, CANONICAL_VELOCITY_M_S)
    search = StrikeSearchConfig()
    plans = plan_safe_center_strikes(
        make_tennis_contact_model(),
        flight,
        search=search,
        seed=40_001,
    )
    if not plans:
        raise SystemExit("canonical feed has no legal nonzero-velocity strike")
    selected = plans[0]
    report = {
        "schema_version": 1,
        "plan": "canonical-safe-center-active-strike-v0",
        "source": repository_state(),
        "incoming_feed": {
            "position_m": CANONICAL_POSITION_M.tolist(),
            "velocity_m_s": CANONICAL_VELOCITY_M_S.tolist(),
            "first_bounce_m": flight.first_bounce_m.tolist(),
        },
        "search": {
            "face_pitch_degrees": list(search.face_pitch_degrees),
            "racket_normal_speeds_m_s": list(search.racket_normal_speeds_m_s),
            "landing_target_xy_m": list(search.landing_target_xy_m),
            "minimum_net_clearance_m": search.minimum_net_clearance_m,
            "feasible_legal_plans": len(plans),
        },
        "selected": selected.metrics(),
        "limitations": [
            "This plan uses exact ball state and the first-order impact model.",
            "The trajectory has not yet been tracked while the MuJoCo ball is live.",
            "The trajectory ends at contact and does not include follow-through.",
            "The canonical feed is a single condition, not a held-out feed envelope.",
            "Spin and calibrated string-bed response are not modeled.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "feasible_legal_plans": len(plans),
                "face_pitch_degrees": selected.face_pitch_degrees,
                "racket_normal_speed_m_s": (
                    selected.requested_racket_normal_speed_m_s
                ),
                "net_clearance_m": selected.predicted_return.net_clearance_m,
                "first_bounce_m": selected.predicted_return.first_bounce_m.tolist(),
                "maximum_joint_speed_rad_s": (
                    selected.trajectory_bounds.maximum_joint_speed_rad_s
                ),
                "maximum_joint_acceleration_rad_s2": (
                    selected.trajectory_bounds.maximum_joint_acceleration_rad_s2
                ),
            },
            indent=2,
        )
    )
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
