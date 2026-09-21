"""Validate integrated MuJoCo ball/racket contact against the impact model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.environment import probe_stationary_racket_contact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tennis/racket_contact_probe_v0.json"),
    )
    args = parser.parse_args()
    probe = probe_stationary_racket_contact(args.speed)
    report = probe.metrics()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={args.output.resolve()}")
    if not probe.contacted or probe.outgoing_normal_speed_m_s is None:
        raise SystemExit("ball did not complete a racket contact")


if __name__ == "__main__":
    main()
