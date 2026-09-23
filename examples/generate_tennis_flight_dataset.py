"""Generate randomized stereo perception data for tennis VLA milestone M1.

Small local dataset:
    scripts/run python examples/generate_tennis_flight_dataset.py

Larger training dataset:
    scripts/run python examples/generate_tennis_flight_dataset.py \
      --train-episodes 1000 --validation-episodes 100 --test-episodes 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.perception import generate_flight_dataset, write_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/tennis-flight-v0")
    )
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--validation-episodes", type=int, default=3)
    parser.add_argument("--test-episodes", type=int, default=3)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--observation-hz", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    report = generate_flight_dataset(
        args.output,
        {
            "train": args.train_episodes,
            "validation": args.validation_episodes,
            "test": args.test_episodes,
        },
        width=args.width,
        height=args.height,
        observation_hz=args.observation_hz,
        overwrite=args.overwrite,
        preview_path=args.preview,
    )
    if args.summary is not None:
        write_summary(report, args.summary)
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "episodes": report["episodes"],
                "frames": report["frames"],
                "splits": report["splits"],
                "validation": report["validation"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
