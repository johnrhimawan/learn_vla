"""Train the M1 heatmap ball detector on a tennis-flight-v0 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.learned_perception import train_ball_heatmap_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = train_ball_heatmap_detector(
        args.dataset,
        args.checkpoint,
        report_path=args.report,
        image_size=(args.width, args.height),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        channels=args.channels,
        device_name=args.device,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "train_images": report["train_images"],
                "validation_images": report["validation_images"],
                "best_validation_pixel_rmse": report[
                    "best_validation_pixel_rmse"
                ],
                "checkpoint": str(args.checkpoint.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
