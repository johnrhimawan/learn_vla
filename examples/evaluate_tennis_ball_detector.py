"""Evaluate a learned heatmap detector on a tennis-flight-v0 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.learned_perception import LearnedBallDetector
from tennis_vla.perception_evaluation import evaluate_stereo_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence-threshold", type=float, default=0.20)
    args = parser.parse_args()
    detector = LearnedBallDetector.load(
        args.checkpoint,
        device_name=args.device,
        confidence_threshold=args.confidence_threshold,
    )
    report = evaluate_stereo_detector(
        args.dataset,
        detector,
        benchmark_name="learned-heatmap-stereo-v0",
        detector_source={
            "type": "BallHeatmapDetector",
            "checkpoint": str(args.checkpoint),
            "confidence_threshold": args.confidence_threshold,
        },
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(json.dumps(report["m1_gate_check"], indent=2))
    if args.output is not None:
        print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
