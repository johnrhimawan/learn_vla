"""Evaluate the fixed stereo baseline on a generated tennis-flight dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vla.perception import evaluate_color_stereo_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_color_stereo_baseline(args.dataset)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(json.dumps(report["m1_gate_check"], indent=2))
    if args.output is not None:
        print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
