"""Inspect the dataset written by examples/mujoco_reach.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--repo-id", default="local/mujoco-reach")
    args = parser.parse_args()

    dataset = LeRobotDataset(
        repo_id=args.repo_id, root=args.dataset, video_backend="pyav"
    )
    sample = dataset[0]
    print(f"episodes: {dataset.num_episodes}")
    print(f"frames:   {dataset.num_frames}")
    print(f"fps:      {dataset.fps}")
    print(f"task:     {sample['task']}")
    for key in sorted(
        key
        for key in sample
        if key == "action" or key.startswith("observation.")
    ):
        value = sample[key]
        print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")


if __name__ == "__main__":
    main()
