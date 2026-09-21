"""Collect deterministic expert demonstrations for obstacle-reach RL warm start."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .environment import ObstacleReachEnv, dataset_features


def collect(output: Path, episodes: int, steps: int, image_size: int, seed: int) -> dict:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = ObstacleReachEnv(image_size=image_size, max_steps=steps)
    dataset = None
    episode_metrics: list[dict] = []
    try:
        dataset = LeRobotDataset.create(
            repo_id="local/obstacle-reach-rl-v0",
            root=output,
            fps=env.control_hz,
            robot_type="mujoco-3dof-obstacle-reacher",
            features=dataset_features(image_size),
            use_videos=False,
            video_backend="pyav",
        )
        for episode in range(episodes):
            episode_seed = seed + episode
            task_key = "red_fast" if episode % 2 == 0 else "green_smooth"
            observation, _ = env.reset(
                seed=episode_seed,
                options={
                    "task_key": task_key,
                    # Early enough that every successful demonstration also
                    # contains a recovery segment.
                    "disturbance_step": min(10, max(1, steps // 3)),
                },
            )
            episode_return = 0.0
            info = {}
            for _ in range(steps):
                action = env.expert_action()
                dataset.add_frame(
                    {
                        "observation.state": observation["observation.state"],
                        "observation.images.camera1": observation[
                            "observation.images.camera1"
                        ],
                        "action": action.astype(np.float32),
                        "task": observation["task"],
                    }
                )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            dataset.save_episode()
            metric = {
                "episode": episode,
                "seed": episode_seed,
                "task": task_key,
                "return": round(episode_return, 6),
                "success": bool(info["is_success"]),
                "collisions": int(info["collision_count"]),
                "disturbance_applied": bool(info["disturbance_applied"]),
                "final_distance": round(float(info["distance"]), 6),
            }
            episode_metrics.append(metric)
            print(json.dumps(metric, sort_keys=True))
    finally:
        if dataset is not None:
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            dataset.finalize()
        env.close()

    summary = {
        "dataset": str(output.resolve()),
        "episodes": episodes,
        "success_rate": float(np.mean([m["success"] for m in episode_metrics])),
        "collision_rate": float(np.mean([m["collisions"] > 0 for m in episode_metrics])),
        "episode_metrics": episode_metrics,
    }
    (output / "collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/obstacle-reach-seed-v0"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1100)
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1:
        parser.error("--episodes and --steps must be positive")
    summary = collect(args.output, args.episodes, args.steps, args.image_size, args.seed)
    print(json.dumps({key: value for key, value in summary.items() if key != "episode_metrics"}, indent=2))


if __name__ == "__main__":
    main()
