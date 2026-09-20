"""Collect a SmolVLA-ready MuJoCo reaching dataset.

Unlike the introductory dataset, this one keeps privileged target coordinates
out of the robot state. The policy receives joint proprioception, an RGB image,
and a language instruction, so it has to use vision to locate the target.

Run from the repository root:

    scripts/run python examples/collect_smolvla_dataset.py --episodes 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mujoco_reach import ACTION_NAMES, ReachingSimulation, RolloutConfig, TASK


STATE_NAMES = [
    "shoulder_yaw.pos",
    "shoulder_pitch.pos",
    "elbow_pitch.pos",
    "shoulder_yaw.vel",
    "shoulder_pitch.vel",
    "elbow_pitch.vel",
]


def dataset_features(image_size: int) -> dict[str, dict]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        # This name matches one of the pretrained checkpoint's visual inputs.
        "observation.images.camera1": {
            "dtype": "image",
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": ACTION_NAMES,
        },
    }


def default_output() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("artifacts") / f"smolvla-mujoco-reach-{timestamp}"


def collect_dataset(
    output: Path,
    episodes: int,
    config: RolloutConfig,
) -> tuple[Path, list[float]]:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    simulation = ReachingSimulation(config)
    dataset = None
    final_distances: list[float] = []
    try:
        dataset = LeRobotDataset.create(
            repo_id="local/smolvla-mujoco-reach",
            root=output,
            fps=ReachingSimulation.control_hz,
            robot_type="mujoco-3dof-reacher",
            features=dataset_features(config.image_size),
            use_videos=False,
            video_backend="pyav",
        )

        for episode in range(episodes):
            simulation.reset()
            for _ in range(config.steps):
                full_state, image = simulation.observe()
                action = simulation.expert_action()
                dataset.add_frame(
                    {
                        # qpos + qvel only. Target coordinates remain privileged
                        # to the expert that generated the demonstration.
                        "observation.state": full_state[: len(STATE_NAMES)].astype(
                            np.float32
                        ),
                        "observation.images.camera1": image,
                        "action": action,
                        "task": TASK,
                    }
                )
                simulation.step(action)

            final_distance = simulation.distance_to_target
            final_distances.append(final_distance)
            dataset.save_episode()
            print(
                f"episode={episode:03d} frames={config.steps} "
                f"final_distance={final_distance:.4f} m"
            )
    finally:
        if dataset is not None:
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            dataset.finalize()
        simulation.close()

    return output, final_distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1:
        parser.error("--episodes and --steps must be positive")
    if args.image_size < 32:
        parser.error("--image-size must be at least 32")
    return args


def main() -> None:
    args = parse_args()
    output = args.output or default_output()
    path, distances = collect_dataset(
        output,
        args.episodes,
        RolloutConfig(
            steps=args.steps,
            image_size=args.image_size,
            seed=args.seed,
            viewer=False,
        ),
    )
    successes = sum(distance < 0.04 for distance in distances)
    print(f"saved={path.resolve()}")
    print(
        f"success={successes}/{len(distances)} "
        f"schema=RGB+6D-proprioception+language->3D-action "
        f"mujoco={mujoco.__version__}"
    )


if __name__ == "__main__":
    main()
