"""Evaluate the expert or immutable SmolVLA base on obstacle-reach-v0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from .environment import ObstacleReachEnv


MODEL_ID = "lerobot/smolvla_base"
MODEL_REVISION = "d9f33c94a60fb382c90dea2164c96845bd955e28"
MODEL_SHA256 = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"


def load_suite(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values), quantile))


def metadata_digest(dataset_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((dataset_root / "meta").rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(dataset_root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def make_smolvla_controller(dataset_root: Path, device: str, action_horizon: int):
    from huggingface_hub import snapshot_download
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    # Import registers the SmolVLA config and policy with LeRobot's factories.
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401

    checkpoint = Path(
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    )
    model_path = checkpoint / "model.safetensors"
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(f"Checkpoint hash mismatch: expected {MODEL_SHA256}, got {digest}")

    metadata = LeRobotDatasetMetadata(
        repo_id="local/obstacle-reach-rl-v0", root=dataset_root
    )
    config = PreTrainedConfig.from_pretrained(
        checkpoint,
        cli_overrides=[f"--device={device}", "--push_to_hub=false"],
    )
    # The checkpoint uses padded state/action projections, so its pretrained
    # weights can be evaluated with the local 6D-state/3D-action embodiment.
    # No task-specific gradient update occurs. Dataset statistics only calibrate
    # input and output units for this embodiment.
    config.pretrained_path = str(checkpoint)
    config.input_features = {}
    policy = make_policy(config, ds_meta=metadata)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        dataset_stats=metadata.stats,
        dataset_meta=metadata,
    )

    queued_actions: list[np.ndarray] = []
    inference_times: list[float] = []
    inference_calls = 0

    def reset() -> None:
        nonlocal queued_actions
        queued_actions = []
        policy.reset()

    def act(observation: dict, seed: int) -> np.ndarray:
        nonlocal queued_actions, inference_calls
        if not queued_actions:
            raw = {
                "observation.state": torch.from_numpy(
                    observation["observation.state"].copy()
                ),
                "observation.images.camera1": torch.from_numpy(
                    observation["observation.images.camera1"].copy()
                ).permute(2, 0, 1).float()
                / 255.0,
                "task": observation["task"],
            }
            torch.manual_seed(seed + inference_calls)
            batch = preprocessor(raw)
            started = time.perf_counter()
            with torch.inference_mode():
                normalized_chunk = policy.predict_action_chunk(batch)
                chunk = postprocessor(normalized_chunk)
            inference_times.append(time.perf_counter() - started)
            inference_calls += 1
            values = chunk.squeeze(0).detach().cpu().numpy()
            queued_actions = [row.copy() for row in values[:action_horizon]]
        return queued_actions.pop(0)

    diagnostics = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
        "dataset_metadata_sha256": metadata_digest(dataset_root),
        "inference_times": inference_times,
        "get_inference_calls": lambda: inference_calls,
    }
    return act, reset, diagnostics


def evaluate(args: argparse.Namespace) -> dict:
    suite, suite_sha256 = load_suite(args.suite)
    episodes = suite["episodes"][: args.limit_episodes]
    action_horizon = int(suite["action_horizon"])
    env = ObstacleReachEnv(
        image_size=int(suite["image_size"]), max_steps=int(suite["max_steps"])
    )
    policy_details: dict = {}

    if args.controller == "smolvla-base":
        act, reset_controller, diagnostics = make_smolvla_controller(
            args.dataset, args.device, action_horizon
        )
        policy_details = {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": diagnostics["checkpoint_sha256"],
            "parameter_count": diagnostics["parameter_count"],
            "training_state": "pretrained_base_no_task_specific_gradient_updates",
            "embodiment_calibration": "seed_dataset_normalization_statistics_only",
            "calibration_dataset": "local/obstacle-reach-rl-v0",
            "calibration_metadata_sha256": diagnostics["dataset_metadata_sha256"],
        }
    else:
        diagnostics = {"inference_times": [], "get_inference_calls": lambda: 0}

        def reset_controller() -> None:
            return None

        def act(observation: dict, seed: int) -> np.ndarray:
            del observation, seed
            return env.expert_action()

        policy_details = {"id": "classical_damped_least_squares_expert"}

    results: list[dict] = []
    try:
        for index, spec in enumerate(episodes):
            observation, info = env.reset(seed=spec["seed"], options=spec)
            reset_controller()
            initial_distance = float(info["distance"])
            episode_return = 0.0
            min_distance = float(info["distance"])
            previous_action: np.ndarray | None = None
            total_action_change = 0.0
            started = time.perf_counter()
            steps = 0
            while True:
                action = act(observation, spec["seed"])
                projected = env.project_action(action)
                if previous_action is not None:
                    total_action_change += float(np.linalg.norm(projected - previous_action))
                previous_action = projected.copy()
                observation, reward, terminated, truncated, info = env.step(projected)
                episode_return += reward
                min_distance = min(min_distance, float(info["distance"]))
                steps += 1
                if terminated or truncated:
                    break
            episode = {
                "episode": index,
                "seed": spec["seed"],
                "task_key": spec["task_key"],
                "disturbance_step": spec["disturbance_step"],
                "disturbance_applied": bool(info["disturbance_applied"]),
                "success": bool(info["is_success"]),
                "collisions": int(info["collision_count"]),
                "return": round(float(episode_return), 6),
                "steps": steps,
                "initial_distance": round(initial_distance, 6),
                "min_distance": round(min_distance, 6),
                "final_distance": round(float(info["distance"]), 6),
                "total_action_change": round(total_action_change, 6),
                "wall_seconds": round(time.perf_counter() - started, 6),
            }
            results.append(episode)
            print(json.dumps(episode, sort_keys=True))
    finally:
        env.close()

    inference_times = diagnostics["inference_times"]
    aggregate = {
        "episodes": len(results),
        "success_rate": float(np.mean([item["success"] for item in results])),
        "collision_episode_rate": float(np.mean([item["collisions"] > 0 for item in results])),
        "mean_return": float(np.mean([item["return"] for item in results])),
        "mean_final_distance": float(np.mean([item["final_distance"] for item in results])),
        "inference_calls": int(diagnostics["get_inference_calls"]()),
        "mean_inference_seconds": statistics.fmean(inference_times) if inference_times else 0.0,
        "p95_inference_seconds": percentile(inference_times, 0.95),
    }
    return {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "controller": args.controller,
        "policy": policy_details,
        "evaluation_suite": {
            "path": str(args.suite),
            "id": suite["suite"],
            "sha256": suite_sha256,
            "action_horizon": action_horizon,
        },
        "runtime": {
            "device": args.device if args.controller == "smolvla-base" else "cpu",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "episodes": results,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("expert", "smolvla-base"), required=True)
    parser.add_argument("--suite", type=Path, default=Path("configs/eval/obstacle_reach_v0.json"))
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/obstacle-reach-seed-v0"))
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit_episodes is not None and args.limit_episodes < 1:
        parser.error("--limit-episodes must be positive")
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
