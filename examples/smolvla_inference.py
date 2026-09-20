"""Run the pretrained SmolVLA policy on one observation from the MuJoCo arm.

This is a plumbing and inference exercise, not a useful controller yet. The
base checkpoint was trained for 6-DoF SO-100 arms, so its six output values are
printed but deliberately not sent to this example's 3-DoF arm. Fine-tuning on
the matching MuJoCo dataset is the step that gives those actions local meaning.

Run from the repository root:

    scripts/run python examples/smolvla_inference.py
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Let PyTorch use CPU implementations for any operation that MPS does not yet
# implement. This must be set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from huggingface_hub import snapshot_download
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.device_utils import auto_select_torch_device

from mujoco_reach import ReachingSimulation, RolloutConfig, TASK


MODEL_ID = "lerobot/smolvla_base"
# Pin the exact checkpoint tested with LeRobot 0.6.1. Override it from the CLI
# when you intentionally want to test a newer model revision.
MODEL_REVISION = "d9f33c94a60fb382c90dea2164c96845bd955e28"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu", "cuda"),
        default="auto",
        help="Inference device. 'auto' selects MPS on Apple Silicon.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and verify the SmolVLA checkpoint without loading it.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return auto_select_torch_device().type
    return requested


def download_checkpoint(model_id: str, revision: str) -> Path:
    """Resolve a complete immutable Hub snapshot in the project HF cache."""
    snapshot = snapshot_download(repo_id=model_id, revision=revision)
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    missing = [name for name in required if not (Path(snapshot) / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint; missing: {', '.join(missing)}")
    return Path(snapshot)


def simulator_observation(seed: int) -> dict[str, torch.Tensor | str]:
    """Create the state, RGB view, and instruction consumed by a VLA policy."""
    simulation = ReachingSimulation(
        RolloutConfig(steps=1, image_size=256, seed=seed, viewer=False)
    )
    try:
        simulation.reset()
        full_state, image = simulation.observe()
    finally:
        simulation.close()

    # SmolVLA's base checkpoint expects six proprioceptive values. We expose
    # only joint position and velocity; target location must come from vision.
    proprioception = torch.from_numpy(full_state[:6])
    camera = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
    return {
        "observation.state": proprioception,
        "observation.images.camera1": camera,
        "task": TASK,
    }


def main() -> None:
    args = parse_args()
    checkpoint = download_checkpoint(args.model, args.revision)
    print(f"checkpoint: {checkpoint}")
    if args.download_only:
        print("checkpoint files verified")
        return

    device = resolve_device(args.device)
    print(f"device:     {device}")
    print("loading the 450M-parameter policy...")
    started = time.perf_counter()

    config = PreTrainedConfig.from_pretrained(
        checkpoint,
        cli_overrides=[f"--device={device}"],
    )
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
    )
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    print(f"parameters: {parameter_count / 1_000_000:.1f}M")
    batch = preprocessor(simulator_observation(args.seed))

    torch.manual_seed(args.seed)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        normalized_action = policy.select_action(batch)
        action = postprocessor(normalized_action)

    elapsed = time.perf_counter() - inference_started
    print(f"load time:  {time.perf_counter() - started - elapsed:.2f}s")
    print(f"inference:  {elapsed:.2f}s")
    print(f"task:       {TASK}")
    print(f"action:     {action.squeeze(0).tolist()}")
    print("The action is not executed: fine-tune SmolVLA for this 3-DoF arm first.")


if __name__ == "__main__":
    main()
