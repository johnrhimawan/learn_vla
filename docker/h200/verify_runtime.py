"""Fail fast if an H200 worker cannot run this project's learner stack."""

import json

import lerobot
import mujoco
import torch


if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; start the container with --gpus all")

capability = torch.cuda.get_device_capability(0)
if capability < (9, 0):
    raise SystemExit(f"Expected Hopper-class compute capability >= 9.0, got {capability}")

probe = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
result = probe @ probe
torch.cuda.synchronize()
print(
    json.dumps(
        {
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "compute_capability": capability,
            "bfloat16_matmul": list(result.shape),
            "torch": torch.__version__,
            "lerobot": getattr(lerobot, "__version__", "0.6.1"),
            "mujoco": mujoco.__version__,
        },
        indent=2,
    )
)
