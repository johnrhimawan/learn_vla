# Learn VLA robotics with LeRobot and MuJoCo

This repository is a compact robotics-first starting point for someone who
already knows machine learning. It uses MuJoCo for dynamics and rendering, and
LeRobot Dataset v3 for the trajectory format used by LeRobot training code.

## Setup

The project uses Python 3.12 because current LeRobot releases require it. `uv`
creates `.venv`, installs the correct Python when needed, and reproduces the
locked environment:

```bash
brew install ffmpeg@8
uv sync
```

The versioned FFmpeg formula matters: TorchCodec 0.11 supports FFmpeg through
version 8, while Homebrew's unversioned formula is now FFmpeg 9. Activate the
environment with `source .venv/bin/activate` for interactive work, or use the
project runner below for commands.

Verify the installation:

```bash
scripts/run python -c "import lerobot, mujoco, torchcodec; print(mujoco.__version__)"
```

`scripts/run` is a thin `uv run` wrapper. On this Mac it also points the dynamic
loader at Homebrew's versioned FFmpeg 8 libraries, since current TorchCodec
does not yet support the unversioned FFmpeg 9 formula.

On Apple Silicon, PyTorch uses MPS where a policy supports it. MuJoCo itself
runs on the CPU and uses the Mac GPU for rendering. The H200 cluster is the
right place for later policy fine-tuning; keep this lockfile for the Mac
simulation environment and create a CUDA-specific lock or container for the
cluster rather than copying the Mac virtual environment.

## First example: Cartesian reaching to a LeRobot dataset

Run three headless episodes:

```bash
scripts/run python examples/mujoco_reach.py --episodes 3
```

The script prints the generated path under `artifacts/`. Inspect it with:

```bash
scripts/run python examples/inspect_dataset.py artifacts/mujoco-reach-YYYYMMDD-HHMMSS
```

For a live window on macOS, MuJoCo requires its `mjpython` launcher:

```bash
scripts/run mjpython examples/mujoco_reach.py --episodes 1 --viewer
```

The generated dataset has the same central fields used for robot learning:

| Field | Meaning | Shape |
| --- | --- | --- |
| `observation.images.overview` | exteroceptive RGB observation | `3 x H x W` |
| `observation.state` | joints, joint velocities, tool pose, target pose | `12` |
| `action` | next joint-position command | `3` |
| `task` | language conditioning | string |

The control loop deliberately uses a classical expert. At every 20 Hz control
tick it observes the scene, computes a Cartesian error, maps that error into
joint motion with a damped pseudoinverse of the end-effector Jacobian, records
the transition, and advances ten 200 Hz physics steps. This exposes the pieces
that tend to be new even with a strong ML background:

- generalized coordinates (`qpos`) and velocities (`qvel`);
- forward kinematics and a site Jacobian;
- singularity-robust inverse kinematics;
- a slower control loop over a faster physics integrator;
- episode boundaries and synchronized multimodal robot data.

Read [`examples/mujoco_reach.py`](examples/mujoco_reach.py) in this order:
`MODEL_XML`, `observe`, `expert_action`, `step`, then `collect_dataset`.

## Where this leads

The example is an expert demonstration generator. A practical learning path is:

1. Replace the expert action with keyboard or gamepad teleoperation and collect
   demonstrations.
2. Train a small behavior-cloning or ACT policy on the resulting dataset and
   close the loop in this simulator.
3. Add randomized target locations, textures, cameras, latency, and sensor
   noise to study generalization and sim-to-real gaps.
4. Move training to the H200 cluster and try a LeRobot VLA policy such as
   SmolVLA, while continuing to evaluate rollouts locally or in parallel
   simulation workers.

The Mac setup is enough for simulation, dataset inspection, policy plumbing,
and small inference experiments. Large VLA fine-tuning is intentionally a
separate CUDA environment because PyTorch wheels and accelerator tooling differ
between macOS/MPS and Linux/CUDA.

## Primary references

- [LeRobot installation](https://huggingface.co/docs/lerobot/installation)
- [LeRobot Dataset API](https://huggingface.co/docs/lerobot/api/datasets)
- [MuJoCo Python bindings](https://mujoco.readthedocs.io/en/stable/python.html)
