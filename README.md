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

## Load and run SmolVLA

The environment includes LeRobot's `smolvla` and `training` extras. The model
used here is the public 450M-parameter `lerobot/smolvla_base` checkpoint, pinned
to revision `d9f33c94a60fb382c90dea2164c96845bd955e28` so the exercise remains
reproducible with LeRobot 0.6.1.

If your Hugging Face token is stored in `~/.env`, export it into the current
shell without printing it:

```bash
set -a
source ~/.env
set +a
```

Download the checkpoint without loading it:

```bash
scripts/run python examples/smolvla_inference.py --download-only
```

Then run one real policy prediction on Apple Silicon:

```bash
scripts/run python examples/smolvla_inference.py --device mps
```

This passes a rendered MuJoCo frame, six proprioceptive values, and the task
text through SmolVLA. It prints the six-dimensional SO-100 action predicted by
the base model. The script deliberately does not execute that action: this
simulator has three joints and needs a fine-tuned three-dimensional action head
before a rollout is meaningful.

The Hugging Face cache lives under `.cache/huggingface/` and is ignored by Git.
The SmolVLA checkpoint plus its SmolVLM backbone support files occupy about
2.8 GB on this machine.

## Fine-tune SmolVLA on the MuJoCo arm

Collect demonstrations using a VLA-ready schema. Unlike the introductory
dataset, this dataset does not put end-effector or target coordinates in
`observation.state`. The target is visible only through `camera1`, so the model
cannot solve the task while ignoring vision.

```bash
scripts/run python examples/collect_smolvla_dataset.py \
  --episodes 50 \
  --output artifacts/smolvla-reach-50
```

Inspect the result:

```bash
scripts/run python examples/inspect_dataset.py \
  artifacts/smolvla-reach-50 \
  --repo-id local/smolvla-mujoco-reach
```

The learning input and target are:

| Field | Role | Shape |
| --- | --- | --- |
| `observation.images.camera1` | RGB scene observation | `3 x 256 x 256` |
| `observation.state` | joint positions and velocities | `6` |
| `task` | language instruction | string |
| `action` | expert joint-position command | `3` |

Run a one-step training check on the Mac:

```bash
SMOLVLA_OUTPUT=artifacts/smolvla-one-step \
  scripts/train_smolvla artifacts/smolvla-reach-50 1 mps
```

For a short learning experiment, increase the step count. Batch size 1 keeps
MPS memory use modest:

```bash
scripts/train_smolvla artifacts/smolvla-reach-50 100 mps
```

For an H200 run, reproduce the environment from `uv.lock`, make the dataset
available on the cluster, and use CUDA with a larger batch:

```bash
export SMOLVLA_MODEL=lerobot/smolvla_base
export SMOLVLA_BATCH_SIZE=64
export SMOLVLA_OUTPUT=outputs/smolvla-reach-h200
scripts/train_smolvla /path/to/smolvla-reach-50 20000 cuda
```

The current dataset has one instruction, so it teaches visually conditioned
reaching rather than language-dependent task selection. Add several tasks with
different instructions and demonstrations when you want to measure whether the
policy responds to language changes.

## Where this leads

The example is an expert demonstration generator. A practical learning path is:

1. Fine-tune SmolVLA on the generated expert demonstrations and add a rollout
   adapter that maps its three learned outputs back into the simulator.
2. Replace the expert action with keyboard or gamepad teleoperation and collect
   demonstrations.
3. Add several language-distinct tasks, plus randomized textures, cameras,
   latency, and sensor noise to study generalization and sim-to-real gaps.
4. Move full training to the H200 cluster while continuing to evaluate policy
   rollouts locally or in parallel simulation workers.

The Mac setup is enough for simulation, dataset inspection, policy plumbing,
and small inference experiments. Large VLA fine-tuning is intentionally a
separate CUDA environment because PyTorch wheels and accelerator tooling differ
between macOS/MPS and Linux/CUDA.

## Active project: a tennis-playing VLA

The active goal is a vision-language-action system that can return tennis balls,
place shots from language instructions, and eventually sustain rallies. The
project starts with a fixed-base 7-DoF racket arm and programmable feeds, then
adds vision, behavior cloning, residual RL, language-conditioned placement,
closed-loop rallies, and guarded transfer to hardware.

Run the first deterministic ball-flight check:

```bash
scripts/run python examples/tennis_ball_flight.py
```

Visualize it on macOS with:

```bash
scripts/run mjpython examples/tennis_ball_flight.py --viewer
```

Audit the pinned 7-DoF arm and generated racket workspace:

```bash
scripts/run python examples/tennis_arm_workspace.py
scripts/run mjpython examples/tennis_arm_workspace.py --viewer
scripts/run python examples/tennis_contact_probe.py
```

Run the calibrated two-camera perception baseline on 25 held-out feeds:

```bash
scripts/run python examples/tennis_perception_baseline.py
```

The command renders 450 stereo observations, reconstructs the ball without
reading its simulator state, predicts its strike-plane crossing at least 150 ms
ahead, and writes a versioned report to
`results/tennis/perception_baseline_v0.json`. Exact simulator state is used only
to score the predictions. The canonical baseline is the start of M1; learned
tracking under randomized appearance and camera conditions is the next gate.

Generate a local randomized `tennis-flight-v0` perception dataset:

```bash
scripts/run python examples/generate_tennis_flight_dataset.py \
  --output artifacts/tennis-flight-v0 \
  --preview artifacts/tennis-flight-v0-preview.png
```

The default run creates 10 training, 3 validation, and 3 test episodes at
512×384 and 50 Hz. Each frame has two RGB images plus exact 3D position,
velocity, bounce state, and time-to-strike-plane labels. Each episode also has
its sampled visual domain and both camera calibrations. The split seed ranges
do not overlap. These privileged values are training targets and evaluation
labels; they are not visual-policy inputs.

For the planned production corpus on a machine with ample storage:

```bash
scripts/run python examples/generate_tennis_flight_dataset.py \
  --output /path/to/tennis-flight-v0 \
  --train-episodes 1000 \
  --validation-episodes 100 \
  --test-episodes 100 \
  --preview /path/to/tennis-flight-v0-preview.png
```

The exact production contract is in
[`configs/tennis/flight_dataset_v0.json`](configs/tennis/flight_dataset_v0.json).
A revision-pinned four-episode smoke run validated 86 referenced images with no
missing files; see
[`results/tennis/flight_dataset_smoke_v0.json`](results/tennis/flight_dataset_smoke_v0.json)
and its
[`stereo preview`](results/tennis/flight_dataset_preview_v0.png).

Measure the fixed color detector on the randomized data:

```bash
scripts/run python examples/evaluate_tennis_perception.py \
  artifacts/tennis-flight-v0-smoke \
  --output results/tennis/randomized_perception_baseline_smoke_v0.json
```

The smoke result deliberately fails the strict M1 gate: 65.1% stereo detection
coverage, 75% timing-prediction coverage, and 39.1 ms contact-time RMSE. This is
the baseline the learned M1 estimator must beat.

Train the full-resolution heatmap detector, then evaluate it through the same
stereo gate:

```bash
scripts/run python examples/train_tennis_ball_detector.py \
  /path/to/tennis-flight-v0 \
  --checkpoint outputs/tennis-ball-detector/model.pt \
  --report outputs/tennis-ball-detector/training.json \
  --device cuda

scripts/run python examples/evaluate_tennis_ball_detector.py \
  /path/to/tennis-flight-v0 \
  outputs/tennis-ball-detector/model.pt \
  --device cuda \
  --output outputs/tennis-ball-detector/evaluation.json
```

The detector keeps the render at full resolution because a distant tennis ball
can occupy one pixel. Its configuration and strict coverage-aware gates are in
[`configs/tennis/perception_training_v0.json`](configs/tennis/perception_training_v0.json).

The milestone plan, control architecture, datasets, RL stages, Mac/H200 split,
and quantitative exit gates are in
[`docs/tennis_vla_plan.md`](docs/tennis_vla_plan.md). The previous
obstacle-reaching project remains as an archived learning prototype in
[`docs/archive/obstacle_reach_rl_project.md`](docs/archive/obstacle_reach_rl_project.md).

## Primary references

- [LeRobot installation](https://huggingface.co/docs/lerobot/installation)
- [LeRobot Dataset API](https://huggingface.co/docs/lerobot/api/datasets)
- [LeRobot SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla)
- [LeRobot HIL-SERL simulation](https://huggingface.co/docs/lerobot/hilserl_sim)
- [SmolVLA base checkpoint](https://huggingface.co/lerobot/smolvla_base)
- [MuJoCo Python bindings](https://mujoco.readthedocs.io/en/stable/python.html)
- [2026 ITF Rules of Tennis](https://www.itftennis.com/media/7221/2026-rules-of-tennis-english.pdf)
