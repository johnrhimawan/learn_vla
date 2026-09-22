# Tennis VLA

This repository develops a vision-language-action system that returns tennis
balls in MuJoCo. The first target is a fixed-base 7-DoF Sawyer arm with a
regulation-size racket returning programmable feeds after one bounce. Later
milestones add visual ball tracking, behavior cloning, residual reinforcement
learning, language-conditioned placement, rallies, and guarded hardware
transfer.

The repository contains only the tennis project. Start with
[`docs/tennis_vla_plan.md`](docs/tennis_vla_plan.md) for the design and
[`configs/tennis/roadmap.yaml`](configs/tennis/roadmap.yaml) for machine-readable
milestone status. [`HANDOFF.md`](HANDOFF.md) records the current engineering
state and the next work in detail.

## Setup

The project uses Python 3.12 and `uv`:

```bash
brew install ffmpeg@8
uv sync
```

For interactive work, activate the environment:

```bash
source .venv/bin/activate
```

Commands in this repository use the project runner, which invokes `uv run` and
selects the supported FFmpeg library on macOS:

```bash
scripts/run python -c "import lerobot, mujoco, torch; print(mujoco.__version__)"
```

If Hugging Face access is needed, put `HF_TOKEN` in `~/.env` and load it into
the current shell without printing it:

```bash
set -a
source ~/.env
set +a
```

Do not commit tokens or `.env` files.

## View the simulator

MuJoCo needs its `mjpython` launcher for an interactive window on macOS:

```bash
scripts/run mjpython examples/tennis_ball_flight.py --viewer
scripts/run mjpython examples/tennis_arm_workspace.py --viewer
```

The ball-flight viewer shows the court and a deterministic feed. The arm viewer
shows the Sawyer racket workspace. The remaining audits run headlessly so they
can produce reproducible JSON reports.

## Current system

The control design is hierarchical:

| Layer | Rate | Responsibility |
| --- | ---: | --- |
| Safety monitor | 1 kHz | enforce motion, workspace, and contact limits |
| Joint servo | 250 Hz | track a dynamically feasible racket trajectory |
| Strike policy | 50 Hz | update the intercept and racket contact state |
| VLA planner | 10 Hz | choose the stroke and landing intent from vision and language |

Implemented work includes the court and no-spin ball physics, a programmable
feeder, the Sawyer racket model, stereo rendering and perception data, a small
learned ball detector, privileged intercept planning, nonzero-velocity racket
strikes, collision and command screening, live MuJoCo contact, legal-return
scoring, and bounded recovery to the ready pose.

The project does **not** yet use a trained tennis VLA or an RL policy. The
active M2 controller is a privileged planning oracle. A development ball
detector is the only learned tennis component. M3 will behavior-clone oracle
returns into a VLA; M4 will train a residual SAC policy for contact timing,
racket state, and shot placement.

## Reproduce the core checks

Run the test suite:

```bash
scripts/run python -m unittest discover -s tests -v
```

Run the main physics, perception, and controller examples:

```bash
scripts/run python examples/tennis_ball_flight.py
scripts/run python examples/tennis_contact_probe.py
scripts/run python examples/tennis_perception_baseline.py
scripts/run python examples/tennis_intercept_oracle.py --seed 1
scripts/run python examples/tennis_strike_execution.py
```

Run the smaller active-strike development audit:

```bash
scripts/run python examples/tennis_active_strike_audit.py \
  --split development --workers 4
```

The expanded 200-seed development audit is unfinished and computationally
expensive:

```bash
scripts/run python examples/tennis_active_strike_audit.py \
  --split expanded-development --workers 8
```

Read `HANDOFF.md` before running it. In particular, investigate the observed
seed 10096 recovery failure and planning latency first. Do not inspect the
reserved seeds 14000–14199 until the v1 planner is frozen.

## Perception data and model

Generate a small randomized stereo dataset:

```bash
scripts/run python examples/generate_tennis_flight_dataset.py \
  --output artifacts/tennis-flight-v0 \
  --preview artifacts/tennis-flight-v0-preview.png
```

Evaluate the fixed color detector:

```bash
scripts/run python examples/evaluate_tennis_perception.py \
  artifacts/tennis-flight-v0 \
  --output results/tennis/randomized_perception_baseline_v0.json
```

Train and evaluate the heatmap detector:

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

The tracked development checkpoint at
[`checkpoints/tennis/ball_detector_dev_v0.pt`](checkpoints/tennis/ball_detector_dev_v0.pt)
reaches 99.48% valid stereo coverage, 2.69 cm 3D position RMSE, and 15.1 ms
contact-time RMSE on its untouched development test split. This is a 320×240,
10 Hz development result; the production M1 training run remains open.

## Verified controller results

| Result | Outcome | Report |
| --- | --- | --- |
| Kinematic contact curriculum | 193/200 feasible | [`contact_curriculum_kinematic_audit_v0.json`](results/tennis/contact_curriculum_kinematic_audit_v0.json) |
| Zero-velocity arrival planning | 191/200 feasible | [`dynamic_intercept_audit_v0.json`](results/tennis/dynamic_intercept_audit_v0.json) |
| Tracked arrival | 190/200 pass | [`arrival_tracking_audit_v0.json`](results/tennis/arrival_tracking_audit_v0.json) |
| Active-strike development prefix | 20/20 strict pass | [`active_strike_development_v1.json`](results/tennis/active_strike_development_v1.json) |
| First sealed active-strike gate | 151/200 strict pass; failed gate | [`active_strike_heldout_v0.json`](results/tennis/active_strike_heldout_v0.json) |
| v1 development baseline | 158/200 strict pass | [`active_strike_v1_development_baseline_v0.json`](results/tennis/active_strike_v1_development_baseline_v0.json) |

Seeds 12000–12199 belong to the failed, retired v0 gate and must not be used
for tuning. Seeds 10000–10199 are the current v1 development set. Seeds
14000–14199 are the untouched future v1 final gate.

## Repository map

| Path | Purpose |
| --- | --- |
| `tennis_vla/` | tennis physics, perception, planning, control, and execution |
| `examples/tennis_*.py` | reproducible simulations, audits, training, and evaluation entry points |
| `tests/` | deterministic tennis regression suite |
| `configs/tennis/` | datasets, evaluation, compute, and milestone contracts |
| `results/tennis/` | versioned evaluation evidence |
| `checkpoints/tennis/` | small tracked development checkpoints |
| `docker/h200/` | CUDA learner runtime for future VLA and RL training |

## References

- [LeRobot SmolVLA](https://huggingface.co/docs/lerobot/smolvla)
- [LeRobot HIL-SERL](https://huggingface.co/docs/lerobot/hilserl_sim)
- [MuJoCo Python bindings](https://mujoco.readthedocs.io/en/stable/python.html)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [2026 ITF Rules of Tennis](https://www.itftennis.com/media/7221/2026-rules-of-tennis-english.pdf)
