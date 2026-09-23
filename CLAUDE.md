# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A MuJoCo tennis vision-language-action project: a fixed-base 7-DoF Sawyer arm with a
regulation racket returns programmable ball-machine feeds after one bounce. The
repository is tennis-only by policy — keep new code and docs tennis-specific.

There is **no trained VLA or RL policy yet**. The active M2 controller is a *privileged
planning oracle* that reads exact simulator ball state. The only learned component is a
small development ball detector. M3 behavior-clones the oracle; M4 adds residual SAC.

Three documents carry the living state and must be updated when a milestone, split, or
threshold changes:
- `HANDOFF.md` — current engineering state, in-flight work, next steps. **Read first.**
- `docs/tennis_vla_plan.md` — design.
- `configs/tennis/roadmap.yaml` — machine-readable milestone status and acceptance gates.

## Commands

Always use the `scripts/run` wrapper instead of bare `python`/`uv run`. It invokes
`uv run` and, on macOS, points dyld at `ffmpeg@8` — the unversioned Homebrew FFmpeg is
newer than the TorchCodec build LeRobot 0.6.1 pins, and imports fail without it. It also
pins `UV_CACHE_DIR=.uv-cache` and `HF_HOME=.cache/huggingface`.

```bash
brew install ffmpeg@8 && uv sync          # setup (Python 3.12, uv, package = false)

scripts/run python -m unittest discover -s tests -v          # full suite (~38 tests, ~45 s)
scripts/run python -m unittest discover -s tests -p test_tennis_strike.py -v   # one module
scripts/run python -m compileall -q tennis_vla examples tests
```

`pytest`, Ruff, and Black are **not installed**. Validation is `unittest`, `compileall`,
and `git diff --check`. `tests/` has no `__init__.py`, so `unittest tests.test_x` does
**not** resolve — select a module with `discover -p`. Finish every change with, in order:

```bash
scripts/run python -m unittest discover -s tests -v
scripts/run python -m compileall -q tennis_vla examples tests
git diff --check
git status --short
```

Interactive MuJoCo windows on macOS need `mjpython`, not `python`:

```bash
scripts/run mjpython examples/tennis_ball_flight.py --viewer
scripts/run mjpython examples/tennis_arm_workspace.py --viewer
```

Everything else runs headless so it can emit reproducible JSON. The main audits:

```bash
scripts/run python examples/tennis_strike_execution.py                          # canonical live strike
scripts/run python examples/tennis_active_strike_audit.py --split development --workers 4
scripts/run python examples/tennis_active_strike_audit.py --seed-start 10096 --count 1 \
  --workers 1 --output /tmp/tennis-seed-10096.json                              # single-seed debug
scripts/run python examples/tennis_strike_planning_diagnostics.py               # planner failure analysis
scripts/run python examples/render_tennis_strike.py                             # README animation
```

The staged planner fallbacks can take minutes per hard feed (observed up to ~130 s under
worker contention); a full 200-seed audit runs 20+ minutes. Budget accordingly and prefer
narrow `--seed-start/--count` runs while debugging.

`scripts/run_h200` is the counterpart runner for the CUDA image (`docker/h200/`); it skips
uv entirely because the NGC image already has CUDA PyTorch, and sets `MUJOCO_GL=egl`.

## Evaluation seed discipline — the most important rule here

Feeds are deterministic functions of a seed, so seed ranges *are* the dataset splits.
Violating this silently invalidates every downstream gate.

| Seeds | Use | State |
| --- | --- | --- |
| 9000–9019 | small v0 development prefix | used |
| 12000–12199 | first sealed v0 gate | **failed and retired — never tune on these** |
| 10000–10199 | v1 development and tuning | active |
| 14000–14199 | sealed v1 final gate | **untouched — do not run, inspect, or diagnose** |

Do not touch 14000–14199 until the v1 search, execution, recovery logic, and thresholds
are frozen; add a dedicated v1 final profile at that point. The existing `final-heldout`
CLI profile still names the retired 12000–12199 range.

When a gate fails, fix the robustness gap. Never relax a safety threshold to make a
metric pass.

## Architecture

The intended runtime is hierarchical; rates are asserted in config validators, not just
documentation:

| Layer | Rate | Role | Code |
| --- | ---: | --- | --- |
| Safety monitor | 1 kHz | motion, workspace, contact limits | `execution.py` |
| Joint servo | 250 Hz | track the feasible racket trajectory | `execution.py` |
| Strike policy | 50 Hz | intercept time, pose, racket velocity | `strike.py` (not yet real-time) |
| VLA planner | 10 Hz | vision + language → stroke intent | planned (M3+) |

Planning-to-execution data flow, one module per stage:

```
feeder.py           seeded feed (rejection-sampled for a legal first bounce)
  ↓
ballistics.py       analytical no-spin RK4 flight  ─┐
execution.py        simulate_mujoco_ball_flight()  ─┴─ two flight models, both used
  ↓
intercept.py        racket-pose IK → InterceptCandidate (time, pose, joint config)
  ↓
strike.py           plan_safe_center_strikes(): search over face pitch/yaw, normal speed,
                    tangent ratio; quintic joint trajectory; execution + recovery screens;
                    ranks StrikePlans by landing-target error
  ↓
trajectory.py       minimum-jerk arrival + dynamic feasibility screens
  ↓
execution.py        execute_strike(): 1 kHz physics, 250 Hz reference tracking, live
                    MuJoCo contact, legal-return scoring, bounded recovery to ready pose
```

`execute_strike` takes an optional `observer` hook, called after each 1 kHz step
of both the strike and the recovery. It is read-only and must stay that way — it
exists so `examples/render_tennis_strike.py` can record the executed motion
without duplicating the controller. The README animation is rendered through it,
so it shows executed motion, not the plan.

### Never index the arm with a literal slice

`arm.EmbodimentLayout` resolves the arm's indices **by joint name**; use
`arm_layout(model)` and its slices instead of `[:7]`. MuJoCo keeps four index spaces —
`arm_joints` (`jnt_range`), `arm_dof` (`qvel`/`qacc`/`qfrc_*`/Jacobian columns),
`arm_actuators` (`ctrl`/`actuator_*`), `arm_qpos` — and they coincide today *only*
because every arm joint is a 1-DoF hinge with a 1:1 actuator. Picking the wrong one
still works on the fixed base and breaks silently later.

This matters because mobile-base joints sort **ahead** of the arm: adding them makes
`qpos[:7]` address `[base_x, base_y, base_yaw, j0, j1, j2, j3]` with the right shape,
no exception, and wrong numbers. Note `qpos[addr : addr + 7]` for the ball is freejoint
width (3 position + 4 quaternion), not an arm slice — leave those alone.

Two ball-flight models coexist deliberately: `ballistics.simulate_ball_flight` is the
analytical reference used by the feeder and tests, while
`execution.simulate_mujoco_ball_flight` is the calibrated contact model the planner and
audits run against. Planning from the wrong one produces plausible but wrong contacts.

Scene conventions (`environment.make_tennis_contact_model`):
- Court along ±x, net at x=0, half-length 11.885 m. Arm base at **x = -10.6**, so the
  robot works at negative x and feeds come from x ≈ +10 heading -x.
- Primary contact box: x ∈ [-10.1, -9.45], |y| ≤ 1.1, z ∈ [0.60, 1.35]. The fallback
  box widens to x ∈ [-10.55, -9.35], |y| ≤ 1.25, z ∈ [0.50, 1.55].
- Court contact (`COURT_CONTACT_SOLREF/SOLIMP/FRICTION`) is calibrated to reproduce the
  analytical v0 bounce and is separate from the ball–racket pair. Changing either
  invalidates the tracked bounce and contact reports.
- The site `racket_center` is the IK and Jacobian target throughout.

Perception is a parallel track: `perception.py` (fixed-color stereo baseline),
`flight_dataset.py` + `domain_randomization.py` (randomized stereo dataset generation),
`learned_perception.py` (heatmap detector training/inference),
`perception_evaluation.py` (baseline scoring).

### Strike planner fallback staging

`StrikeSearchConfig` runs a fast primary search, then escalates only on failure:
primary single IK branch → 4 IK branches → expanded contact bounds → extra tangent
ratios (-0.1, -0.2) → a 10° face yaw toward court center. Each stage costs real time, so
changes here move audit runtime as much as they move coverage. Always pass
`diagnostics=` to `plan_safe_center_strikes` in development reports — it counts
candidates, velocity directions, screen outcomes, and which fallbacks fired.

## Reports and evidence

Audits write versioned JSON to tracked `results/tennis/`; large generated datasets go to
ignored `artifacts/` or external storage. Rendered animations for the README live in
tracked `docs/media/` and should stay small. Never claim a milestone from a single canonical
example, and never treat a partial console stream as a result — audit scripts only write
the report at the end.

Report-producing scripts embed `repository_state()` (`git_revision`,
`tracked_files_dirty`). **Commit before running one**, so the report records
`tracked_files_dirty: false`.

When reading a strike audit, use `strict_pass`, not `controller_safe`: the aggregate
`controller_safe` count also includes no-plan episodes, which default to safe.

Simulation motion limits are 4 rad/s, 15 rad/s², and a 0.03 rad joint-limit margin. These
are project simulation limits, **not** verified Sawyer hardware ratings — no real-robot
claims follow from them.

## Conventions

- `tennis_vla/` is a plain package (`package = false` in `pyproject.toml`); scripts in
  `examples/` prepend the repo root via `sys.path.insert(0, ...)` before importing it.
- Config and result objects are frozen dataclasses with explicit unit suffixes
  (`_m`, `_m_s`, `_rad_s2`, `_hz`, `_deg`) and a `validate()` method raising `ValueError`.
- Each example is an argparse CLI with an `--output` path defaulting under
  `results/tennis/`, printing a JSON summary plus `saved=<path>`.
- Tests are `unittest.TestCase` with expensive MuJoCo models built once in `setUpClass`.
- Commit subjects are short, capitalized, imperative, no prefixes ("Add lateral racket
  face yaw search").
- `HF_TOKEN` lives in `~/.env`; load with `set -a; source ~/.env; set +a` only when
  Hugging Face access is needed. Never print or commit it.
