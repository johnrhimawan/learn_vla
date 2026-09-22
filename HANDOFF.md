# Tennis VLA engineering handoff

This document hands the repository to the next engineer or coding agent. It is
current as of 2026-09-22. The project is in the middle of M2, the privileged
tennis strike controller. Preserve the evaluation split discipline below: the
next valid result depends on it.

## Goal and present boundary

The target is a vision-language-action system that sees an incoming tennis
ball, understands a placement instruction, executes a safe racket contact, and
eventually sustains a ten-shot singles rally. The first embodiment is a
fixed-base 7-DoF Sawyer arm returning programmable feeds after one bounce.

The system is not yet controlled by a trained VLA or reinforcement-learning
policy. Its active strike stack uses exact simulator ball state, analytical
ballistics, inverse kinematics, trajectory screening, inverse-dynamics
feedforward, and MuJoCo execution. That stack is intended to become the oracle
for `tennis-strike-oracle-v0` and the M3 behavior-cloning dataset. The only
learned component currently tracked is a small development heatmap detector
for the ball.

Milestone status:

| Milestone | State | Meaning |
| --- | --- | --- |
| M0 physics | Core complete | no-spin court, feed, bounce, racket contact, and Sawyer reference work |
| M1 perception | In progress | development detector passes its small split; production corpus and temporal estimator remain |
| M2 strike oracle | In progress | canonical strike works; 200-feed coverage and safety gate is still open |
| M3 behavior cloning | Planned | train the tennis VLA from oracle demonstrations |
| M4 residual RL | Planned | frozen-VLA residual SAC for timing, contact state, and placement |
| M5–M7 | Planned | language shots, rallies, and guarded transfer |

The complete design is in `docs/tennis_vla_plan.md`; structured status and
acceptance gates are in `configs/tennis/roadmap.yaml`.

## Environment

The repository expects Python 3.12, `uv`, MuJoCo 3.13, MuJoCo Menagerie, and
LeRobot 0.6.1. On macOS:

```bash
brew install ffmpeg@8
uv sync
source .venv/bin/activate
```

Prefer the runner for repository commands:

```bash
scripts/run python -m unittest discover -s tests -v
```

`scripts/run` invokes `uv run` and selects Homebrew FFmpeg 8 where needed.
`HF_TOKEN` is available in `~/.env`; load it only when Hugging Face access is
needed:

```bash
set -a
source ~/.env
set +a
```

Never print it or commit it.

The last complete regression run passed all 35 tests. The most recent timed run
completed in 45.4 seconds. `pytest`, Ruff, and Black are not installed; the
project currently validates with `unittest`, `py_compile`, and
`git diff --check`.

For interactive MuJoCo windows on macOS, use:

```bash
scripts/run mjpython examples/tennis_ball_flight.py --viewer
scripts/run mjpython examples/tennis_arm_workspace.py --viewer
```

## Architecture

The intended runtime layers are:

| Layer | Rate | Role |
| --- | ---: | --- |
| Safety monitor | 1 kHz | motion, workspace, contact, and eventual human limits |
| Joint servo | 250 Hz | track the feasible racket trajectory |
| Strike policy | 50 Hz | update intercept time, pose, and racket velocity |
| VLA planner | 10 Hz | map vision, state, history, and language to stroke intent |

Important source files:

| Path | Role |
| --- | --- |
| `tennis_vla/environment.py` | integrated Sawyer, racket, court, ball, and camera model |
| `tennis_vla/feeder.py` | deterministic feeder envelopes and sampled feeds |
| `tennis_vla/ballistics.py` | analytical no-spin flight and court outcome |
| `tennis_vla/impact.py` | first-order racket impact map |
| `tennis_vla/intercept.py` | racket-pose IK and intercept candidates |
| `tennis_vla/trajectory.py` | minimum-jerk trajectories and dynamic screens |
| `tennis_vla/strike.py` | active contact-state search, return prediction, and fallbacks |
| `tennis_vla/execution.py` | 1 kHz physics, 250 Hz tracking, contact, scoring, and recovery |
| `tennis_vla/flight_dataset.py` | randomized stereo flight dataset generation |
| `tennis_vla/learned_perception.py` | heatmap detector training and inference |
| `examples/tennis_active_strike_audit.py` | parallel, report-producing M2 audit |
| `examples/tennis_strike_planning_diagnostics.py` | focused planner failure analysis |

## Verified evidence

Do not infer a milestone pass from a single canonical example. These tracked
reports are the evidence available at handoff:

| Report | Result |
| --- | --- |
| `results/tennis/racket_contact_probe_v0.json` | held-racket contact agrees with the independent first-order impact check |
| `results/tennis/perception_baseline_v0.json` | canonical stereo baseline passes, without the production randomization gate |
| `results/tennis/ball_detector_dev_evaluation_v0.json` | 99.48% valid stereo coverage, 2.69 cm position RMSE, 15.1 ms timing RMSE on the development test split |
| `results/tennis/contact_curriculum_kinematic_audit_v0.json` | 193/200 feeds have buffered kinematic contacts |
| `results/tennis/dynamic_intercept_audit_v0.json` | 191/200 feeds have feasible zero-velocity arrival plans |
| `results/tennis/arrival_tracking_audit_v0.json` | 190/200 feeds pass tracked arrival at 1 kHz physics and 250 Hz control |
| `results/tennis/active_strike_development_v1.json` | 20/20 planned, contacted, landed legally, recovered, and passed strictly |
| `results/tennis/canonical_strike_execution_v1.json` | canonical live strike contacts about 2 ms after plan, clears the net, lands legally, and recovers |
| `results/tennis/active_strike_heldout_v0.json` | failed sealed v0 gate: 151/200 strict passes, 160/200 planned contacts and legal returns |
| `results/tennis/active_strike_v1_development_baseline_v0.json` | v1 development baseline: 158/200 strict, 164/200 planned/contacted/legal/recovered |
| `results/tennis/active_strike_v1_no_plan_diagnostics_v0.json` | classification of the 36 baseline no-plan feeds |

The failed v0 gate had nine executed episodes with controller failures: eight
actual-acceleration violations, one speed violation, and two unexpected-contact
labels, with some overlap. The report's aggregate `controller_safe` count also
includes no-plan episodes, which default to safe; use `strict_pass` when judging
the complete behavior.

In the v1 baseline, all 164 planned feeds contacted, returned legally, and
recovered. Six failed strict safety because measured joint acceleration peaked
between 15.031 and 15.101 rad/s², just above the 15 rad/s² gate. Of the 36 feeds
without plans:

- 14 had no kinematic candidates;
- 18 had recovery-safe motions but no coarse legal return;
- 3 had no feasible recovery;
- 1 failed the approach execution screen.

## Evaluation seeds: preserve this contract

| Seeds | Use | State |
| --- | --- | --- |
| 9000–9019 | small v0 development prefix | used |
| 12000–12199 | first sealed v0 gate | failed and retired |
| 10000–10199 | v1 development and tuning | active development set |
| 14000–14199 | future sealed v1 final gate | untouched |

Never tune on 12000–12199. Never run, inspect, or diagnose 14000–14199 until
the v1 search, execution, recovery logic, and thresholds are frozen. Add a
dedicated v1 final profile at that point; the existing `final-heldout` CLI
profile still names the retired 12000–12199 range.

## Current planner changes

The v1 development work is committed but its expanded audit did not finish.
`StrikeSearchConfig` keeps the original primary search, then stages slower
fallbacks only if primary planning fails:

1. the primary single IK branch;
2. up to four IK branches;
3. expanded contact bounds;
4. extra racket tangent ratios of -0.1 and -0.2;
5. one 10-degree racket-face yaw toward court center.

The expanded contact bounds are x=-10.55 to -9.35 m, |y| up to 1.25 m, and
z=0.50 to 1.55 m. The primary bounds remain x=-10.1 to -9.45 m, |y| up to 1.1
m, and z=0.60 to 1.35 m. The planner also supports an optional free wrist-roll
velocity, but the default remains fixed at zero because the experiment did not
recover any of nine target misses. Planned approach headroom is 3.9 rad/s and
14.0 rad/s² against measured gates of 4 rad/s and 15 rad/s².

Focused searches predicted that the expanded fallbacks could plan 190/200 v1
development feeds. Expanded bounds recovered 24 of the 36 baseline misses, a
-0.1 tangent ratio recovered seed 10089, and a 10-degree center yaw recovered
seed 10128. Very late contact bounds and free-wrist searches did not help their
target subsets.

Planning diagnostics are exposed through
`plan_safe_center_strikes(..., diagnostics=...)`. They count kinematic
candidates, velocity directions, impact approaches, execution and recovery
screens, coarse outcome categories, legal returns, and fallback use. Keep these
diagnostics in all development reports.

## Interrupted expanded audit

The following command was running when handoff was requested:

```bash
scripts/run python examples/tennis_active_strike_audit.py \
  --split expanded-development --workers 8
```

It was stopped cleanly. The script only writes its report at the end, so
`results/tennis/active_strike_v1_expanded_development_v0.json` does not exist.
Do not treat the partial console stream as an evaluation result.

The partial stream showed many previous no-plan feeds recovering and the five
known baseline acceleration failures seen so far passing. It also showed no
plans for seeds 10010, 10014, 10029, 10064, 10065, and 10111. Most critically,
seed 10096 planned, contacted, and landed legally but failed recovery and the
controller safety check. Its exact failure reason was lost because the report
was not written.

Before another full audit:

```bash
scripts/run python examples/tennis_active_strike_audit.py \
  --seed-start 10096 --count 1 --workers 1 \
  --output /tmp/tennis-seed-10096.json
```

Inspect its `execution.failure_reasons`, measured recovery state, acceleration,
speed, joint margin, and tracking error. Fix a real robustness gap in planning
or execution; do not relax the safety gates to make the metric pass.

The staged fallbacks can take minutes for hard feeds. The diagnostic baseline
observed planning times up to roughly 130 seconds under worker contention, and
the interrupted expanded audit ran for more than 20 minutes. Planning latency
is therefore an M2 deliverable. Record per-episode planning time, profile the
fallback stages, cache invariant work, and add bounded search budgets before
claiming the 50 Hz strike-policy architecture is implemented.

## Recommended next work

1. Reproduce seed 10096 alone and harden measured-state recovery until it
   passes without weakening limits.
2. Add planning-duration and per-stage timing to the audit report. Profile the
   difficult no-plan seeds and reduce repeated IK, trajectory, and flight work.
3. Rerun all 35 tests and the small 9000–9019 development audit.
4. Run the complete 10000–10199 expanded development audit from a clean commit.
   Require at least 95% contact, legal return, recovery, and strict safety,
   with zero controller safety failures.
5. Freeze the v1 planner, controller, recovery rules, thresholds, and report
   schema. Only then add and run the untouched 14000–14199 final profile once.
6. If the sealed gate passes, export `tennis-strike-oracle-v0` with images,
   deployable state, language instructions, stroke chunks, provenance, and
   privileged labels separated from policy inputs.
7. Finish the production M1 corpus and temporal visual estimator so M3 uses
   visual ball state rather than exact simulator state.
8. Train M3 behavior cloning on H200. After establishing its held-out tennis
   baseline, implement M4 residual SAC with the frozen VLA as the base policy
   and privileged simulator state only in the critic and reward.

M4 is where RL is needed: imitation can copy successful oracle strokes, while
the residual policy should refine contact time, contact pose, and racket
velocity for long-horizon landing accuracy under perception and dynamics
variation. Keep residual actions bounded and preserve the deterministic safety
and servo layers outside the learner.

## Acceptance gates and constraints

The immediate M2 gate is at least 95% contact across the sealed feeder
envelope. Development should additionally require at least 95% strict pass and
zero safety violations before consuming the final split. M3 requires at least
70% legal returns; M4 requires at least 85% legal returns and 60% target-zone
accuracy. These contracts are also recorded in `configs/tennis/roadmap.yaml`.

Current simulation motion limits are 4 rad/s speed, 15 rad/s² acceleration,
and a 0.03 rad joint-limit margin. They are project simulation limits, not
verified Sawyer hardware ratings. The controller still lacks a real torque
gate, hardware stop path, calibrated spin, flexible string response, and human
safety review.

Known scope limits:

- no spin dynamics or trained spin classifier;
- no calibrated string bed;
- exact simulator state remains in the M2 oracle;
- fixed-base single arm only;
- no trained tennis VLA, RL actor, target-zone policy, or rally controller;
- the development detector runs at 320×240 and 10 Hz, below the production
  perception contract;
- no real-robot claims should be made from the current controller limits.

## Repository hygiene

The generic Cartesian reaching, SmolVLA tutorial, and obstacle-reaching RL
prototype were removed. Keep new code and documentation tennis-specific. Large
generated datasets belong under ignored `artifacts/` or external storage;
tracked `results/tennis/` should contain compact, reproducible reports. Small
development checkpoints may live under `checkpoints/tennis/` when their source
data, evaluation split, and hash are documented.

Before each report-producing audit, require a clean tracked worktree so the
report records `tracked_files_dirty: false`. Finish changes in this order:

```bash
scripts/run python -m unittest discover -s tests -v
scripts/run python -m compileall -q tennis_vla examples tests
git diff --check
git status --short
```

Review changes to `configs/tennis/roadmap.yaml`, `docs/tennis_vla_plan.md`, and
this file whenever a milestone or reserved split changes.
