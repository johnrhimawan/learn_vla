# Tennis VLA Project Plan

The target is a robot that sees an incoming tennis ball, understands a shot
instruction, moves a racket to the right contact state, and lands a legal
return in the requested part of the opposite court. The first complete system
will be a fixed-base 7-DoF arm returning programmable feeds after one bounce.
That is already a serious perception and control problem, but it removes
locomotion and opponent strategy until the strike system works.

The longer-term success test is a ten-shot singles rally with commands such as
“deep cross-court forehand,” “safe through the middle,” and “down-the-line
backhand.” Serving, footwork on a mobile base, score-aware strategy, and play
against a human come after reliable rallying.

## Why the policy is hierarchical

A regulation ball crosses the workspace too quickly for a large VLA to close a
joint-level loop by itself. The system will use four rates:

| Layer | Rate | Responsibility |
| --- | ---: | --- |
| Safety monitor | 1 kHz | joint, torque, workspace, and human exclusion limits |
| Joint controller | 250 Hz | track a dynamically feasible racket trajectory |
| Strike policy | 50 Hz | update intercept time, racket pose, and velocity |
| VLA planner | 5–10 Hz | interpret language and choose stroke and landing intent |

SmolVLA receives camera views, proprioception, ball-track history, and an
instruction. Its action chunk represents a compact stroke plan: stroke family,
contact time and pose, racket velocity, and landing target. An optimizer or
fast residual controller turns that plan into joint commands. This division
keeps language and vision in the VLA while deterministic safety and fast
tracking remain outside it.

## Manageable milestones

### M0 — Court, ball, and contact physics

Build the ITF singles court, net, ball feeder, racket collision geometry, and a
fixed-base 7-DoF arm. Validate free flight, bounce, net clearance, and racket
impulse separately. The first ball-flight implementation is already in
`tennis_vla/ballistics.py`; it intentionally omits spin until the no-spin model
has calibration data.

The simulation reference arm is the Apache-2.0 Sawyer model from MuJoCo
Menagerie 2026.9.0. Its seven joints and longer reach are useful for the
fixed-base contact study. The model archive and checksum are pinned in
`tennis_vla/arm.py`, and the generated racket is attached at the wrist. A
20,000-pose audit found a maximum sampled racket-center radial reach of 1.31 m.
The MJCF does not provide authoritative joint-velocity limits, so this is a
kinematic reference and does not select the eventual physical robot.

Exit gate: deterministic tests cover flight, bounce, net collision, court
bounds, racket contact, and seeded randomization. A visual rollout runs on the
Mac without training.

Core M0 is complete for the no-spin model. The integrated contact probe sends a
5.0 m/s ball into the held racket and measures a 3.846 m/s rebound, within
0.054 m/s of the independent first-order impact prediction. Spin and flexible
string-bed calibration remain a follow-up before high-speed training.

### M1 — Ball perception and prediction

Render two fixed cameras plus an optional wrist camera with randomized light,
court texture, ball color, motion blur, and camera calibration. Train a detector
and temporal estimator to infer 3D position, velocity, spin class, bounce time,
and the reachable intercept window. Keep true ball state available only to the
simulator, labels, reward, and an asymmetric critic.

Exit gate: held-out 3D position RMSE at most 5 cm and predicted contact-time
error at most 25 ms over the phase-one feeder envelope.

### M2 — Racket control and contact

Start with a stationary ball, then slow feeds, then feeds between 8 and 18 m/s.
Use inverse kinematics and trajectory optimization to create a privileged
oracle. It selects a reachable contact state and tracks it with the fast joint
controller. This becomes both the system fallback and the demonstration
generator.

Exit gate: at least 95% racket contact, no joint-limit or workspace violations,
and reproducible outgoing-ball speed for held-out feeder seeds.

### M3 — Behavior-cloned visual returns

Generate oracle trajectories across feed position, speed, bounce, camera,
dynamics, and landing target. Convert successful episodes to LeRobot Dataset
v3 with cameras, proprioception, instruction, and action chunks. Fine-tune
SmolVLA on the H200 and evaluate it with privileged inputs removed.

Exit gate: at least 70% legal returns across held-out seeds, with no more than a
10-point drop when the language wording is paraphrased.

### M4 — Reinforcement learning for timing and placement

Freeze the behavior-cloned VLA first. Train a bounded SAC residual over contact
time, pose, and racket velocity. Seed replay with oracle demonstrations. The
actor uses only deployable observations; an asymmetric critic may use exact
ball and contact state. Reward components are racket contact, net clearance,
legal first bounce, distance to the requested landing zone, recovery pose,
energy, and hard safety penalties.

RL is valuable here because a demonstration says which swing the oracle chose,
while the simulator can score the actual landing point. Tiny timing and racket
angle errors have large effects after contact, and enumerating corrective
demonstrations for every miss is inefficient.

Exit gate: at least 85% legal returns and 60% target-zone accuracy on 100
held-out feeds. After the residual policy is stable, advantage-weight successful
replay and fine-tune the VLA action expert, then distill the residual into one
policy for deployment.

### M5 — Language-conditioned shot selection

Balance demonstrations and RL tasks across cross-court, down-the-line, safe
middle, deep, short, forehand, and backhand intents. Hold out paraphrases and
some instruction/trajectory combinations. Evaluate whether changing only the
instruction changes the landing distribution in the requested direction.

Exit gate: at least 85% instruction compliance on unseen wording without
reducing legal-return rate below the M4 gate.

### M6 — Closed-loop rallies

Replace the one-shot feeder with an opponent return model. Add recovery to a
ready pose, shot-to-shot memory, reachability decisions, and rally-level reward.
Only after fixed-base reach limits are understood should the embodiment gain a
linear rail or mobile base.

Exit gate: median rally length of ten over 100 seeded rallies, with results
reported separately by feed speed, spin class, and requested shot.

### M7 — Guarded sim-to-real transfer

Begin with an enclosed cell, physical emergency stop, hardware joint/torque
limits, and ITF Stage-1 green balls. Calibrate cameras and ball dynamics from
real trajectories, randomize the remaining sim gap, and use human intervention
for online RL only after the deterministic controller is reliable. Regulation
Type-2 balls and any human-facing trial are later gates.

Exit gate: 1,000 automated Stage-1 feeds without a safety-envelope violation,
followed by an explicit hardware review before increasing ball speed or adding
a person.

## Data products

| Dataset | Contents | Purpose |
| --- | --- | --- |
| `tennis-flight-v0` | randomized rendered ball flights with privileged tracks | perception and prediction |
| `tennis-strike-oracle-v0` | oracle contact and trajectory solutions | low-level imitation |
| `tennis-vla-bc-v0` | images + state + language → stroke/action chunks | SmolVLA fine-tuning |
| `tennis-rl-replay-v0` | transitions, rewards, interventions, VLA proposals | SAC and advantage weighting |
| `tennis-real-calibration-v0` | synchronized real cameras and measured bounces | physics and camera calibration |

Splits are by feeder seed, camera layout, and physics parameters rather than by
random frames. This prevents adjacent frames from the same flight appearing in
both training and evaluation. Every dataset records simulator commit, config,
random seed, units, camera calibration, and model/checkpoint hashes.

## Mac Mini and H200 responsibilities

The Mac runs the visual simulator, deterministic physics regressions, dataset
inspection, policy integration, and short MPS inference checks. A frozen
evaluation suite remains local so every checkpoint is compared on identical
feeds.

The H200 runs SmolVLA fine-tuning, residual SAC, advantage-weighted distillation,
and large evaluation sweeps. The existing CUDA container remains useful. The
staged learner design is in `configs/tennis/h200_training.yaml`; M2 freezes its
final action schema after the racket controller establishes reachable units and
bounds. LeRobot's actor/learner transport can be reused; the SmolVLA residual
and distillation path is custom.

## Evaluation contract

The fast regression suite is defined in `configs/tennis/eval_v0.json`. Each
report must separate contact rate, legal return rate, target-zone accuracy,
contact-time error, landing error, safety violations, and latency. Results are
also broken out by feed speed, lateral position, spin class, instruction, and
whether the condition was represented in training.

The first official baseline is recorded only after M0–M2 produce a controller
that can physically contact the ball. A random or cross-embodiment SmolVLA
action has no meaningful tennis units, so the pre-training comparison will
include the oracle, a do-nothing controller, and the unchanged SmolVLA base
adapted only through a versioned action schema.

## Immediate work queue

1. Validate the headless ball-flight scenario and inspect it in the MuJoCo
   viewer. **Implemented.**
2. Add the full court, net collision, configurable feeder, and trajectory tests.
   **Implemented for the no-spin model.**
3. Select and pin a 7-DoF MJCF reference arm and audit its kinematic workspace.
   **Implemented with Sawyer; physical hardware remains open.**
4. Integrate the arm, racket, ball, and court into one MuJoCo contact scene and
   validate the outgoing-ball map against the first-order impact model.
   **Implemented for normal, stationary-racket impact.**
5. Add calibrated spin and string-bed response.
6. Implement the privileged intercept oracle and create `tennis-strike-oracle-v0`.

The machine-readable status and gates live in `configs/tennis/roadmap.yaml`.

## References

- [2026 ITF Rules of Tennis](https://www.itftennis.com/media/7221/2026-rules-of-tennis-english.pdf)
- [MuJoCo force callbacks](https://mujoco.readthedocs.io/en/latest/APIreference/APIglobals.html)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [LeRobot SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla)
- [LeRobot HIL-SERL actor/learner workflow](https://huggingface.co/docs/lerobot/main/hilserl)
