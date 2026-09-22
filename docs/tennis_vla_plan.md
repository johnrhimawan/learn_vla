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
5.0 m/s ball into the held racket and measures a 3.912 m/s rebound, within
0.012 m/s of the independent first-order impact prediction. The probe now uses
the collision-free tennis-ready pose shared by the controller. Spin and flexible
string-bed calibration remain a follow-up before high-speed training.

### M1 — Ball perception and prediction

Render two fixed cameras plus an optional wrist camera with randomized light,
court texture, ball color, motion blur, and camera calibration. Train a detector
and temporal estimator to infer 3D position, velocity, spin class, bounce time,
and the reachable intercept window. Keep true ball state available only to the
simulator, labels, reward, and an asymmetric critic.

Exit gate: held-out 3D position RMSE at most 5 cm and predicted contact-time
RMSE at most 25 ms over the phase-one feeder envelope.

The first canonical-rendering baseline is implemented. Two calibrated cameras
behind the near baseline feed an illumination-tolerant color segmenter, ray
triangulator, and five-frame local velocity fit. On 25 held-out feeder seeds it
detected all 450 stereo frames, measured 5.9 mm 3D position RMSE, and predicted
the x=-9.25 m strike-plane crossing at least 150 ms ahead with 10.7 ms RMSE.
These figures pass the numeric thresholds in the canonical scene, but M1
remains open: the baseline has no domain randomization, motion blur, spin
classifier, or learned temporal model. The versioned report is
`results/tennis/perception_baseline_v0.json`.

The x=-9.25 m plane is an early timing reference, not a commanded racket pose.
The arm workspace audit found no sampled racket centers at that plane. M2 must
choose a later contact pose from the reachable post-bounce trajectory.

The `tennis-flight-v0` generator now provides the next M1 data layer. Each
episode receives its own court, line, ball, net, lighting, camera pose,
field of view, exposure, noise, and gamma sample. It writes stereo PNGs, exact
camera calibration, ball position and velocity, bounce state, and time to the
strike plane. Train, validation, and test episodes use non-overlapping feeder
seed ranges. A validator checks manifest counts, split isolation, label shapes,
and every referenced image. The tracked smoke report contains four episodes
and 86 images generated from clean revision `de0c8e6`; every validation check
passes. Spin and texture-map randomization remain open.

The fixed color baseline fails the strict randomized smoke gate. It detects
28/43 stereo frames (65.1%), produces contact-time estimates for three of four
episodes, and therefore fails both coverage gates. Its 1.57 cm position RMSE
applies only to detected frames, so coverage prevents that selective result
from being treated as a pass. This measured failure is the training target for
the learned detector and temporal estimator.

The learned detector pipeline is implemented as a full-resolution pixel
heatmap model. It deliberately avoids spatial downsampling because the ball can
occupy one pixel in early receiving-half frames. A depthwise 5x5 layer adds
local shape context without discarding that pixel. Training uses only the train
split, selects a checkpoint by validation pixel RMSE, and evaluates the frozen
checkpoint through the same calibrated stereo and timing gate as the fixed
baseline. A small smoke probe verified optimization and checkpoint loading;
the next recorded experiment expands the number of visual domains before any
claim about held-out performance.

The loss mines the 256 hardest background pixels in every image. This is
necessary because averaging across roughly 77,000 pixels hid rare bright court
and robot false positives. In the smoke diagnostic, hard-negative mining
reduced test pixel RMSE from 38.8 px to 0.41 px; the broader split remains the
authoritative development evaluation.

The first broader development experiment uses 100 training, 20 validation, and
20 test domains at 320x240 and 10 Hz. The frozen checkpoint has 769 parameters
and uses only RGB at inference. Calibrated stereo rejects pairs whose
observation rays remain more than 10 cm apart, and the local velocity fit uses
a fixed 100 ms history so its behavior does not change with frame rate. On the
untouched 20-domain test split, the system retains 99.48% valid stereo coverage,
measures 2.69 cm 3D position RMSE, predicts contact for every episode, and
reaches 15.1 ms contact-time RMSE. This passes the development numeric gate.
M1 stays in progress until the 512x384, 50 Hz production split passes and the
spin class and reachable intercept window are implemented. The dataset, training,
baseline, and learned evaluation reports are stored under `results/tennis/`,
and the small selected checkpoint is stored under `checkpoints/tennis/`.

### M2 — Racket control and contact

Start with a stationary ball, then slow feeds, then feeds between 8 and 18 m/s.
Use inverse kinematics and trajectory optimization to create a privileged
oracle. It selects a reachable contact state and tracks it with the fast joint
controller. This becomes both the system fallback and the demonstration
generator.

Exit gate: at least 95% racket contact, no joint-limit or workspace violations,
and reproducible outgoing-ball speed for held-out feeder seeds.

The first privileged oracle layer now solves racket-center position and face
normal with damped least-squares inverse kinematics and deterministic restarts.
It searches post-bounce ball trajectories for contact poses, including the ball
radius and racket thickness offset. The search rejects contacts below 0.60 m,
where early tracking experiments showed the wrist or racket could intersect the
court. Across the first 100 broad feeder seeds, 42 have at least one buffered,
floor-safe pose. Their first reachable ball positions span x=-10.08 to -9.60 m,
y=-0.94 to 0.91 m, and z=0.60 to 1.31 m. The result is recorded in
`results/tennis/intercept_kinematic_audit_v0.json`. This 42% ceiling is evidence
for a restricted M2 feeder curriculum; it is not a contact-rate result.

The phase-one contact curriculum narrows source lateral position to +/-0.5 m,
lateral velocity to +/-0.5 m/s, launch height to 1.2-1.6 m, forward speed to
16-18 m/s, and vertical speed to 4-5 m/s. It leaves the broader perception
distribution unchanged. Separate train, validation, and test seed ranges must
each exceed 95% kinematic eligibility before dynamic swing work uses them.
The IK solver reserves 0.03 rad at every joint limit. With that buffer, the
recorded 200-feed splits reach 98.5% for train, 97.0% for validation, and 96.5%
for the untouched test seeds. The report is
`results/tennis/contact_curriculum_kinematic_audit_v0.json`. These rates prove
pose eligibility and do not yet prove executed contact.

The next oracle layer connects a collision-free tennis-ready pose to each
contact pose with a quintic minimum-jerk joint trajectory. Its current
simulation contract limits each joint to 4 rad/s, 15 rad/s^2, and the same
0.03 rad joint-limit buffer. Planning begins at the programmable feeder trigger,
which is known to the privileged M2 oracle. On the held-out test split, 191 of
200 feeds (95.5%) have an analytically feasible arrival. Seven lack a buffered,
floor-safe IK pose and two exceed a motion limit. Selected arrivals have a
median peak joint speed of 1.78 rad/s and a 95th-percentile peak of 3.11 rad/s;
their 95th-percentile peak acceleration is 6.22 rad/s^2. The evidence is in
`results/tennis/dynamic_intercept_audit_v0.json`.

The 250 Hz position servo adds velocity-damping compensation and unconstrained
rigid-body inverse-dynamics feedforward, then runs at 1 kHz MuJoCo physics. It
checks actual joint speed and acceleration, final pose error, joint margin,
command clipping, and unexpected collision. The held-out split passes 190 of
200 feeds (95.0%): seven have no floor-safe IK pose, two fail analytical motion
limits, and one collides during execution. Across the attempted held-out
trajectories, median final joint error is 0.0029 rad, median racket-position
tracking error is 1.3 mm, and the 95th-percentile actual joint speed and
acceleration are 3.14 rad/s and 8.71 rad/s^2. The report is
`results/tennis/arrival_tracking_audit_v0.json`.

These trajectories end at zero joint velocity, and the tracking audit parks the
ball away from the arm. The result validates collision-aware arrival control,
but it is not an active tennis swing or a contact-rate result. The project
motion limits are simulation assumptions; the reported feedforward torque has
no acceptance gate because the reference MJCF lacks authoritative actuator
limits.

The canonical active-strike checkpoint adds a quintic trajectory with nonzero
terminal joint velocity. It searches racket-face pitch and normal speed, solves
the redundant racket-velocity Jacobian with a deterministic minimum-infinity
norm solution, and rejects trajectories that exceed the existing speed,
acceleration, or joint-margin gates. For the center feed at `[10.5, 0, 1.4]` m
and `[-17.75, 0, 4.6]` m/s, it selects a 2.1 m/s racket-normal speed and predicts
a legal return. The clean analytical report is
`results/tennis/canonical_strike_plan_v0.json`.

The same strike now runs against the live MuJoCo ball. An explicit ball-court
contact pair matches the analytical canonical bounce at separation within 3 ms,
2.8 cm, and 0.153 m/s. A 250 Hz trajectory reference is interpolated by a 1 kHz
inverse-dynamics and feedback loop. It contacts the ball 2 ms before the planned
time, separates 10 ms later, stays within 0.0087 rad maximum joint-tracking
error, and produces a measured outgoing velocity of `[8.766, 0.016, 7.332]`
m/s. Analytical continuation from that measured state clears the net by 1.129
m and bounces legally at x=2.122 m. There are no clipped controls or unexpected
contacts. The execution report is
`results/tennis/canonical_strike_execution_v0.json`.

The active-strike development planner extends the action from normal racket
speed to a three-dimensional velocity with a vertical tangential component.
It evaluates multiple redundant IK branches, prefers plans with motion-limit
headroom, and rejects collisions or compensated commands that would clip during
the approach and 50 ms contact continuation. Most critically, its privileged
contact-state prediction now rolls the ball through the calibrated MuJoCo
court contact. The independent analytical flight remains a calibration check;
it is no longer assumed exact after the bounce.

On the 20-seed test-development prefix, all 20 feeds produce a safe plan, live
ball-racket contact, and a legal return. Every strict execution check passes.
Median contact-time error is 2.0 ms, median contact-position error is 1.78 cm,
median outgoing-velocity model error is 0.925 m/s, and the 95th-percentile
maximum joint-tracking error is 0.00945 rad. The minimum measured net clearance
is 1.835 m. The clean evidence is
`results/tennis/active_strike_development_v0.json`.

This completes the 20-feed development gate, not M2. Because those seeds drove
planner corrections, the final active-strike test is reserved at seeds
12000–12199. That untouched 200-feed audit, a bounded recovery trajectory after
separation, and hardware velocity, acceleration, and torque limits remain
before exporting `tennis-strike-oracle-v0`. Contact-phase acceleration is
recorded but still has
no hardware-derived acceptance gate.

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
5. Add the calibrated stereo scene and canonical ball-tracking benchmark.
   **Implemented; M1 remains in progress.**
6. Generate `tennis-flight-v0` with held-out lighting, texture, camera, blur,
   ball-color, and physics splits; train the learned M1 estimator. **The
   generator, calibration export, blur model, split discipline, and smoke
   validator are implemented; production generation and learning remain.**
7. Add calibrated spin and string-bed response.
8. Implement the privileged intercept oracle and create `tennis-strike-oracle-v0`.
   **Buffered floor-safe IK, zero-velocity minimum-jerk arrival, 250 Hz arrival
   tracking, and a 20-feed live active-strike development gate are implemented;
   the 200-feed held-out audit, recovery motion, and dataset export remain.**

The machine-readable status and gates live in `configs/tennis/roadmap.yaml`.

## References

- [2026 ITF Rules of Tennis](https://www.itftennis.com/media/7221/2026-rules-of-tennis-english.pdf)
- [MuJoCo force callbacks](https://mujoco.readthedocs.io/en/latest/APIreference/APIglobals.html)
- [MuJoCo contact parameters](https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [LeRobot SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla)
- [LeRobot HIL-SERL actor/learner workflow](https://huggingface.co/docs/lerobot/main/hilserl)
