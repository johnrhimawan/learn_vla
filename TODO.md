# TODO

Open work on the **mobile base**, plus items found while doing it.

This file tracks the mobile-base arc only. `HANDOFF.md` remains the living
engineering state and owns the M2 queue (seed 10096, planner latency, the v1
expanded audit, freezing the planner). Where the two touch, HANDOFF wins.

Why this arc exists: M6 (`closed_loop_rallies`) is blocked by the embodiment.
The arm reaches y ∈ [-0.94, +0.91] of an 8.23 m singles court, only 42/100
feeds on the broad envelope are reachable, and `PHASE_ONE_CONTACT_ENVELOPE` was
narrowed to work around exactly that. `docs/tennis_vla_plan.md:310` already
schedules the change.

## Done

- **Stage 0** — arm addressed by `EmbodimentLayout` (resolved by joint name)
  instead of literal `[:7]`, across 125 sites. Four index spaces kept separate:
  `arm_joints`, `arm_dof`, `arm_actuators`, `arm_qpos`.
- **Stage 1** — `FixedBase()` / `MobileBase()` on `make_tennis_contact_model`
  and `make_sawyer_racket_spec`. Slide-x, slide-y, yaw hinge on the Sawyer root
  body; values are displacements from the bolted stance, so zero reproduces the
  fixed-base geometry.

Both verified against `results/tennis/canonical_strike_execution_v1.json`
(`mujoco_execution`, `court_bounce_calibration`, `passed` identical). On the
mobile model the canonical strike contacts at the same 1.492 s, lands legally,
recovers, and holds station within 4.3 mm.

## Stage 2 — stance selection (next)

Nothing commands the base yet. This is the substantial stage.

- [ ] **Make contact bounds base-relative.** `StrikeSearchConfig`
      (`strike.py:91-96`) and `find_kinematic_intercepts`
      (`intercept.py:362-364`) hold world-frame numbers tuned to a base at
      x = -10.6. They must become offsets from the stance.
- [ ] **Unify the duplicated bounds.** `plan_intercept_arrivals`
      (`trajectory.py`) calls `find_kinematic_intercepts` *without* forwarding
      bounds, so it silently uses the `intercept.py` copy. Two independent sets
      of the same constants.
- [ ] **New `tennis_vla/stance.py`.** Given predicted flight, base limits and
      current base state, propose contact-time stances reachable in the time
      available. `find_kinematic_intercepts` iterates only over ball time
      samples, so stance belongs outside it or as a new argument.
- [ ] **Two-phase trajectory.** Today the strike is a single quintic from ready
      to contact with `duration_s = candidate.time_s`; there is no travel phase,
      and `QuinticJointTrajectory.from_boundary_conditions` takes only
      start/end, no waypoint. Chain travel-then-swing with continuity at the
      junction.
- [ ] **Freeze the base at contact.** `minimum_infinity_joint_velocity` solves
      the minimax LP by vertex enumeration, `C(columns, active) × 2^active`:
      240 today, **4608** at 10 DoF with only wrist roll pinned — ~19× more work
      in the innermost loop, on planning that already averages 44.9 s. Pinning
      the base restores 240 *and* keeps the verified strike mechanics unchanged.
      Reuse the existing `fixed_joint_velocities` partition; pass the base
      indices. Cost: no "stepping into the shot" racket speed — revisit after
      Stage 3.
- [ ] **Heterogeneous motion limits.** `SimulationJointMotionLimits` is three
      scalars in rad. Add a sibling `BaseMotionLimits` in m rather than
      overloading it, and make every utilization ratio per-DoF *before* the max,
      or base motion will dominate or vanish on unit scale alone.
- [ ] **Unit-safe IK.** `maximum_step_rad` clips a step vector that would mix m
      and rad, and `joint_limit_margin_rad = 0.03` does not mean the same thing
      for 3 cm of travel. Apply a diagonal scaling before the step clip and
      before LP enumeration.
- [ ] **IK seeding.** The task is rank-5 (3 position + 2 from `np.cross` on the
      face normal), so the nullspace grows from 2-D to ~5-D. `posture_weight`
      defaults to 0.0 and uniform random restarts over a 10-D box are far too
      sparse at `restarts=24`. Seed the base deterministically from predicted
      ball lateral position; keep arm seeding from ready.
- [ ] **Recovery to a stance.** `plan_ready_recovery_trajectory` returns to one
      hardcoded pose and is a hard gate on every plan. With base travel it
      becomes pose + recovery *position* and will dominate the duration search,
      likely making the 1.0 s candidate infeasible. Recover to a fixed home
      stance for now; strategic positioning is M6.
- [ ] **Extend the renderer** (`examples/render_tennis_strike.py`) to draw the
      base path, so stance bugs (sliding, jitter, arriving late) are visible
      rather than inferred from JSON.

## Stage 3 — broaden the envelope

- [ ] Restore the default broad `FeedEnvelope` and measure against the 42%
      fixed-base baseline (`intercept_kinematic_audit_v0.json`).
- [ ] **Allocate new seed ranges** for the mobile embodiment. Existing splits
      are fixed-base evidence and must not be reused or retuned. Mobile results
      start a new series.
- [ ] **Choose base speed deliberately.** Flight time is ~1.5 s: at 1 m/s the
      base covers 1.5 m and mobility is marginal; at 3 m/s it covers ~4.5 m and
      the court opens up. Start at 3 m/s and 6 m/s², record as project
      simulation limits alongside 4 rad/s and 15 rad/s², claim nothing about
      hardware.
- [ ] Keep `diagnostics=` populated so `terminal_rejection_stage_counts` stays
      comparable across embodiments.

## Small, independent

- [ ] **Regenerate `canonical_strike_execution_v1.json`.** It predates the
      `face_yaw_degrees` field added in 27fe384, so a rerun differs in that one
      key. Needs a clean tracked worktree so the report records
      `tracked_files_dirty: false`.
- [ ] **Retire the stale `final-heldout` profile.** It still names the failed
      v0 range 12000–12199 (`examples/tennis_active_strike_audit.py:225`). Add
      a v1 final profile only once the planner is frozen — and not before, per
      the seed contract.

## Blocked / prerequisite

- [ ] **Planner latency** (HANDOFF item 2) is effectively a prerequisite for
      Stage 2: stance adds a search dimension to planning that already averages
      44.9 s and peaks at 129.6 s. Keep stance search analytic and cheap.

## Documentation to update when the base lands

All of these encode a fixed base and are currently correct only for
`FixedBase()`:

- `README.md:4` — "fixed-base 7-DoF Sawyer arm"
- `docs/tennis_vla_plan.md:6,39` — embodiment; `:310` — the mobile-base gate
- `configs/tennis/roadmap.yaml:7` — `initial_embodiment`, plus M2/M6 work items
- `CLAUDE.md` — scene conventions (contact boxes are world-frame today)
- `HANDOFF.md` — engineering state

## Out of scope (recorded, not planned)

- Rally loop, opponent model, self-play, language-conditioned profiles.
- Spin, and pre-bounce interception (volleys) — the search is
  `post_bounce_only`, and a volley is played near the net where even a mobile
  base within the current travel range does not go.
- **Perception is coupled but untouched.** The stereo rig is world-fixed
  (`environment.py:17-21`) and `strike_plane_x_m = -9.25`
  (`flight_dataset.py:181`) assumes a single strike plane, which stops being
  well-defined once the base roams. Needs its own milestone.
