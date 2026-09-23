from __future__ import annotations

import unittest

import mujoco
import numpy as np

from tennis_vla.robot import (
    ARM_JOINT_NAMES,
    EmbodimentLayout,
    MobileBase,
    arm_layout,
    audit_workspace,
    home_configuration,
    make_sawyer_racket_model,
    tennis_ready_configuration,
)
from tennis_vla.environment import make_tennis_contact_model, probe_stationary_racket_contact


class TennisArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = make_sawyer_racket_model()

    def test_reference_arm_and_racket_schema(self) -> None:
        self.assertEqual(self.model.nq, 7)
        self.assertEqual(self.model.nu, 7)
        self.assertGreater(self.model.geom("racket_head").id, -1)
        self.assertGreater(self.model.site("racket_center").id, -1)
        self.assertGreater(float(self.model.body("tennis_racket").mass[0]), 0.1)

    def test_layout_resolves_the_arm_by_joint_name(self) -> None:
        layout = arm_layout(self.model)
        self.assertEqual(layout.arm_dof_count, len(ARM_JOINT_NAMES))
        self.assertFalse(layout.has_mobile_base)
        self.assertEqual(layout.base_dof_count, 0)
        resolved = [
            self.model.joint(index).name
            for index in range(
                layout.arm_joints.start, layout.arm_joints.stop
            )
        ]
        self.assertEqual(resolved, list(ARM_JOINT_NAMES))

    def test_layout_matches_the_fixed_base_prefix_slices(self) -> None:
        # Stage 0 must be a pure refactor: with no base joints the layout has
        # to reproduce the literal [:7] slices it replaced.
        for model in (self.model, make_tennis_contact_model()):
            layout = arm_layout(model)
            self.assertEqual(layout.arm_qpos, slice(0, 7))
            self.assertEqual(layout.arm_dof, slice(0, 7))
            self.assertEqual(layout.arm_joints, slice(0, 7))
            self.assertEqual(layout.arm_actuators, slice(0, 7))

    def test_layout_rejects_a_model_without_the_arm(self) -> None:
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="lonely").add_freejoint(name="drift")
        with self.assertRaises(ValueError):
            EmbodimentLayout.from_model(spec.compile())

    def test_mobile_base_adds_three_leading_degrees_of_freedom(self) -> None:
        model = make_tennis_contact_model(base=MobileBase())
        layout = arm_layout(model)
        self.assertEqual(model.nq, 17)
        self.assertEqual(model.nv, 16)
        self.assertEqual(model.nu, 10)
        self.assertTrue(layout.has_mobile_base)
        self.assertEqual(layout.base_dof_count, 3)
        # The base sorts ahead of the arm in qpos, but its actuators are
        # appended after the arm's.  A single index space would be wrong.
        self.assertEqual(layout.base_qpos, slice(0, 3))
        self.assertEqual(layout.arm_qpos, slice(3, 10))
        self.assertEqual(layout.arm_actuators, slice(0, 7))
        self.assertEqual(layout.base_actuators, slice(7, 10))

    def test_zero_base_configuration_is_the_bolted_stance(self) -> None:
        fixed = make_tennis_contact_model()
        mobile = make_tennis_contact_model(base=MobileBase())
        positions = []
        for model in (fixed, mobile):
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            positions.append(data.site_xpos[model.site("racket_center").id].copy())
        np.testing.assert_allclose(positions[0], positions[1], atol=1e-12)

    def test_trajectory_bounds_read_the_arm_range_not_a_prefix(self) -> None:
        # A prefix slice would read the base travel limits on a mobile model
        # and silently reject every arm trajectory.
        from tennis_vla.planning import QuinticJointTrajectory

        model = make_tennis_contact_model(base=MobileBase())
        ready = tennis_ready_configuration(model)
        trajectory = QuinticJointTrajectory.from_boundary_conditions(
            ready, ready, duration_s=1.0
        )
        bounds = trajectory.bounds(model)
        self.assertGreater(bounds.minimum_joint_limit_margin_rad, 0.0)

    def test_mobile_base_rejects_a_range_excluding_the_bolted_stance(self) -> None:
        with self.assertRaises(ValueError):
            MobileBase(travel_y_m=(1.0, 5.0)).validate()

    def test_home_pose_is_finite_and_above_floor(self) -> None:
        data = mujoco.MjData(self.model)
        data.qpos[arm_layout(self.model).arm_qpos] = home_configuration(self.model)
        mujoco.mj_forward(self.model, data)
        center = data.site("racket_center").xpos
        self.assertTrue(np.isfinite(center).all())
        self.assertGreater(center[2], 0.5)

    def test_workspace_audit_is_deterministic(self) -> None:
        first = audit_workspace(samples=64, seed=91)
        second = audit_workspace(samples=64, seed=91)
        self.assertEqual(first["arm"], second["arm"])
        self.assertEqual(first["racket_center_bounds_m"], second["racket_center_bounds_m"])
        self.assertEqual(first["radial_reach_m"], second["radial_reach_m"])

    def test_tennis_ready_pose_has_no_self_contact(self) -> None:
        data = mujoco.MjData(self.model)
        data.qpos[arm_layout(self.model).arm_qpos] = tennis_ready_configuration(
            self.model
        )
        mujoco.mj_forward(self.model, data)
        self.assertEqual(data.ncon, 0)

    def test_integrated_contact_scene_schema(self) -> None:
        model = make_tennis_contact_model()
        self.assertEqual(model.nq, 14)
        self.assertEqual(model.nv, 13)
        self.assertAlmostEqual(float(model.body("tennis_ball").mass[0]), 0.0577)
        self.assertGreater(model.geom("tennis_net").id, -1)
        self.assertGreater(model.camera("court_camera").id, -1)
        self.assertGreater(model.camera("camera1").id, -1)
        self.assertGreater(model.camera("camera2").id, -1)

    def test_integrated_racket_rebound_matches_impact_model(self) -> None:
        probe = probe_stationary_racket_contact(5.0)
        self.assertTrue(probe.contacted)
        self.assertIsNotNone(probe.separation_step)
        self.assertGreater(probe.outgoing_normal_speed_m_s, 0.0)
        self.assertAlmostEqual(
            probe.outgoing_normal_speed_m_s,
            probe.predicted_outgoing_normal_speed_m_s,
            delta=0.15,
        )


if __name__ == "__main__":
    unittest.main()
