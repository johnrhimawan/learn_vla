from __future__ import annotations

import unittest

import mujoco
import numpy as np

from tennis_vla.arm import audit_workspace, home_configuration, make_sawyer_racket_model
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

    def test_home_pose_is_finite_and_above_floor(self) -> None:
        data = mujoco.MjData(self.model)
        data.qpos[:] = home_configuration(self.model)
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
