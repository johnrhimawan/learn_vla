from __future__ import annotations

import unittest

import mujoco
import numpy as np

from tennis_vla.robot import arm_layout, tennis_ready_configuration
from tennis_vla.environment import ProgrammableFeeder, make_tennis_contact_model
from tennis_vla.planning import find_kinematic_intercepts, solve_racket_pose


class TennisInterceptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = make_tennis_contact_model()

    def test_solver_recovers_a_known_forward_kinematics_pose(self) -> None:
        data = mujoco.MjData(self.model)
        data.qpos[arm_layout(self.model).arm_qpos] = tennis_ready_configuration(
            self.model
        )
        mujoco.mj_forward(self.model, data)
        site = self.model.site("racket_center").id
        target_position = data.site_xpos[site].copy()
        target_normal = data.site_xmat[site].reshape(3, 3)[:, 2].copy()
        result = solve_racket_pose(
            self.model, target_position, target_normal, seed=17
        )
        self.assertTrue(result.converged)
        self.assertLess(result.position_error_m, 1e-9)
        self.assertLess(result.normal_error_deg, 1e-5)

    def test_seed_one_has_repeatable_post_bounce_intercepts(self) -> None:
        _, flight = ProgrammableFeeder().sample_legal_feed(1)
        first = find_kinematic_intercepts(
            self.model, flight, maximum_candidates=3, seed=91
        )
        second = find_kinematic_intercepts(
            self.model, flight, maximum_candidates=3, seed=91
        )
        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 3)
        np.testing.assert_allclose(
            [candidate.time_s for candidate in first],
            [candidate.time_s for candidate in second],
        )
        for candidate in first:
            self.assertTrue(candidate.solution.converged)
            self.assertLessEqual(candidate.solution.position_error_m, 0.015)
            self.assertLessEqual(candidate.solution.normal_error_deg, 5.0)
            self.assertGreater(candidate.ball_position_m[0], -10.1)
            self.assertLess(candidate.ball_position_m[0], -9.45)
            self.assertGreaterEqual(candidate.ball_position_m[2], 0.60)

    def test_reference_plane_is_not_assumed_to_be_the_contact_pose(self) -> None:
        unreachable = solve_racket_pose(
            self.model,
            np.array([-9.20, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            seed=7,
        )
        self.assertFalse(unreachable.converged)


if __name__ == "__main__":
    unittest.main()
