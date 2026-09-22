from __future__ import annotations

import unittest

import numpy as np

from tennis_vla.ballistics import simulate_ball_flight
from tennis_vla.environment import make_tennis_contact_model
from tennis_vla.strike import (
    QuinticJointTrajectory,
    minimum_infinity_joint_velocity,
    plan_safe_center_strikes,
    trajectory_is_execution_safe,
)


class TennisStrikeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = make_tennis_contact_model()

    def test_quintic_matches_all_boundary_conditions(self) -> None:
        start = np.array([-0.5, 0.2])
        target = np.array([1.0, -0.7])
        start_velocity = np.array([0.1, -0.2])
        target_velocity = np.array([0.4, 0.3])
        start_acceleration = np.array([0.2, 0.1])
        target_acceleration = np.array([-0.3, 0.2])
        trajectory = QuinticJointTrajectory.from_boundary_conditions(
            start,
            target,
            duration_s=1.7,
            start_velocity_rad_s=start_velocity,
            target_velocity_rad_s=target_velocity,
            start_acceleration_rad_s2=start_acceleration,
            target_acceleration_rad_s2=target_acceleration,
        )
        at_start = trajectory.sample(0.0)
        at_end = trajectory.sample(1.7)
        np.testing.assert_allclose(at_start[0], start, atol=1e-12)
        np.testing.assert_allclose(at_start[1], start_velocity, atol=1e-12)
        np.testing.assert_allclose(at_start[2], start_acceleration, atol=1e-12)
        np.testing.assert_allclose(at_end[0], target, atol=1e-12)
        np.testing.assert_allclose(at_end[1], target_velocity, atol=1e-12)
        np.testing.assert_allclose(at_end[2], target_acceleration, atol=1e-11)

    def test_minimum_infinity_velocity_balances_redundant_joints(self) -> None:
        solution = minimum_infinity_joint_velocity(
            np.array([[1.0, 1.0]]),
            np.array([1.0]),
        )
        np.testing.assert_allclose(solution, [0.5, 0.5], atol=1e-12)

    def test_canonical_feed_has_a_legal_active_return_plan(self) -> None:
        flight = simulate_ball_flight(
            np.array([10.5, 0.0, 1.4]),
            np.array([-17.75, 0.0, 4.6]),
        )
        self.assertTrue(flight.legal_first_bounce)
        plans = plan_safe_center_strikes(self.model, flight, seed=40_001)
        self.assertTrue(plans)
        plan = plans[0]
        self.assertTrue(plan.predicted_return.legal_first_bounce)
        self.assertGreaterEqual(plan.predicted_return.net_clearance_m, 0.10)
        self.assertGreater(
            np.linalg.norm(plan.contact_racket_velocity_m_s),
            1.0,
        )
        self.assertLessEqual(
            plan.trajectory_bounds.maximum_joint_speed_rad_s,
            4.0,
        )
        self.assertLessEqual(
            plan.trajectory_bounds.maximum_joint_acceleration_rad_s2,
            15.0,
        )
        self.assertGreaterEqual(
            plan.trajectory_bounds.minimum_joint_limit_margin_rad,
            0.03,
        )
        self.assertTrue(
            trajectory_is_execution_safe(self.model, plan.trajectory)
        )


if __name__ == "__main__":
    unittest.main()
