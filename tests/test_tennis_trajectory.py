from __future__ import annotations

import unittest

import numpy as np

from tennis_vla.environment import (
    PHASE_ONE_CONTACT_ENVELOPE,
    ProgrammableFeeder,
    make_tennis_contact_model,
)
from tennis_vla.planning import RacketIKConfig
from tennis_vla.planning import (
    MINIMUM_JERK_PEAK_ACCELERATION,
    MINIMUM_JERK_PEAK_SPEED,
    earliest_feasible_arrival,
    plan_intercept_arrivals,
    sample_minimum_jerk,
    track_intercept_arrival,
)


class TennisTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = make_tennis_contact_model()

    def test_minimum_jerk_has_zero_boundary_velocity_and_acceleration(self) -> None:
        start = np.array([-0.5, 0.25])
        target = np.array([1.5, -0.75])
        at_start = sample_minimum_jerk(start, target, 0.0, 2.0)
        at_end = sample_minimum_jerk(start, target, 2.0, 2.0)
        np.testing.assert_allclose(at_start[0], start)
        np.testing.assert_allclose(at_start[1], 0.0)
        np.testing.assert_allclose(at_start[2], 0.0)
        np.testing.assert_allclose(at_end[0], target)
        np.testing.assert_allclose(at_end[1], 0.0, atol=1e-14)
        np.testing.assert_allclose(at_end[2], 0.0, atol=1e-14)

    def test_analytical_peak_constants_match_sampled_trajectory(self) -> None:
        samples = [
            sample_minimum_jerk(np.zeros(1), np.ones(1), time, 1.0)
            for time in np.linspace(0.0, 1.0, 10_001)
        ]
        sampled_speed = max(abs(state[1][0]) for state in samples)
        sampled_acceleration = max(abs(state[2][0]) for state in samples)
        self.assertAlmostEqual(sampled_speed, MINIMUM_JERK_PEAK_SPEED, places=7)
        self.assertAlmostEqual(
            sampled_acceleration,
            MINIMUM_JERK_PEAK_ACCELERATION,
            places=6,
        )

    def test_phase_one_feed_has_a_feasible_arrival_from_launch(self) -> None:
        _, flight = ProgrammableFeeder(
            PHASE_ONE_CONTACT_ENVELOPE
        ).sample_legal_feed(1)
        plans = plan_intercept_arrivals(
            self.model,
            flight,
            ik_config=RacketIKConfig(restarts=4, maximum_iterations=180),
            seed=40_001,
        )
        selected = earliest_feasible_arrival(plans)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertGreaterEqual(selected.minimum_joint_limit_margin_rad, 0.03)
        self.assertLessEqual(selected.maximum_joint_speed_rad_s, 4.0)
        self.assertLessEqual(selected.maximum_joint_acceleration_rad_s2, 15.0)

        tracking = track_intercept_arrival(self.model, selected)
        self.assertTrue(tracking.passed, tracking.failure_reasons)
        self.assertLess(tracking.final_joint_tracking_error_rad, 0.01)
        self.assertLess(tracking.racket_position_tracking_error_m, 0.01)
        self.assertLess(tracking.racket_normal_tracking_error_deg, 1.0)
        self.assertEqual(tracking.unexpected_contact_steps, 0)


if __name__ == "__main__":
    unittest.main()
