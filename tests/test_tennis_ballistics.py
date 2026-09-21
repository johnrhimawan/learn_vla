from __future__ import annotations

import unittest

import numpy as np

from tennis_vla.ballistics import BallFlightConfig, simulate_ball_flight


class BallFlightTests(unittest.TestCase):
    def test_vacuum_flight_matches_closed_form_solution(self) -> None:
        config = BallFlightConfig(
            drag_coefficient=0.0,
            dt_s=0.001,
            duration_s=0.5,
        )
        position = np.array([1.0, -0.5, 10.0])
        velocity = np.array([2.0, 1.0, 3.0])
        result = simulate_ball_flight(position, velocity, config)
        duration = config.duration_s
        expected_position = position + velocity * duration
        expected_position[2] -= 0.5 * config.gravity_m_s2 * duration**2
        expected_velocity = velocity.copy()
        expected_velocity[2] -= config.gravity_m_s2 * duration
        np.testing.assert_allclose(result.positions_m[-1], expected_position, atol=1e-9)
        np.testing.assert_allclose(result.velocities_m_s[-1], expected_velocity, atol=1e-9)
        self.assertEqual(result.bounce_count, 0)

    def test_reference_feed_clears_net_and_bounces_in_near_court(self) -> None:
        result = simulate_ball_flight(
            position_m=np.array([10.5, 0.0, 1.4]),
            velocity_m_s=np.array([-19.0, 0.6, 4.5]),
        )
        self.assertIsNotNone(result.net_crossing_m)
        self.assertGreater(result.net_crossing_m[2], 0.914)
        self.assertIsNotNone(result.first_bounce_m)
        self.assertGreaterEqual(result.first_bounce_m[0], -11.885)
        self.assertLessEqual(result.first_bounce_m[0], 0.0)
        self.assertLessEqual(abs(result.first_bounce_m[1]), 4.115)

    def test_seedless_simulation_is_exactly_repeatable(self) -> None:
        inputs = (np.array([9.0, 1.0, 2.0]), np.array([-13.0, -0.5, 2.5]))
        first = simulate_ball_flight(*inputs)
        second = simulate_ball_flight(*inputs)
        np.testing.assert_array_equal(first.positions_m, second.positions_m)
        np.testing.assert_array_equal(first.velocities_m_s, second.velocities_m_s)


if __name__ == "__main__":
    unittest.main()
