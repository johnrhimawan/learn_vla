from __future__ import annotations

import unittest

import numpy as np

from tennis_vla.ballistics import BallFlightConfig, simulate_ball_flight
from tennis_vla.court import TennisCourtSpec
from tennis_vla.feeder import PHASE_ONE_CONTACT_ENVELOPE, ProgrammableFeeder
from tennis_vla.impact import apply_racket_impact


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
        self.assertEqual(result.outcome, "airborne")

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
        self.assertTrue(result.legal_first_bounce)
        self.assertEqual(result.outcome, "legal_first_bounce")

    def test_low_flight_hits_the_net_and_stops(self) -> None:
        config = BallFlightConfig(
            drag_coefficient=0.0,
            gravity_m_s2=0.0,
            duration_s=1.0,
        )
        result = simulate_ball_flight(
            position_m=np.array([2.0, 0.0, 0.75]),
            velocity_m_s=np.array([-4.0, 0.0, 0.0]),
            config=config,
        )
        self.assertTrue(result.hit_net)
        self.assertEqual(result.outcome, "hit_net")
        self.assertLessEqual(result.net_clearance_m, 0.0)
        self.assertAlmostEqual(result.positions_m[-1, 0], 0.0, places=12)

    def test_ball_overlapping_a_singles_line_is_in(self) -> None:
        court = TennisCourtSpec()
        radius = 0.0335
        point = np.array([0.0, court.singles_half_width_m + radius * 0.5, 0.0])
        self.assertTrue(court.contains_singles_bounce(point, radius))
        point[1] = court.singles_half_width_m + radius * 1.1
        self.assertFalse(court.contains_singles_bounce(point, radius))

    def test_seeded_feeder_produces_repeatable_legal_feeds(self) -> None:
        feeder = ProgrammableFeeder()
        for seed in range(10):
            first, first_result = feeder.sample_legal_feed(seed)
            second, second_result = feeder.sample_legal_feed(seed)
            np.testing.assert_array_equal(first.position_m, second.position_m)
            np.testing.assert_array_equal(first.velocity_m_s, second.velocity_m_s)
            self.assertEqual(first.attempt, second.attempt)
            self.assertTrue(first_result.legal_first_bounce)
            self.assertTrue(second_result.legal_first_bounce)

    def test_moving_racket_reverses_and_accelerates_incoming_ball(self) -> None:
        incoming = np.array([-12.0, 1.0, -2.0])
        stationary = apply_racket_impact(incoming, np.zeros(3), np.array([1.0, 0.0, 0.0]))
        moving = apply_racket_impact(
            incoming,
            np.array([5.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        self.assertGreater(stationary[0], 0.0)
        self.assertGreater(moving[0], stationary[0])
        self.assertAlmostEqual(stationary[1], 0.82)
        self.assertAlmostEqual(stationary[2], -1.64)

    def test_seedless_simulation_is_exactly_repeatable(self) -> None:
        inputs = (np.array([9.0, 1.0, 2.0]), np.array([-13.0, -0.5, 2.5]))
        first = simulate_ball_flight(*inputs)
        second = simulate_ball_flight(*inputs)
        np.testing.assert_array_equal(first.positions_m, second.positions_m)
        np.testing.assert_array_equal(first.velocities_m_s, second.velocities_m_s)

    def test_contact_curriculum_is_narrower_than_perception_feeder(self) -> None:
        default = ProgrammableFeeder().envelope
        contact = PHASE_ONE_CONTACT_ENVELOPE
        self.assertLess(contact.source_y_m[1], default.source_y_m[1])
        self.assertGreater(contact.source_y_m[0], default.source_y_m[0])
        self.assertLess(
            contact.lateral_speed_m_s[1], default.lateral_speed_m_s[1]
        )
        for seed in range(10):
            _, result = ProgrammableFeeder(contact).sample_legal_feed(seed)
            self.assertTrue(result.legal_first_bounce)


if __name__ == "__main__":
    unittest.main()
