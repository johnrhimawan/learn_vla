from __future__ import annotations

import unittest

import numpy as np

from tennis_vla.environment import make_tennis_contact_model
from tennis_vla.execution import (
    audit_court_bounce,
    execute_strike,
    simulate_mujoco_ball_flight,
)
from tennis_vla.strike import plan_safe_center_strikes


CANONICAL_POSITION_M = np.array([10.5, 0.0, 1.4])
CANONICAL_VELOCITY_M_S = np.array([-17.75, 0.0, 4.6])


class TennisExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = make_tennis_contact_model()
        cls.flight = simulate_mujoco_ball_flight(
            cls.model,
            CANONICAL_POSITION_M,
            CANONICAL_VELOCITY_M_S,
        )
        plans = plan_safe_center_strikes(cls.model, cls.flight, seed=40_001)
        if not plans:
            raise RuntimeError("canonical feed has no active-strike plan")
        cls.plan = plans[0]

    def test_mujoco_court_bounce_matches_analytical_calibration(self) -> None:
        result = audit_court_bounce(
            self.model,
            CANONICAL_POSITION_M,
            CANONICAL_VELOCITY_M_S,
        )
        self.assertTrue(result.passed)
        self.assertLessEqual(result.time_error_s, 0.005)
        self.assertLessEqual(result.position_error_m, 0.050)
        self.assertLessEqual(result.velocity_error_m_s, 0.250)

    def test_canonical_strike_contacts_and_returns_live_ball(self) -> None:
        result = execute_strike(
            self.model,
            self.plan,
            CANONICAL_POSITION_M,
            CANONICAL_VELOCITY_M_S,
        )
        self.assertTrue(result.passed, result.failure_reasons)
        self.assertTrue(result.contacted)
        self.assertTrue(result.separated)
        self.assertIsNotNone(result.measured_return)
        self.assertTrue(result.measured_return.legal_first_bounce)
        self.assertGreaterEqual(result.measured_return.net_clearance_m, 0.10)
        self.assertEqual(result.unexpected_contact_steps, 0)
        self.assertLessEqual(result.contact_position_error_m, 0.03)
        self.assertIsNotNone(result.recovery)
        self.assertTrue(result.recovery.passed, result.recovery.failure_reasons)
        self.assertLessEqual(result.recovery.final_joint_error_rad, 0.01)
        self.assertLessEqual(
            result.recovery.maximum_actual_joint_acceleration_rad_s2,
            15.0,
        )


if __name__ == "__main__":
    unittest.main()
