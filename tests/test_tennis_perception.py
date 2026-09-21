from __future__ import annotations

import unittest

import mujoco
import numpy as np

from tennis_vla.arm import home_configuration
from tennis_vla.environment import make_tennis_contact_model
from tennis_vla.perception import (
    BallTrackEstimate,
    detect_yellow_ball,
    fit_constant_velocity_track,
    predict_x_crossing_time,
    triangulate_ball_from_images,
    triangulate_rays,
)


class TennisPerceptionTests(unittest.TestCase):
    def test_detector_uses_dark_illumination_tolerant_yellow(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        image[8:12, 13:18] = [105, 120, 16]
        image[:, :5] = [29, 92, 60]
        detection = detect_yellow_ball(image)
        self.assertIsNotNone(detection)
        np.testing.assert_allclose(detection.pixel_xy, [15.0, 9.5])
        self.assertEqual(detection.pixel_count, 20)

    def test_two_ideal_rays_recover_intersection(self) -> None:
        target = np.array([-7.0, 0.4, 1.3])
        origin1 = np.array([-12.5, -5.5, 3.4])
        origin2 = np.array([-12.5, 5.5, 3.4])
        result = triangulate_rays(
            origin1,
            target - origin1,
            origin2,
            target - origin2,
        )
        np.testing.assert_allclose(result.position_m, target, atol=1e-12)
        self.assertAlmostEqual(result.ray_gap_m, 0.0, places=12)

    def test_local_track_predicts_future_plane_crossing(self) -> None:
        times = np.array([0.00, 0.02, 0.04, 0.06, 0.08])
        velocity = np.array([-8.0, 0.5, 1.0])
        positions = np.array([-7.0, 0.1, 0.8]) + times[:, None] * velocity
        track = fit_constant_velocity_track(times, positions)
        np.testing.assert_allclose(track.velocity_m_s, velocity, atol=1e-12)
        self.assertAlmostEqual(predict_x_crossing_time(track, -9.0), 0.25)

    def test_crossing_prediction_rejects_past_or_stationary_plane(self) -> None:
        stationary = BallTrackEstimate(1.0, np.zeros(3), np.zeros(3))
        self.assertIsNone(predict_x_crossing_time(stationary, -1.0))
        moving_away = BallTrackEstimate(
            1.0, np.zeros(3), np.array([1.0, 0.0, 0.0])
        )
        self.assertIsNone(predict_x_crossing_time(moving_away, -1.0))

    def test_rendered_stereo_pair_recovers_ball_position(self) -> None:
        model = make_tennis_contact_model()
        data = mujoco.MjData(model)
        data.qpos[:7] = home_configuration(model)
        data.ctrl[:] = data.qpos[:7]
        expected = np.array([-7.0, 0.4, 1.3])
        qpos_address = int(model.joint("ball_free").qposadr[0])
        data.qpos[qpos_address : qpos_address + 3] = expected
        data.qpos[qpos_address + 3 : qpos_address + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, height=384, width=512)
        try:
            images = []
            for camera in ("camera1", "camera2"):
                renderer.update_scene(data, camera=camera)
                images.append(renderer.render().copy())
        finally:
            renderer.close()
        result = triangulate_ball_from_images(model, data, *images)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.position_m, expected, atol=0.02)


if __name__ == "__main__":
    unittest.main()
