from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from tennis_vla.perception import (
    apply_render_domain,
    postprocess_render,
    sample_render_domain,
)
from tennis_vla.environment import COURT_LINE_GEOMS, make_tennis_contact_model
from tennis_vla.perception import (
    generate_flight_dataset,
    validate_flight_dataset,
)
from tennis_vla.perception import (
    FlightBallImageDataset,
    LearnedBallDetector,
    train_ball_heatmap_detector,
)
from tennis_vla.perception import (
    camera_calibration,
    camera_ray,
    camera_ray_from_calibration,
)
from tennis_vla.perception import evaluate_color_stereo_baseline


class TennisFlightDatasetTests(unittest.TestCase):
    def test_domain_sampling_and_sensor_noise_are_repeatable(self) -> None:
        first = sample_render_domain(42)
        second = sample_render_domain(42)
        different = sample_render_domain(43)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertNotEqual(first.to_dict(), different.to_dict())

        image = np.full((16, 24, 3), 100, dtype=np.uint8)
        processed1 = postprocess_render(image, first, noise_seed=99)
        processed2 = postprocess_render(image, first, noise_seed=99)
        np.testing.assert_array_equal(processed1, processed2)

    def test_randomized_camera_calibration_matches_mujoco_pose(self) -> None:
        model = make_tennis_contact_model()
        data = mujoco.MjData(model)
        domain = sample_render_domain(123)
        apply_render_domain(model, data, domain)
        for camera_name, expected in zip(
            ("camera1", "camera2"), domain.cameras, strict=True
        ):
            camera_id = model.camera(camera_name).id
            rotation = data.cam_xmat[camera_id].reshape(3, 3)
            forward = np.asarray(expected.target_m) - np.asarray(expected.position_m)
            forward /= np.linalg.norm(forward)
            np.testing.assert_allclose(data.cam_xpos[camera_id], expected.position_m)
            np.testing.assert_allclose(-rotation[:, 2], forward, atol=1e-12)

            calibration = camera_calibration(model, data, camera_name, (120, 160))
            origin, principal_ray = camera_ray(
                model,
                data,
                camera_name,
                np.asarray(calibration["principal_point_px"]),
                (120, 160),
            )
            np.testing.assert_allclose(origin, expected.position_m)
            np.testing.assert_allclose(principal_ray, forward, atol=1e-12)
            exported_origin, exported_ray = camera_ray_from_calibration(
                calibration, np.asarray(calibration["principal_point_px"])
            )
            np.testing.assert_allclose(exported_origin, origin)
            np.testing.assert_allclose(exported_ray, principal_ray)

    def test_integrated_scene_has_visual_regulation_lines(self) -> None:
        model = make_tennis_contact_model()
        for name in COURT_LINE_GEOMS:
            geom = model.geom(name)
            self.assertEqual(int(geom.contype[0]), 0)
            self.assertEqual(int(geom.conaffinity[0]), 0)

    def test_small_dataset_has_disjoint_manifests_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "tennis-flight-v0"
            preview = Path(temporary_directory) / "preview.png"
            report = generate_flight_dataset(
                root,
                {"train": 1, "validation": 1, "test": 1},
                width=128,
                height=96,
                observation_hz=5.0,
                preview_path=preview,
            )
            self.assertEqual(report["episodes"], 3)
            self.assertGreater(report["frames"], 10)
            self.assertTrue(report["validation"]["manifest_counts_match"])
            self.assertTrue(report["validation"]["split_seeds_disjoint"])
            self.assertTrue(preview.is_file())
            self.assertEqual(validate_flight_dataset(root), report["validation"])

            frames = [
                json.loads(line)
                for line in (root / "frames.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            first = frames[0]
            self.assertEqual(len(first["ball"]["position_world_m"]), 3)
            self.assertEqual(len(first["ball"]["velocity_world_m_s"]), 3)
            self.assertGreater(first["prediction_targets"]["time_to_strike_plane_s"], 0)
            for relative_path in first["images"].values():
                with Image.open(root / relative_path) as image:
                    self.assertEqual(image.size, (128, 96))
                    self.assertEqual(image.mode, "RGB")

            baseline = evaluate_color_stereo_baseline(root)
            self.assertEqual(baseline["metrics"]["frames"], report["frames"])
            self.assertGreaterEqual(baseline["metrics"]["detection_rate"], 0.0)
            self.assertLessEqual(baseline["metrics"]["detection_rate"], 1.0)

            training_images = FlightBallImageDataset(
                root, "train", image_size=(64, 48)
            )
            sample = training_images[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 48, 64))
            self.assertEqual(tuple(sample["heatmap"].shape), (1, 48, 64))
            checkpoint = Path(temporary_directory) / "detector.pt"
            training = train_ball_heatmap_detector(
                root,
                checkpoint,
                image_size=(64, 48),
                epochs=1,
                batch_size=4,
                channels=4,
                device_name="cpu",
            )
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(training["train_images"], len(training_images))
            detector = LearnedBallDetector.load(
                checkpoint, device_name="cpu", confidence_threshold=0.0
            )
            with Image.open(root / first["images"]["camera1"]) as image:
                detection = detector(np.asarray(image.convert("RGB")))
            self.assertIsNotNone(detection)


if __name__ == "__main__":
    unittest.main()
