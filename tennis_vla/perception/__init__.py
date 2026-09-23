"""Ball detection, randomized dataset generation, and evaluation."""

from __future__ import annotations

from .detection import (
    BallDetection,
    BallTrackEstimate,
    TriangulatedBall,
    camera_calibration,
    camera_ray,
    camera_ray_from_calibration,
    detect_yellow_ball,
    fit_constant_velocity_track,
    predict_x_crossing_time,
    triangulate_ball_from_images,
    triangulate_rays,
)
from .domain_randomization import (
    CameraDomain,
    RenderDomain,
    apply_render_domain,
    look_at_quaternion,
    postprocess_render,
    sample_render_domain,
)
from .learned import (
    BallHeatmapDetector,
    FlightBallImageDataset,
    LearnedBallDetector,
    decode_heatmaps,
    focal_heatmap_loss,
    project_world_point,
    train_ball_heatmap_detector,
)
from .evaluation import (
    evaluate_color_stereo_baseline,
    evaluate_stereo_detector,
)
from .flight_dataset import (
    DATASET_NAME,
    DATASET_SCHEMA_VERSION,
    SPLIT_SEED_BASES,
    generate_flight_dataset,
    validate_flight_dataset,
    write_summary,
)

__all__ = [
    "BallDetection",
    "BallHeatmapDetector",
    "BallTrackEstimate",
    "CameraDomain",
    "DATASET_NAME",
    "DATASET_SCHEMA_VERSION",
    "FlightBallImageDataset",
    "LearnedBallDetector",
    "RenderDomain",
    "SPLIT_SEED_BASES",
    "TriangulatedBall",
    "apply_render_domain",
    "camera_calibration",
    "camera_ray",
    "camera_ray_from_calibration",
    "decode_heatmaps",
    "detect_yellow_ball",
    "evaluate_color_stereo_baseline",
    "evaluate_stereo_detector",
    "fit_constant_velocity_track",
    "focal_heatmap_loss",
    "generate_flight_dataset",
    "look_at_quaternion",
    "postprocess_render",
    "predict_x_crossing_time",
    "project_world_point",
    "sample_render_domain",
    "train_ball_heatmap_detector",
    "triangulate_ball_from_images",
    "triangulate_rays",
    "validate_flight_dataset",
    "write_summary",
]
