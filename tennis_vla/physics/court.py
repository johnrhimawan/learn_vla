"""Regulation court geometry and deterministic line-call helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TennisCourtSpec:
    length_m: float = 23.77
    singles_width_m: float = 8.23
    doubles_width_m: float = 10.97
    service_line_from_net_m: float = 6.40
    net_center_height_m: float = 0.914
    net_post_height_m: float = 1.07

    @property
    def half_length_m(self) -> float:
        return self.length_m / 2.0

    @property
    def singles_half_width_m(self) -> float:
        return self.singles_width_m / 2.0

    @property
    def doubles_half_width_m(self) -> float:
        return self.doubles_width_m / 2.0

    def net_height_m(self, lateral_m: float) -> float:
        """Approximate the net-cord sag linearly between center and posts."""
        fraction = min(abs(lateral_m) / self.doubles_half_width_m, 1.0)
        return self.net_center_height_m + fraction * (
            self.net_post_height_m - self.net_center_height_m
        )

    def contains_singles_bounce(
        self, point_m: np.ndarray, ball_radius_m: float = 0.0
    ) -> bool:
        """Treat a ball that overlaps a line as in, matching tennis line calls."""
        point = np.asarray(point_m, dtype=np.float64).reshape(3)
        return bool(
            abs(point[0]) <= self.half_length_m + ball_radius_m
            and abs(point[1]) <= self.singles_half_width_m + ball_radius_m
        )


def court_scene_xml(ball_radius_m: float = 0.0335) -> str:
    """Return a lightweight regulation-court MJCF for visualization."""
    court = TennisCourtSpec()
    half_length = court.half_length_m
    singles_half_width = court.singles_half_width_m
    doubles_half_width = court.doubles_half_width_m
    service = court.service_line_from_net_m
    return f"""
<mujoco model="tennis_court_v0">
  <visual>
    <global offwidth="960" offheight="540"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.75 0.75 0.75"/>
  </visual>
  <worldbody>
    <geom name="court" type="plane" size="15 8 0.05" rgba="0.12 0.38 0.25 1"/>
    <geom name="net" type="box" pos="0 0 {court.net_center_height_m / 2:.6f}"
          size="0.025 {doubles_half_width:.6f} {court.net_center_height_m / 2:.6f}"
          rgba="0.92 0.92 0.92 0.72" contype="0" conaffinity="0"/>
    <geom name="near_baseline" type="box" pos="{-half_length:.6f} 0 0.006"
          size="0.025 {doubles_half_width:.6f} 0.006" rgba="1 1 1 1"/>
    <geom name="far_baseline" type="box" pos="{half_length:.6f} 0 0.006"
          size="0.025 {doubles_half_width:.6f} 0.006" rgba="1 1 1 1"/>
    <geom name="left_doubles_sideline" type="box" pos="0 {-doubles_half_width:.6f} 0.006"
          size="{half_length:.6f} 0.025 0.006" rgba="1 1 1 1"/>
    <geom name="right_doubles_sideline" type="box" pos="0 {doubles_half_width:.6f} 0.006"
          size="{half_length:.6f} 0.025 0.006" rgba="1 1 1 1"/>
    <geom name="left_singles_sideline" type="box" pos="0 {-singles_half_width:.6f} 0.007"
          size="{half_length:.6f} 0.018 0.007" rgba="1 1 1 1"/>
    <geom name="right_singles_sideline" type="box" pos="0 {singles_half_width:.6f} 0.007"
          size="{half_length:.6f} 0.018 0.007" rgba="1 1 1 1"/>
    <geom name="near_service_line" type="box" pos="{-service:.6f} 0 0.007"
          size="0.018 {singles_half_width:.6f} 0.007" rgba="1 1 1 1"/>
    <geom name="far_service_line" type="box" pos="{service:.6f} 0 0.007"
          size="0.018 {singles_half_width:.6f} 0.007" rgba="1 1 1 1"/>
    <geom name="center_service_line" type="box" pos="0 0 0.007"
          size="{service:.6f} 0.018 0.007" rgba="1 1 1 1"/>
    <camera name="court_camera" pos="-15 -14 9"
            xyaxes="0.683 -0.731 0 0.292 0.273 0.917"/>
    <body name="ball" mocap="true" pos="10.5 0 1.4">
      <geom type="sphere" size="{ball_radius_m:.6f}" rgba="0.82 0.95 0.12 1"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""
