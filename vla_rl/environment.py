"""A deterministic language-conditioned reaching task with recovery events.

The policy receives one RGB image, joint position/velocity, and a language
instruction. Target and obstacle coordinates stay private to the simulator and
are used only by the reward, metrics, and demonstration controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


MODEL_XML = r"""
<mujoco model="language_obstacle_reach_v0">
  <compiler angle="radian"/>
  <option timestep="0.005" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="640" offheight="640"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.75 0.75 0.75"/>
  </visual>
  <default>
    <joint damping="1.5" armature="0.04" limited="true"/>
    <geom type="capsule" size="0.032" rgba="0.18 0.48 0.88 1"
          contype="0" conaffinity="0"/>
    <position kp="80" kv="12" ctrllimited="true"/>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.16 0.18 0.20"
             rgb2="0.28 0.30 0.32" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="4 4" reflectance="0.08"/>
  </asset>
  <worldbody>
    <light pos="0 -0.5 1.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="floor"
          contype="0" conaffinity="0"/>
    <camera name="camera1" pos="1.00 -1.05 0.82"
            xyaxes="0.724 0.690 0 -0.355 0.373 0.857"/>
    <body name="base" pos="0 0 0.10">
      <geom name="base_geom" type="cylinder" size="0.10 0.10"
            rgba="0.10 0.12 0.16 1" contype="0" conaffinity="0"/>
      <body name="shoulder" pos="0 0 0.12">
        <joint name="shoulder_yaw" axis="0 0 1" range="-2.6 2.6"/>
        <joint name="shoulder_pitch" axis="0 1 0" range="-1.7 1.7"/>
        <geom name="upper_arm" fromto="0 0 0 0.32 0 0"/>
        <body name="elbow" pos="0.32 0 0">
          <joint name="elbow_pitch" axis="0 1 0" range="-2.5 2.5"/>
          <geom name="forearm" fromto="0 0 0 0.27 0 0"
                rgba="0.96 0.47 0.14 1"/>
          <geom name="tool_collision" type="sphere" pos="0.27 0 0" size="0.022"
                rgba="0.98 0.86 0.12 1" contype="1" conaffinity="1"/>
          <site name="end_effector" pos="0.27 0 0" size="0.050"
                rgba="0.98 0.86 0.12 1"/>
        </body>
      </body>
    </body>
    <body name="red_target" mocap="true" pos="0.40 -0.18 0.32">
      <geom name="red_target_geom" type="sphere" size="0.045"
            rgba="0.95 0.12 0.10 0.82" contype="0" conaffinity="0"/>
    </body>
    <body name="green_target" mocap="true" pos="0.40 0.18 0.32">
      <geom name="green_target_geom" type="sphere" size="0.045"
            rgba="0.10 0.92 0.28 0.82" contype="0" conaffinity="0"/>
    </body>
    <body name="obstacle" mocap="true" pos="0.38 0 0.32">
      <geom name="obstacle_geom" type="sphere" size="0.058"
            rgba="0.08 0.28 0.98 0.90" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="yaw_position" joint="shoulder_yaw" ctrlrange="-2.6 2.6"/>
    <position name="shoulder_position" joint="shoulder_pitch" ctrlrange="-1.7 1.7"/>
    <position name="elbow_position" joint="elbow_pitch" ctrlrange="-2.5 2.5"/>
  </actuator>
</mujoco>
"""


STATE_NAMES = [
    "shoulder_yaw.pos",
    "shoulder_pitch.pos",
    "elbow_pitch.pos",
    "shoulder_yaw.vel",
    "shoulder_pitch.vel",
    "elbow_pitch.vel",
]
ACTION_NAMES = [
    "shoulder_yaw.target",
    "shoulder_pitch.target",
    "elbow_pitch.target",
]


@dataclass(frozen=True)
class TaskSpec:
    key: str
    target: str
    instruction: str
    max_joint_delta: float
    smoothness_weight: float


TASKS = {
    "red_fast": TaskSpec(
        key="red_fast",
        target="red",
        instruction="Reach the red target quickly without touching the blue obstacle.",
        max_joint_delta=0.12,
        smoothness_weight=0.015,
    ),
    "green_smooth": TaskSpec(
        key="green_smooth",
        target="green",
        instruction="Reach the green target smoothly without touching the blue obstacle.",
        max_joint_delta=0.10,
        smoothness_weight=0.080,
    ),
}


class ObstacleReachEnv:
    """Minimal Gymnasium-style environment without a Gym dependency."""

    control_hz = 20
    physics_steps_per_control = 10
    success_distance = 0.045
    success_hold_steps = 3

    def __init__(self, image_size: int = 256, max_steps: int = 80) -> None:
        self.image_size = image_size
        self.max_steps = max_steps
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model, height=image_size, width=image_size
        )
        self.ee_site_id = self.model.site("end_effector").id
        self.obstacle_geom_id = self.model.geom("obstacle_geom").id
        self.mocap_ids = {
            name: self.model.body(name).mocapid[0]
            for name in ("red_target", "green_target", "obstacle")
        }
        self.jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        self.rng = np.random.default_rng(0)
        self.task = TASKS["red_fast"]
        self.step_count = 0
        self.success_streak = 0
        self.collision_count = 0
        self.disturbance_step: int | None = None
        self.disturbance_applied = False
        self.previous_distance = 0.0
        self.previous_action = np.zeros(3, dtype=np.float32)

    def close(self) -> None:
        self.renderer.close()

    @property
    def end_effector_position(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    @property
    def target_position(self) -> np.ndarray:
        return self.data.mocap_pos[self.mocap_ids[f"{self.task.target}_target"]].copy()

    @property
    def obstacle_position(self) -> np.ndarray:
        return self.data.mocap_pos[self.mocap_ids["obstacle"]].copy()

    @property
    def distance_to_target(self) -> float:
        return float(np.linalg.norm(self.target_position - self.end_effector_position))

    def _has_obstacle_contact(self) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.obstacle_geom_id in (contact.geom1, contact.geom2):
                return True
        return False

    def _observation(self) -> dict[str, Any]:
        self.renderer.update_scene(self.data, camera="camera1")
        image = self.renderer.render().copy()
        return {
            "observation.state": np.concatenate(
                (self.data.qpos, self.data.qvel)
            ).astype(np.float32),
            "observation.images.camera1": image,
            "task": self.task.instruction,
        }

    def _info(self) -> dict[str, Any]:
        return {
            "task_key": self.task.key,
            "distance": self.distance_to_target,
            "collision_count": self.collision_count,
            "disturbance_applied": self.disturbance_applied,
            "is_success": self.success_streak >= self.success_hold_steps,
        }

    def reset(
        self,
        seed: int = 0,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        options = options or {}
        self.rng = np.random.default_rng(seed)
        task_key = options.get("task_key", "red_fast" if seed % 2 == 0 else "green_smooth")
        if task_key not in TASKS:
            raise ValueError(f"Unknown task {task_key!r}; choose from {sorted(TASKS)}")
        self.task = TASKS[task_key]
        self.disturbance_step = options.get("disturbance_step")
        self.disturbance_applied = False
        self.step_count = 0
        self.success_streak = 0
        self.collision_count = 0

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.array([0.0, -0.55, 1.25])
        self.data.ctrl[:] = self.data.qpos

        # Both targets stay in view. The instruction selects which one matters.
        red = np.array(
            [
                self.rng.uniform(0.37, 0.49),
                self.rng.uniform(-0.38, -0.26),
                self.rng.uniform(0.22, 0.44),
            ]
        )
        green = np.array(
            [
                self.rng.uniform(0.37, 0.49),
                self.rng.uniform(0.26, 0.38),
                self.rng.uniform(0.22, 0.44),
            ]
        )
        self.data.mocap_pos[self.mocap_ids["red_target"]] = red
        self.data.mocap_pos[self.mocap_ids["green_target"]] = green
        mujoco.mj_forward(self.model, self.data)

        start = self.end_effector_position
        selected = red if self.task.target == "red" else green
        path = selected - start
        lateral = np.array([-path[1], path[0], 0.0])
        lateral /= max(float(np.linalg.norm(lateral)), 1e-6)
        midpoint = start + 0.52 * path + 0.105 * lateral
        midpoint[2] += self.rng.uniform(-0.012, 0.018)
        self.data.mocap_pos[self.mocap_ids["obstacle"]] = midpoint

        # Mild dynamics randomization belongs to the task definition and is
        # deterministic for each evaluation seed.
        self.model.dof_damping[:] = self.rng.uniform(1.25, 1.80, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self.previous_distance = self.distance_to_target
        self.previous_action = self.data.qpos.astype(np.float32).copy()
        return self._observation(), self._info()

    def project_action(self, action: np.ndarray) -> np.ndarray:
        """Apply the same position and slew limits to every controller."""
        action = np.asarray(action, dtype=np.float64).reshape(3)
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        delta = np.clip(
            action - self.data.qpos,
            -self.task.max_joint_delta,
            self.task.max_joint_delta,
        )
        lower = self.model.jnt_range[:, 0]
        upper = self.model.jnt_range[:, 1]
        return np.clip(self.data.qpos + delta, lower, upper).astype(np.float32)

    def expert_action(self) -> np.ndarray:
        """Damped least-squares controller with local obstacle repulsion."""
        ee = self.end_effector_position
        target = self.target_position
        obstacle = self.obstacle_position
        attractive = 2.8 * (target - ee)
        away = ee - self.obstacle_position
        clearance = float(np.linalg.norm(away))
        repulsive = np.zeros(3)
        influence = 0.105
        if clearance < influence:
            direction = away / max(clearance, 1e-6)
            repulsive = 0.012 * (1.0 / max(clearance, 0.060) - 1.0 / influence) * direction
            repulsive[2] += 0.08 * (influence - clearance) / influence
        cartesian_delta = attractive + repulsive

        self.jacobian.fill(0.0)
        mujoco.mj_jacSite(
            self.model, self.data, self.jacobian, None, self.ee_site_id
        )
        damping = 0.065
        task_matrix = self.jacobian @ self.jacobian.T
        joint_delta = self.jacobian.T @ np.linalg.solve(
            task_matrix + damping**2 * np.eye(3), cartesian_delta
        )
        return self.project_action(self.data.qpos + joint_delta)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = self.project_action(action)
        if self.disturbance_step is not None and self.step_count == self.disturbance_step:
            direction = -1.0 if self.task.target == "red" else 1.0
            self.data.qvel[0] += direction * 2.2
            self.data.qvel[1] -= 1.0
            self.disturbance_applied = True

        self.data.ctrl[:] = action
        collided_this_step = False
        for _ in range(self.physics_steps_per_control):
            mujoco.mj_step(self.model, self.data)
            collided_this_step = collided_this_step or self._has_obstacle_contact()

        self.step_count += 1
        if collided_this_step:
            self.collision_count += 1
        distance = self.distance_to_target
        self.success_streak = self.success_streak + 1 if distance < self.success_distance else 0
        success = self.success_streak >= self.success_hold_steps

        progress = self.previous_distance - distance
        action_change = float(np.linalg.norm(action - self.previous_action))
        reward = 12.0 * progress - 0.02
        reward -= self.task.smoothness_weight * action_change**2
        reward -= 4.0 if collided_this_step else 0.0
        reward += 20.0 if success else 0.0
        self.previous_distance = distance
        self.previous_action = action.copy()

        terminated = success
        truncated = self.step_count >= self.max_steps and not success
        return self._observation(), float(reward), terminated, truncated, self._info()


def dataset_features(image_size: int) -> dict[str, dict[str, Any]]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "observation.images.camera1": {
            "dtype": "image",
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": ACTION_NAMES,
        },
    }
