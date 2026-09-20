"""Collect a tiny MuJoCo reaching dataset in LeRobot Dataset v3 format.

The simulated robot is a 3-DoF arm.  A classical damped least-squares
Jacobian controller acts as the expert policy, leaving the example focused on
the robotics-specific data flow shared by imitation learning and VLA systems:

    observation (camera + robot state) -> policy -> action -> physics

Run from the repository root:

    scripts/run python examples/mujoco_reach.py --episodes 3
"""

from __future__ import annotations

import argparse
import importlib.metadata
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


MODEL_XML = r"""
<mujoco model="three_dof_reacher">
  <compiler angle="radian"/>
  <option timestep="0.005" integrator="implicitfast" gravity="0 0 -9.81"/>

  <visual>
    <global offwidth="640" offheight="640"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.75 0.75 0.75"/>
  </visual>

  <default>
    <joint damping="1.5" armature="0.04" limited="true"/>
    <geom type="capsule" size="0.035" rgba="0.20 0.45 0.85 1"
          contype="0" conaffinity="0"/>
    <position kp="80" kv="12" ctrllimited="true"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.20 0.22"
             rgb2="0.28 0.30 0.32" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="4 4" reflectance="0.08"/>
  </asset>

  <worldbody>
    <light pos="0 -0.5 1.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="floor"
          contype="1" conaffinity="1"/>
    <camera name="overview" pos="1.05 -1.05 0.80"
            xyaxes="0.707 0.707 0 -0.360 0.360 0.861"/>

    <body name="base" pos="0 0 0.10">
      <geom type="cylinder" size="0.10 0.10" rgba="0.12 0.14 0.18 1"/>
      <body name="shoulder" pos="0 0 0.12">
        <joint name="shoulder_yaw" axis="0 0 1" range="-2.6 2.6"/>
        <joint name="shoulder_pitch" axis="0 1 0" range="-1.7 1.7"/>
        <geom fromto="0 0 0 0.32 0 0"/>
        <body name="elbow" pos="0.32 0 0">
          <joint name="elbow_pitch" axis="0 1 0" range="-2.5 2.5"/>
          <geom fromto="0 0 0 0.27 0 0" rgba="0.95 0.45 0.16 1"/>
          <site name="end_effector" pos="0.27 0 0" size="0.055"
                rgba="0.95 0.85 0.15 1"/>
        </body>
      </body>
    </body>

    <body name="target" mocap="true" pos="0.40 0 0.34">
      <geom type="sphere" size="0.045" rgba="0.20 0.95 0.35 0.75"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <actuator>
    <position name="yaw_position" joint="shoulder_yaw" ctrlrange="-2.6 2.6"/>
    <position name="shoulder_position" joint="shoulder_pitch" ctrlrange="-1.7 1.7"/>
    <position name="elbow_position" joint="elbow_pitch" ctrlrange="-2.5 2.5"/>
  </actuator>
</mujoco>
"""

TASK = "Move the yellow end effector to the green target."
STATE_NAMES = [
    "shoulder_yaw.pos",
    "shoulder_pitch.pos",
    "elbow_pitch.pos",
    "shoulder_yaw.vel",
    "shoulder_pitch.vel",
    "elbow_pitch.vel",
    "end_effector.x",
    "end_effector.y",
    "end_effector.z",
    "target.x",
    "target.y",
    "target.z",
]
ACTION_NAMES = [
    "shoulder_yaw.target",
    "shoulder_pitch.target",
    "elbow_pitch.target",
]


@dataclass(frozen=True)
class RolloutConfig:
    steps: int
    image_size: int
    seed: int
    viewer: bool


class ReachingSimulation:
    """Small robot simulation with a Cartesian expert and LeRobot-style I/O."""

    control_hz = 20
    physics_steps_per_control = 10

    def __init__(self, config: RolloutConfig) -> None:
        self.config = config
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(config.seed)
        self.renderer = mujoco.Renderer(
            self.model, height=config.image_size, width=config.image_size
        )
        self.ee_site_id = self.model.site("end_effector").id
        self.target_mocap_id = self.model.body("target").mocapid[0]
        self.jacobian = np.zeros((3, self.model.nv), dtype=np.float64)

    def close(self) -> None:
        self.renderer.close()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.array([0.0, -0.55, 1.25])
        self.data.ctrl[:] = self.data.qpos

        radius = self.rng.uniform(0.32, 0.50)
        azimuth = self.rng.uniform(-0.65, 0.65)
        self.data.mocap_pos[self.target_mocap_id] = np.array(
            [
                radius * np.cos(azimuth),
                radius * np.sin(azimuth),
                self.rng.uniform(0.18, 0.48),
            ]
        )
        mujoco.mj_forward(self.model, self.data)

    @property
    def target_position(self) -> np.ndarray:
        return self.data.mocap_pos[self.target_mocap_id].copy()

    @property
    def end_effector_position(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    @property
    def distance_to_target(self) -> float:
        return float(np.linalg.norm(self.target_position - self.end_effector_position))

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        state = np.concatenate(
            [
                self.data.qpos,
                self.data.qvel,
                self.end_effector_position,
                self.target_position,
            ]
        ).astype(np.float32)
        self.renderer.update_scene(self.data, camera="overview")
        image = self.renderer.render().copy()
        return state, image

    def expert_action(self) -> np.ndarray:
        """Convert Cartesian error into joint targets with damped least squares."""
        error = self.target_position - self.end_effector_position
        self.jacobian.fill(0.0)
        mujoco.mj_jacSite(
            self.model, self.data, self.jacobian, None, self.ee_site_id
        )

        # Damped least squares is stable near singular arm configurations:
        # dq = J^T (J J^T + lambda^2 I)^-1 dx
        damping = 0.06
        task_matrix = self.jacobian @ self.jacobian.T
        joint_delta = self.jacobian.T @ np.linalg.solve(
            task_matrix + damping**2 * np.eye(3), 2.5 * error
        )
        joint_delta = np.clip(joint_delta, -0.10, 0.10)
        lower = self.model.jnt_range[:, 0]
        upper = self.model.jnt_range[:, 1]
        return np.clip(self.data.qpos + joint_delta, lower, upper).astype(np.float32)

    def step(self, action: np.ndarray) -> None:
        self.data.ctrl[:] = action
        for _ in range(self.physics_steps_per_control):
            mujoco.mj_step(self.model, self.data)


def dataset_features(image_size: int) -> dict[str, dict]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "observation.images.overview": {
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


def collect_dataset(
    output: Path, episodes: int, config: RolloutConfig
) -> tuple[Path, list[float]]:
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new --output directory."
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    simulation = ReachingSimulation(config)
    dataset = None
    final_distances: list[float] = []

    viewer = None
    try:
        dataset = LeRobotDataset.create(
            repo_id="local/mujoco-reach",
            root=output,
            fps=ReachingSimulation.control_hz,
            robot_type="mujoco-3dof-reacher",
            features=dataset_features(config.image_size),
            use_videos=False,
            # Images are stored as PNGs in this tutorial. Selecting the backend
            # explicitly also avoids loading an unused video decoder.
            video_backend="pyav",
        )
        if config.viewer:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(simulation.model, simulation.data)

        for episode in range(episodes):
            simulation.reset()
            for _ in range(config.steps):
                state, image = simulation.observe()
                action = simulation.expert_action()
                dataset.add_frame(
                    {
                        "observation.state": state,
                        "observation.images.overview": image,
                        "action": action,
                        "task": TASK,
                    }
                )
                simulation.step(action)
                if viewer is not None:
                    viewer.sync()
                    time.sleep(1.0 / ReachingSimulation.control_hz)

            final_distance = simulation.distance_to_target
            final_distances.append(final_distance)
            dataset.save_episode()
            print(
                f"episode={episode:02d} frames={config.steps} "
                f"final_distance={final_distance:.4f} m"
            )
    finally:
        if dataset is not None:
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            dataset.finalize()
        if viewer is not None:
            viewer.close()
        simulation.close()

    return output, final_distances


def default_output() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("artifacts") / f"mujoco-reach-{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show the live MuJoCo viewer (on macOS, launch with `scripts/run mjpython`).",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1:
        parser.error("--episodes and --steps must be positive")
    if args.image_size < 32:
        parser.error("--image-size must be at least 32")
    return args


def main() -> None:
    args = parse_args()
    output = args.output or default_output()
    config = RolloutConfig(
        steps=args.steps,
        image_size=args.image_size,
        seed=args.seed,
        viewer=args.viewer,
    )
    path, distances = collect_dataset(output, args.episodes, config)
    successes = sum(distance < 0.04 for distance in distances)
    print(f"saved={path.resolve()}")
    print(
        f"success={successes}/{len(distances)} "
        f"lerobot={importlib.metadata.version('lerobot')} "
        f"mujoco={mujoco.__version__}"
    )


if __name__ == "__main__":
    main()
