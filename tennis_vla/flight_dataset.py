"""Generate and validate the randomized ``tennis-flight-v0`` dataset."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .arm import home_configuration
from .ballistics import BallFlightConfig, BallFlightResult
from .domain_randomization import (
    RenderDomain,
    apply_render_domain,
    postprocess_render,
    sample_render_domain,
)
from .environment import make_tennis_contact_model
from .feeder import ProgrammableFeeder
from .perception import camera_calibration


DATASET_NAME = "tennis-flight-v0"
DATASET_SCHEMA_VERSION = 1
SPLIT_SEED_BASES = {"train": 1000, "validation": 6000, "test": 7000}


def _interpolate_vector(
    times_s: np.ndarray,
    vectors: np.ndarray,
    query_time_s: float,
) -> np.ndarray:
    return np.array(
        [np.interp(query_time_s, times_s, vectors[:, axis]) for axis in range(3)],
        dtype=np.float64,
    )


def _first_crossing_time(
    times_s: np.ndarray,
    x_positions_m: np.ndarray,
    plane_x_m: float,
) -> float:
    indices = np.flatnonzero(x_positions_m <= plane_x_m)
    if not len(indices) or indices[0] == 0:
        raise ValueError(f"trajectory does not cross x={plane_x_m}")
    index = int(indices[0])
    x0, x1 = x_positions_m[index - 1 : index + 1]
    fraction = (plane_x_m - x0) / (x1 - x0)
    return float(times_s[index - 1] + fraction * (times_s[index] - times_s[index - 1]))


def _first_bounce_time(flight: BallFlightResult, radius_m: float) -> float:
    candidates = np.flatnonzero(
        (flight.positions_m[:, 2] <= radius_m + 1e-9)
        & (flight.velocities_m_s[:, 2] > 0.0)
    )
    if not len(candidates):
        raise ValueError("trajectory has no bounce")
    return float(flight.times_s[int(candidates[0])])


def _repository_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"git_revision": revision, "tracked_files_dirty": dirty}


def _noise_seed(domain_seed: int, frame_index: int, camera_index: int) -> int:
    sequence = np.random.SeedSequence([domain_seed, frame_index, camera_index])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _render_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    flight: BallFlightResult,
    timestamp_s: float,
    ball_qpos_address: int,
    domain: RenderDomain,
    frame_index: int,
) -> list[np.ndarray]:
    exposure_offsets = np.linspace(-0.5, 0.5, 3) * domain.exposure_s
    accumulated: list[list[np.ndarray]] = [[], []]
    for offset in exposure_offsets:
        sample_time = float(
            np.clip(timestamp_s + offset, flight.times_s[0], flight.times_s[-1])
        )
        position = _interpolate_vector(
            flight.times_s, flight.positions_m, sample_time
        )
        data.qpos[ball_qpos_address : ball_qpos_address + 3] = position
        data.qpos[ball_qpos_address + 3 : ball_qpos_address + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        mujoco.mj_forward(model, data)
        for camera_index, camera_name in enumerate(("camera1", "camera2")):
            renderer.update_scene(data, camera=camera_name)
            accumulated[camera_index].append(
                renderer.render().astype(np.float64)
            )

    rendered = []
    for camera_index, subframes in enumerate(accumulated):
        blurred = np.rint(np.mean(subframes, axis=0)).astype(np.uint8)
        rendered.append(
            postprocess_render(
                blurred,
                domain,
                _noise_seed(domain.seed, frame_index, camera_index),
            )
        )
    return rendered


def _write_preview(
    output_dir: Path,
    frame_records: list[dict[str, Any]],
    preview_path: Path,
) -> None:
    if not frame_records:
        return
    indices = sorted({0, len(frame_records) // 2, len(frame_records) - 1})
    selected = [frame_records[index] for index in indices]
    first = Image.open(output_dir / selected[0]["images"]["camera1"]).convert("RGB")
    tile_width, tile_height = first.size
    title_height = 22
    canvas = Image.new(
        "RGB", (tile_width * len(selected), (tile_height + title_height) * 2), "black"
    )
    draw = ImageDraw.Draw(canvas)
    for column, record in enumerate(selected):
        for row, camera_name in enumerate(("camera1", "camera2")):
            image = Image.open(output_dir / record["images"][camera_name]).convert(
                "RGB"
            )
            x = column * tile_width
            y = row * (tile_height + title_height)
            canvas.paste(image, (x, y + title_height))
            draw.text(
                (x + 5, y + 4),
                f"{camera_name}  t={record['timestamp_s']:.3f}s",
                fill="white",
            )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_path)


def generate_flight_dataset(
    output_dir: Path,
    split_counts: dict[str, int],
    *,
    width: int = 512,
    height: int = 384,
    observation_hz: float = 50.0,
    start_plane_x_m: float = 2.0,
    strike_plane_x_m: float = -9.25,
    overwrite: bool = False,
    preview_path: Path | None = None,
) -> dict[str, Any]:
    """Render randomized stereo episodes and exact privileged labels."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"dataset already exists: {output_dir}")
        shutil.rmtree(output_dir)
    unknown_splits = set(split_counts) - set(SPLIT_SEED_BASES)
    if unknown_splits:
        raise ValueError(f"unknown splits: {sorted(unknown_splits)}")
    if any(count < 0 or count > 1000 for count in split_counts.values()):
        raise ValueError("split counts must be between zero and 1000")
    if width < 32 or height < 32 or observation_hz <= 0.0:
        raise ValueError("image dimensions and observation_hz must be positive")

    output_dir.mkdir(parents=True)
    model = make_tennis_contact_model()
    data = mujoco.MjData(model)
    data.qpos[:7] = home_configuration(model)
    data.ctrl[:] = data.qpos[:7]
    ball_qpos_address = int(model.joint("ball_free").qposadr[0])
    renderer = mujoco.Renderer(model, width=width, height=height)
    feeder = ProgrammableFeeder()
    flight_config = BallFlightConfig()
    episode_records: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    split_summary: dict[str, dict[str, int]] = {}
    global_episode_index = 0

    try:
        for split in ("train", "validation", "test"):
            episode_count = int(split_counts.get(split, 0))
            split_frame_count = 0
            for split_episode_index in range(episode_count):
                feed_seed = SPLIT_SEED_BASES[split] + split_episode_index
                domain_seed = 1_000_000 + feed_seed
                feed, flight = feeder.sample_legal_feed(feed_seed)
                domain = sample_render_domain(domain_seed)
                apply_render_domain(model, data, domain)
                calibrations = {
                    camera_name: camera_calibration(
                        model, data, camera_name, (height, width)
                    )
                    for camera_name in ("camera1", "camera2")
                }
                start_time = _first_crossing_time(
                    flight.times_s, flight.positions_m[:, 0], start_plane_x_m
                )
                contact_time = _first_crossing_time(
                    flight.times_s, flight.positions_m[:, 0], strike_plane_x_m
                )
                bounce_time = _first_bounce_time(flight, flight_config.radius_m)
                first_frame_time = np.ceil(start_time * observation_hz) / observation_hz
                timestamps = np.arange(
                    first_frame_time,
                    contact_time + 1e-12,
                    1.0 / observation_hz,
                )
                episode_id = f"episode_{global_episode_index:06d}"
                episode_dir = Path("images") / split / episode_id
                for camera_name in ("camera1", "camera2"):
                    (output_dir / episode_dir / camera_name).mkdir(
                        parents=True, exist_ok=True
                    )
                episode_frame_records: list[dict[str, Any]] = []
                for frame_index, timestamp in enumerate(timestamps):
                    timestamp = float(timestamp)
                    position = _interpolate_vector(
                        flight.times_s, flight.positions_m, timestamp
                    )
                    velocity = _interpolate_vector(
                        flight.times_s, flight.velocities_m_s, timestamp
                    )
                    images = _render_observation(
                        model,
                        data,
                        renderer,
                        flight,
                        timestamp,
                        ball_qpos_address,
                        domain,
                        frame_index,
                    )
                    image_paths: dict[str, str] = {}
                    for camera_index, camera_name in enumerate(("camera1", "camera2")):
                        relative_path = (
                            episode_dir
                            / camera_name
                            / f"frame_{frame_index:05d}.png"
                        )
                        Image.fromarray(images[camera_index]).save(
                            output_dir / relative_path
                        )
                        image_paths[camera_name] = relative_path.as_posix()

                    record = {
                        "schema_version": DATASET_SCHEMA_VERSION,
                        "split": split,
                        "episode_id": episode_id,
                        "frame_index": frame_index,
                        "timestamp_s": timestamp,
                        "images": image_paths,
                        "ball": {
                            "position_world_m": position.tolist(),
                            "velocity_world_m_s": velocity.tolist(),
                            "spin_class": "none",
                            "bounce_occurred": timestamp >= bounce_time,
                        },
                        "prediction_targets": {
                            "strike_plane_x_m": strike_plane_x_m,
                            "time_to_strike_plane_s": contact_time - timestamp,
                        },
                    }
                    frame_records.append(record)
                    episode_frame_records.append(record)

                episode_records.append(
                    {
                        "schema_version": DATASET_SCHEMA_VERSION,
                        "split": split,
                        "episode_id": episode_id,
                        "feed": {
                            "seed": feed.seed,
                            "rejection_sample_attempt": feed.attempt,
                            "initial_position_m": feed.position_m.tolist(),
                            "initial_velocity_m_s": feed.velocity_m_s.tolist(),
                            "first_bounce_m": flight.first_bounce_m.tolist(),
                            "bounce_time_s": bounce_time,
                            "strike_plane_time_s": contact_time,
                        },
                        "domain": domain.to_dict(),
                        "camera_calibration": calibrations,
                        "frames": len(episode_frame_records),
                    }
                )
                if global_episode_index == 0 and preview_path is not None:
                    _write_preview(output_dir, episode_frame_records, preview_path)
                split_frame_count += len(episode_frame_records)
                global_episode_index += 1
            split_summary[split] = {
                "episodes": episode_count,
                "frames": split_frame_count,
            }
    finally:
        renderer.close()

    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for record in episode_records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    with (output_dir / "frames.jsonl").open("w", encoding="utf-8") as handle:
        for record in frame_records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    info = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "source": _repository_state(),
        "coordinate_frame": "x along court toward far side; y lateral; z up",
        "image_format": "RGB PNG",
        "splits": split_summary,
        "episodes": len(episode_records),
        "frames": len(frame_records),
        "cameras_per_frame": 2,
        "observation_hz": observation_hz,
        "image_size_px": [width, height],
        "start_plane_x_m": start_plane_x_m,
        "strike_plane_x_m": strike_plane_x_m,
        "privileged_fields": [
            "ball.position_world_m",
            "ball.velocity_world_m_s",
            "ball.spin_class",
            "ball.bounce_occurred",
            "prediction_targets.time_to_strike_plane_s",
        ],
        "limitations": [
            "The current ball model has no spin.",
            "Court appearance varies by color and lighting but has no image texture maps.",
            "Motion blur is approximated by averaging three exposure-time renders.",
        ],
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    validation = validate_flight_dataset(output_dir)
    return {**info, "validation": validation}


def validate_flight_dataset(output_dir: Path) -> dict[str, Any]:
    """Check manifest counts, split isolation, labels, and referenced images."""
    output_dir = Path(output_dir)
    info = json.loads((output_dir / "dataset_info.json").read_text(encoding="utf-8"))
    episodes = [
        json.loads(line)
        for line in (output_dir / "episodes.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    frames = [
        json.loads(line)
        for line in (output_dir / "frames.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    if len(episodes) != info["episodes"] or len(frames) != info["frames"]:
        raise ValueError("manifest counts do not match dataset_info.json")

    split_seeds: dict[str, set[int]] = {name: set() for name in SPLIT_SEED_BASES}
    for episode in episodes:
        split_seeds[episode["split"]].add(int(episode["feed"]["seed"]))
    for first_name, first_seeds in split_seeds.items():
        for second_name, second_seeds in split_seeds.items():
            if first_name < second_name and first_seeds & second_seeds:
                raise ValueError("feed seed appears in more than one split")

    missing_images = 0
    for frame in frames:
        if len(frame["ball"]["position_world_m"]) != 3:
            raise ValueError("invalid position label")
        if len(frame["ball"]["velocity_world_m_s"]) != 3:
            raise ValueError("invalid velocity label")
        for relative_path in frame["images"].values():
            if not (output_dir / relative_path).is_file():
                missing_images += 1
    if missing_images:
        raise FileNotFoundError(f"dataset references {missing_images} missing images")
    return {
        "manifest_counts_match": True,
        "split_seeds_disjoint": True,
        "referenced_images": len(frames) * 2,
        "missing_images": 0,
    }


def write_summary(report: dict[str, Any], path: Path) -> None:
    summary = {
        key: report[key]
        for key in (
            "schema_version",
            "dataset",
            "source",
            "splits",
            "episodes",
            "frames",
            "cameras_per_frame",
            "observation_hz",
            "image_size_px",
            "validation",
            "limitations",
        )
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
