"""Small heatmap detector for randomized tennis-ball images."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .perception import BallDetection


def project_world_point(
    calibration: dict[str, Any], point_world_m: np.ndarray
) -> np.ndarray:
    """Project a world point using the dataset's MuJoCo camera convention."""
    origin = np.asarray(calibration["camera_origin_world_m"], dtype=np.float64)
    rotation = np.asarray(
        calibration["camera_to_world_rotation"], dtype=np.float64
    ).reshape(3, 3)
    point_camera = rotation.T @ (
        np.asarray(point_world_m, dtype=np.float64).reshape(3) - origin
    )
    if point_camera[2] >= -1e-9:
        raise ValueError("point is behind the camera")
    focal_x, focal_y = np.asarray(calibration["focal_length_px"], dtype=np.float64)
    center_x, center_y = np.asarray(
        calibration["principal_point_px"], dtype=np.float64
    )
    return np.array(
        [
            center_x + focal_x * point_camera[0] / -point_camera[2],
            center_y - focal_y * point_camera[1] / -point_camera[2],
        ],
        dtype=np.float64,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _gaussian_heatmap(
    pixel_xy: np.ndarray,
    image_size: tuple[int, int],
    output_size: tuple[int, int],
    sigma: float = 1.25,
) -> np.ndarray:
    image_width, image_height = image_size
    output_width, output_height = output_size
    center_x = float(pixel_xy[0]) * output_width / image_width
    center_y = float(pixel_xy[1]) * output_height / image_height
    rows, columns = np.mgrid[:output_height, :output_width]
    heatmap = np.exp(
        -((columns - center_x) ** 2 + (rows - center_y) ** 2) / (2.0 * sigma**2)
    ).astype(np.float32)
    peak_x = int(np.clip(round(center_x), 0, output_width - 1))
    peak_y = int(np.clip(round(center_y), 0, output_height - 1))
    heatmap[peak_y, peak_x] = 1.0
    return heatmap


class FlightBallImageDataset(Dataset[dict[str, Any]]):
    """Flatten stereo frames into supervised camera images and heatmaps."""

    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        image_size: tuple[int, int] = (256, 192),
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        episodes = _read_jsonl(self.dataset_dir / "episodes.jsonl")
        self.episode_by_id = {
            episode["episode_id"]: episode
            for episode in episodes
            if episode["split"] == split
        }
        frames = _read_jsonl(self.dataset_dir / "frames.jsonl")
        self.samples: list[dict[str, Any]] = []
        for frame in frames:
            if frame["split"] != split:
                continue
            episode = self.episode_by_id[frame["episode_id"]]
            for camera_name in ("camera1", "camera2"):
                calibration = episode["camera_calibration"][camera_name]
                pixel_xy = project_world_point(
                    calibration, np.asarray(frame["ball"]["position_world_m"])
                )
                source_width, source_height = calibration["image_size_px"]
                scaled_pixel = pixel_xy * np.array(
                    [image_size[0] / source_width, image_size[1] / source_height]
                )
                if not (
                    0.0 <= scaled_pixel[0] < image_size[0]
                    and 0.0 <= scaled_pixel[1] < image_size[1]
                ):
                    continue
                self.samples.append(
                    {
                        "image_path": frame["images"][camera_name],
                        "pixel_xy": scaled_pixel,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with Image.open(self.dataset_dir / sample["image_path"]) as source:
            image = source.convert("RGB")
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32).copy()
        tensor = torch.from_numpy(pixels).permute(2, 0, 1) / 255.0
        output_size = self.image_size
        heatmap = _gaussian_heatmap(
            sample["pixel_xy"], self.image_size, output_size
        )
        return {
            "image": tensor,
            "heatmap": torch.from_numpy(heatmap).unsqueeze(0),
            "pixel_xy": torch.tensor(sample["pixel_xy"], dtype=torch.float32),
        }


class BallHeatmapDetector(nn.Module):
    """Classify every pixel with local context and no spatial downsampling."""

    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        self.channels = channels
        self.network = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=5,
                padding=2,
                groups=channels,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def focal_heatmap_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """CenterNet-style focal loss for one ball center per image."""
    probabilities = logits.sigmoid().clamp(1e-5, 1.0 - 1e-5)
    positive = targets.eq(1.0)
    negative = targets.lt(1.0)
    negative_weights = (1.0 - targets).pow(4)
    positive_penalty = -(
        torch.log(probabilities) * (1.0 - probabilities).pow(2) * positive
    )
    negative_penalty = -(
        torch.log(1.0 - probabilities)
        * probabilities.pow(2)
        * negative_weights
        * negative
    )
    positive_loss = positive_penalty.flatten(1).sum(dim=1).mean()
    flattened_negative = negative_penalty.flatten(1)
    hard_negative_count = min(256, flattened_negative.shape[1])
    hard_negative_loss = flattened_negative.topk(
        hard_negative_count, dim=1
    ).values.mean()
    # Hard-negative mining prevents rare bright court or robot pixels from
    # disappearing inside the average of roughly 77,000 background pixels.
    return positive_loss + hard_negative_loss


def decode_heatmaps(
    logits: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode peak heatmap cells into input-image pixel coordinates."""
    probabilities = logits.sigmoid()
    batch, _, output_height, output_width = probabilities.shape
    flattened = probabilities.reshape(batch, -1)
    confidence, flat_index = flattened.max(dim=1)
    row = torch.div(flat_index, output_width, rounding_mode="floor")
    column = flat_index % output_width
    scale_x = image_size[0] / output_width
    scale_y = image_size[1] / output_height
    coordinates = torch.stack(
        (column.float() * scale_x, row.float() * scale_y), dim=1
    )
    return coordinates, confidence


def _select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    return device


def _pixel_rmse(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: tuple[int, int],
) -> float:
    squared_errors: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            predictions, _ = decode_heatmaps(
                model(batch["image"].to(device)), image_size
            )
            targets = batch["pixel_xy"].to(device)
            squared_errors.append(((predictions - targets) ** 2).sum(dim=1).cpu())
    if not squared_errors:
        raise ValueError("validation split has no projected ball images")
    return float(torch.cat(squared_errors).mean().sqrt())


def _source_state() -> dict[str, Any]:
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


def train_ball_heatmap_detector(
    dataset_dir: Path,
    checkpoint_path: Path,
    *,
    report_path: Path | None = None,
    image_size: tuple[int, int] = (256, 192),
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    channels: int = 24,
    device_name: str = "auto",
    seed: int = 2026,
) -> dict[str, Any]:
    """Train on the train split, select by validation pixel RMSE, and save."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _select_device(device_name)
    train_dataset = FlightBallImageDataset(dataset_dir, "train", image_size)
    validation_dataset = FlightBallImageDataset(
        dataset_dir, "validation", image_size
    )
    if not train_dataset or not validation_dataset:
        raise ValueError("train and validation splits must both contain images")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = BallHeatmapDetector(channels=channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float | int]] = []
    best_rmse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"].to(device))
            loss = focal_heatmap_loss(logits, batch["heatmap"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        validation_rmse = _pixel_rmse(
            model, validation_loader, device, image_size
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / batches,
                "validation_pixel_rmse": validation_rmse,
            }
        )
        if validation_rmse < best_rmse:
            best_rmse = validation_rmse
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")

    dataset_info = json.loads(
        (Path(dataset_dir) / "dataset_info.json").read_text(encoding="utf-8")
    )
    checkpoint = {
        "schema_version": 1,
        "model": "BallHeatmapDetector",
        "channels": channels,
        "image_size": list(image_size),
        "state_dict": best_state,
        "training": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": str(device),
            "source": _source_state(),
            "dataset_source": dataset_info["source"],
        },
    }
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    report = {
        "schema_version": 1,
        "model": "BallHeatmapDetector",
        "checkpoint": str(checkpoint_path),
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "best_validation_pixel_rmse": best_rmse,
        "history": history,
        "training": checkpoint["training"],
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


@dataclass
class LearnedBallDetector:
    model: BallHeatmapDetector
    image_size: tuple[int, int]
    device: torch.device
    confidence_threshold: float

    @classmethod
    def load(
        cls,
        checkpoint_path: Path,
        *,
        device_name: str = "auto",
        confidence_threshold: float = 0.20,
    ) -> "LearnedBallDetector":
        device = _select_device(device_name)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        model = BallHeatmapDetector(channels=int(checkpoint["channels"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        return cls(
            model=model,
            image_size=tuple(checkpoint["image_size"]),
            device=device,
            confidence_threshold=confidence_threshold,
        )

    def __call__(self, image: np.ndarray) -> BallDetection | None:
        source_height, source_width = image.shape[:2]
        resized = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize(
            self.image_size, Image.Resampling.BILINEAR
        )
        pixels = np.asarray(resized, dtype=np.float32).copy()
        tensor = (
            torch.from_numpy(pixels)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
            / 255.0
        )
        with torch.inference_mode():
            coordinates, confidence = decode_heatmaps(
                self.model(tensor), self.image_size
            )
        score = float(confidence[0].cpu())
        if score < self.confidence_threshold:
            return None
        pixel = coordinates[0].cpu().numpy()
        pixel *= np.array(
            [source_width / self.image_size[0], source_height / self.image_size[1]]
        )
        return BallDetection(pixel_xy=pixel, pixel_count=1, confidence=score)
