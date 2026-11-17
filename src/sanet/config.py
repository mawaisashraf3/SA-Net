"""Validated, serializable experiment configuration; no implicit dataset downloads."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GridConfig:
    x_min: float = 0.0
    x_max: float = 80.0
    y_min: float = -12.0
    y_max: float = 12.0
    resolution: float = 0.5
    z_min: float = -4.0
    z_max: float = 6.0
    voxel_height: float = 0.5

    @property
    def shape(self) -> tuple[int, int]:
        return (
            round((self.x_max - self.x_min) / self.resolution),
            round((self.y_max - self.y_min) / self.resolution),
        )

    def validate(self) -> None:
        if self.resolution <= 0 or self.voxel_height <= 0:
            raise ValueError("Voxel sizes must be positive")
        for lo, hi, step in [
            (self.x_min, self.x_max, self.resolution),
            (self.y_min, self.y_max, self.resolution),
            (self.z_min, self.z_max, self.voxel_height),
        ]:
            n = (hi - lo) / step
            if n <= 0 or abs(n - round(n)) > 1e-6:
                raise ValueError("Grid ranges must be positive integer multiples of voxel sizes")


@dataclass
class ModelConfig:
    mode: str = "image"
    backbone: str = "resnet34"
    pretrained: bool = False
    channels: int = 64
    embedding_dim: int = 4
    depth_min: float = 2.0
    depth_max: float = 80.0
    depth_bins: int = 40
    top_k: int = 3
    lane_threshold: float = 0.3
    min_pixels: int = 16
    max_pixels: int = 4096
    sfm: bool = True
    ampm: bool = True
    era: bool = True
    dilations: list[int] = field(default_factory=lambda: [1, 2, 3])
    base_kernels: int = 4
    grid: GridConfig = field(default_factory=GridConfig)


@dataclass
class DataConfig:
    manifest: str = "data/manifest.jsonl"
    root: str = "data"
    height: int = 320
    width: int = 800
    crop_top: int = 0
    line_width: int = 12  # original-image pixels, scaled when resized
    brightness: float = 0.2
    contrast: float = 0.2
    flip_probability: float = 0.5
    workers: int = 0


@dataclass
class TrainConfig:
    epochs: int = 36
    batch_size: int = 32
    learning_rate: float = 0.0006
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    seed: int = 17
    device: str = "cpu"
    amp: bool = False
    grad_clip: float = 5.0
    balanced_confidence: bool = True
    max_positive_weight: float = 20.0
    confidence: float = 2.0
    embedding: float = 0.5
    offset: float = 4.0
    height: float = 1.0
    segmentation: float = 0.5
    depth: float = 0.5
    variance_margin: float = 0.5
    distance_margin: float = 1.5
    threads: int = 2


@dataclass
class DecodeConfig:
    confidence: float = 0.5
    embedding_radius: float = 0.7
    min_points: int = 6
    row_step: int = 2
    max_lanes: int = 12
    max_candidates: int = 12000


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)

    def validate(self) -> None:
        m, d, t = self.model, self.data, self.train
        m.grid.validate()
        if m.mode not in {"image", "bev"} or m.backbone not in {"resnet34", "compact"}:
            raise ValueError("mode must be image/bev; backbone must be resnet34/compact")
        if min(m.channels, m.embedding_dim, m.depth_bins, m.base_kernels) < 1:
            raise ValueError("Model dimensions must be positive")
        if not 0 < m.depth_min < m.depth_max or not 1 <= m.top_k <= m.depth_bins:
            raise ValueError("Invalid depth bins or top_k")
        if not 0 <= m.lane_threshold <= 1 or not 0 <= m.min_pixels <= m.max_pixels:
            raise ValueError("Invalid sparse selection limits")
        if not m.dilations or min(m.dilations) < 1:
            raise ValueError("At least one positive dilation required")
        if min(d.height, d.width) < 32 or d.crop_top < 0 or d.line_width < 1:
            raise ValueError("Images must be >=32 pixels; crop and line width must be valid")
        if not 0 <= d.flip_probability <= 1 or min(d.brightness, d.contrast) < 0:
            raise ValueError("Invalid augmentation parameters")
        if min(t.epochs, t.batch_size, t.threads) < 1 or t.learning_rate <= 0:
            raise ValueError("Invalid training schedule")
        if t.warmup_epochs < 0 or min(t.weight_decay, t.grad_clip) < 0:
            raise ValueError("Invalid optimizer parameters")
        if any(
            getattr(t, k) < 0
            for k in ["confidence", "embedding", "offset", "height", "segmentation", "depth"]
        ):
            raise ValueError("Loss weights cannot be negative")
        q = self.decode
        if not 0 <= q.confidence <= 1 or q.embedding_radius <= 0:
            raise ValueError("Invalid decoder thresholds")
        if min(q.min_points, q.row_step, q.max_lanes, q.max_candidates) < 1:
            raise ValueError("Invalid decoder limits")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _make(cls: type, value: dict[str, Any]):
    allowed = {f.name for f in fields(cls)}
    if extra := set(value) - allowed:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(extra)}")
    return cls(**value)


def from_dict(raw: dict[str, Any]) -> Config:
    raw = dict(raw)
    model = dict(raw.pop("model", {}))
    grid = _make(GridConfig, model.pop("grid", {}))
    c = Config(
        model=_make(ModelConfig, {**model, "grid": grid}),
        data=_make(DataConfig, raw.pop("data", {})),
        train=_make(TrainConfig, raw.pop("train", {})),
        decode=_make(DecodeConfig, raw.pop("decode", {})),
    )
    if raw:
        raise ValueError(f"Unknown config sections: {sorted(raw)}")
    c.validate()
    return c


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    return from_dict(raw)
