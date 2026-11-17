"""Analytic fixtures for software tests only; never used as research data/results."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from sanet.config import from_dict
from sanet.io import write_jsonl


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.fixture
def config(tmp_path):
    return from_dict(
        {
            "model": {
                "backbone": "compact",
                "channels": 8,
                "embedding_dim": 2,
                "depth_bins": 5,
                "depth_min": 2,
                "depth_max": 14,
                "top_k": 2,
                "min_pixels": 2,
                "max_pixels": 20,
                "grid": {"x_min": 0, "x_max": 16, "y_min": -8, "y_max": 8, "resolution": 1},
            },
            "data": {
                "root": str(tmp_path),
                "manifest": str(tmp_path / "manifest.jsonl"),
                "height": 32,
                "width": 64,
                "line_width": 3,
            },
            "train": {"epochs": 2, "batch_size": 2, "threads": 1, "warmup_epochs": 1},
            "decode": {"min_points": 2, "row_step": 1},
        }
    )


@pytest.fixture
def records(config):
    root = Path(config.data.root)
    result = []
    transform = [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]]
    intrinsics = [[40, 0, 31.5], [0, 40, 15.5], [0, 0, 1]]
    for i, split in enumerate(["train", "train", "val", "test"]):
        im = Image.new("RGB", (64, 32), (i * 20, 30, 40))
        draw = ImageDraw.Draw(im)
        draw.line([(20, 16), (8, 31)], fill="white", width=2)
        draw.line([(44, 16), (56, 31)], fill="white", width=2)
        name = f"fixture-{i}.png"
        im.save(root / name)
        cloud = np.array([[3, -2, 0, 0.5], [4, -2, 0, 0.7], [6, 2, 0, 0.4]], dtype=np.float32)
        cloud[:, 3] += i * 0.01
        np.save(root / f"points-{i}.npy", cloud)
        result.append(
            {
                "id": str(i),
                "image": name,
                "group": f"group-{split}",
                "split": split,
                "lanes_2d": [[[20, 16], [8, 31]], [[44, 16], [56, 31]]],
                "lanes_3d": [[[2, -2, 0], [14, -2, 0]], [[2, 2, 0], [14, 2, 0]]],
                "intrinsics": intrinsics,
                "T_ego_from_camera": transform,
                "points": f"points-{i}.npy",
            }
        )
    write_jsonl(config.data.manifest, result)
    return result
