"""Native annotation adapters → explicit, common JSONL records.

Adapters never infer calibration or replace unavailable LiDAR. ONCE coordinate
conversion requires an explicit camera-to-ego transform supplied by the caller.
"""

import json
from pathlib import Path

import numpy as np
import torch

from sanet.geometry import validate_calibration
from sanet.io import read_jsonl, rooted


def tusimple(labels: str | Path, split: str) -> list[dict]:
    records = []
    for item in read_jsonl(labels):
        rows = item["h_samples"]
        lanes = []
        for lane in item["lanes"]:
            if len(lane) != len(rows):
                raise ValueError("TuSimple lane/h_samples lengths differ")
            points = [[float(x), float(y)] for x, y in zip(lane, rows) if x >= 0]
            if len(points) >= 2:
                lanes.append(points)
        path = Path(item["raw_file"]).as_posix()
        records.append(
            {
                "id": path,
                "group": Path(path).parts[1],
                "split": split,
                "image": path,
                "lanes_2d": lanes,
                "source": "TuSimple",
                "h_samples": rows,
                "native_lanes": item["lanes"],
            }
        )
    return records


def culane(root: str | Path, listing: str | Path, split: str) -> list[dict]:
    records = []
    for line in Path(listing).read_text().splitlines():
        if not line.strip():
            continue
        path = line.split()[0].lstrip("/")
        annotation = rooted(root, str(Path(path).with_suffix(".lines.txt")))
        lanes = []
        for coordinates in annotation.read_text().splitlines():
            values = list(map(float, coordinates.split()))
            if len(values) % 2:
                raise ValueError(f"Odd coordinate count in {annotation}")
            if len(values) >= 4:
                lanes.append(np.asarray(values).reshape(-1, 2).tolist())
        records.append(
            {
                "id": path,
                "group": str(Path(path).parent),
                "split": split,
                "image": path,
                "lanes_2d": lanes,
                "source": "CULane",
            }
        )
    return records


def _polyline(poly: dict, samples: int = 20) -> list[list[float]]:
    vertices = np.asarray(poly["vertices"], dtype=float)
    types = poly.get("types", "L" * len(vertices))
    if len(types) != len(vertices):
        raise ValueError("BDD poly2d type/vertex count mismatch")
    result, i = [vertices[0]], 1
    while i < len(vertices):
        if types[i].upper() == "L":
            result.append(vertices[i])
            i += 1
        elif types[i : i + 3].upper() == "CCC" and i + 2 < len(vertices):
            p0, p1, p2, p3 = result[-1], vertices[i], vertices[i + 1], vertices[i + 2]
            for t in np.linspace(0, 1, samples + 1)[1:]:
                result.append(
                    (1 - t) ** 3 * p0
                    + 3 * (1 - t) ** 2 * t * p1
                    + 3 * (1 - t) * t * t * p2
                    + t**3 * p3
                )
            i += 3
        else:
            raise ValueError(f"Unsupported BDD poly2d types: {types}")
    return np.asarray(result).tolist()


def bdd100k(labels: str | Path, split: str, image_prefix: str = "") -> list[dict]:
    raw = json.loads(Path(labels).read_text())
    if not isinstance(raw, list):
        raise ValueError("Expected BDD/Scalabel frame array")
    records = []
    for item in raw:
        lanes = []
        for label in item.get("labels", []):
            category = label["category"]
            if category != "lane" and not category.startswith("lane/"):
                continue
            for poly in label.get("poly2d", []):
                if len(poly["vertices"]) >= 2:
                    lanes.append(_polyline(poly))
        name = item["name"]
        records.append(
            {
                "id": name,
                "group": item.get("videoName", Path(name).stem),
                "split": split,
                "image": str(Path(image_prefix) / name),
                "lanes_2d": lanes,
                "source": "BDD100K",
                "attributes": item.get("attributes", {}),
            }
        )
    return records


def once(
    root: str | Path,
    listing: str | Path,
    split: str,
    image_prefix: str,
    transform: list[list[float]],
) -> list[dict]:
    """ONCE raw annotations are optical camera xyz. Transform must be declared.

    Official descriptive axes text conflicts with example numeric coordinates and
    calibration usage. This adapter uses the evaluator's optical xyz convention;
    inspect/reproject your release before converting. No guessed ego transform.
    """
    t = torch.tensor(transform, dtype=torch.float32)
    records = []
    for line in Path(listing).read_text().splitlines():
        if not line.strip():
            continue
        name = line.split()[0].lstrip("/")
        ann_path = str(Path(name).with_suffix(".json"))
        raw = json.loads(rooted(root, ann_path).read_text())
        k = torch.tensor(raw["calibration"], dtype=torch.float32)[:, :3]
        validate_calibration(k, t)
        lanes3d, lanes2d = [], []
        for lane in raw["lanes"]:
            optical = torch.tensor(lane, dtype=torch.float32)
            optical = optical[torch.isfinite(optical).all(1) & (optical[:, 2] > 0)]
            if len(optical) < 2:
                continue
            ego = optical @ t[:3, :3].T + t[:3, 3]
            uvw = optical @ k.T
            uv = uvw[:, :2] / uvw[:, 2:]
            lanes3d.append(ego.tolist())
            lanes2d.append(uv.tolist())
        image = str(Path(image_prefix) / Path(name).with_suffix(".jpg"))
        records.append(
            {
                "id": name,
                "group": Path(name).parts[0],
                "split": split,
                "image": image,
                "lanes_2d": lanes2d,
                "lanes_3d": lanes3d,
                "intrinsics": k.tolist(),
                "T_ego_from_camera": transform,
                "source": "ONCE-3DLanes",
                "points": None,
            }
        )
    return records
