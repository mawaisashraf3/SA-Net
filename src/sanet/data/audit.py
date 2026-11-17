"""Detect split leakage, ambiguous labels, and invalid sensor inputs before training."""

import hashlib
from collections import Counter

import numpy as np
import torch
from PIL import Image

from sanet.geometry import validate_calibration
from sanet.io import read_jsonl, rooted, sha256


def audit_manifest(manifest, root, mode="image", check_hashes=True) -> dict:
    records = read_jsonl(manifest)
    if not records:
        raise ValueError("Manifest is empty")
    ids, groups, hashes, paths, point_hashes = set(), {}, {}, {}, {}
    asset_hashes = []
    class_counts, splits, sensors = Counter(), Counter(), Counter()
    group_overlaps = set()
    for record in records:
        identity, group, split = record["id"], record["group"], record["split"]
        if split not in {"train", "val", "test", "inference"}:
            raise ValueError(f"Unknown split {split}")
        if not isinstance(group, str) or not group:
            raise ValueError("Every record needs a nonempty sequence group")
        if identity in ids:
            raise ValueError(f"Duplicate sample id: {identity}")
        ids.add(identity)
        groups.setdefault(group, set()).add(split)
        if len(groups[group]) > 1:
            group_overlaps.add(group)
        splits[split] += 1
        if split != "inference" and "views" not in record and "lanes_2d" not in record:
            raise ValueError(f"Missing 2D annotation (empty list is valid): {identity}")
        for key, dim in [("lanes_2d", 2), ("lanes_3d", 3)]:
            for lane in record.get(key, []):
                p = np.asarray(lane)
                if p.ndim != 2 or p.shape[1] != dim or len(p) < 2 or not np.isfinite(p).all():
                    raise ValueError(f"Invalid {key} polyline in {identity}")
        class_counts[len(record.get("lanes_2d", []))] += 1
        views = record.get("views", [record])
        if mode == "image" and len(views) != 1:
            raise ValueError("Image mode requires one view")
        for view in views:
            path = rooted(root, view["image"])
            if str(path) in paths:
                raise ValueError(f"Duplicate image path: {path}")
            paths[str(path)] = split
            with Image.open(path) as im:
                im.verify()
            if check_hashes:
                digest = sha256(path)
                if digest in hashes:
                    raise ValueError(f"Duplicate image bytes: {identity}, {hashes[digest]}")
                hashes[digest] = identity
                asset_hashes.append((view["image"], digest))
            if split != "inference" and "lanes_2d" not in view:
                raise ValueError(f"Each view needs its own 2D annotations: {identity}")
            for lane in view.get("lanes_2d", []):
                vertices = np.asarray(lane)
                if (
                    vertices.ndim != 2
                    or vertices.shape[1] != 2
                    or len(vertices) < 2
                    or not np.isfinite(vertices).all()
                ):
                    raise ValueError(f"Invalid image polyline: {identity}")
            if mode == "bev":
                if "intrinsics" not in view or "T_ego_from_camera" not in view:
                    raise ValueError(f"BEV mode requires real calibration: {identity}")
                validate_calibration(
                    torch.tensor(view["intrinsics"], dtype=torch.float32),
                    torch.tensor(view["T_ego_from_camera"], dtype=torch.float32),
                )
        if mode == "bev" and split != "inference" and "lanes_3d" not in record:
            raise ValueError(f"Missing 3D annotations: {identity}")
        if record.get("points"):
            p = np.load(rooted(root, record["points"]), allow_pickle=False)
            if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all():
                raise ValueError(f"Invalid N x 4 point cloud: {identity}")
            sensors["lidar"] += 1
            if check_hashes:
                digest = sha256(rooted(root, record["points"]))
                if digest in point_hashes and point_hashes[digest] != split:
                    raise ValueError(f"Point-cloud bytes shared across splits: {identity}")
                point_hashes[digest] = split
                asset_hashes.append((record["points"], digest))
        else:
            sensors["camera_only"] += 1
    # Official CULane uses some common drive identifiers in its provided split.
    # Default is strict. Callers must explicitly remap validated finer groups if justified.
    if group_overlaps:
        raise ValueError(f"Sequence-group split leakage: {sorted(group_overlaps)[:10]}")
    dataset_digest = hashlib.sha256(sha256(manifest).encode())
    for path, digest in sorted(asset_hashes):
        dataset_digest.update(f"{path}\0{digest}\n".encode())
    return {
        "records": len(records),
        "splits": dict(splits),
        "groups": len(groups),
        "lane_count_histogram": dict(class_counts),
        "sensors": dict(sensors),
        "image_hashes_checked": check_hashes,
        "manifest_sha256": sha256(manifest),
        "dataset_sha256": dataset_digest.hexdigest() if check_hashes else None,
    }
