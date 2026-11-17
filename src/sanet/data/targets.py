"""Instance and metric keypoint targets, with explicit collision/missingness masks."""

import cv2
import numpy as np
import torch

from sanet.config import GridConfig


def rasterize(lanes: list, height: int, width: int, thickness: int) -> torch.Tensor:
    instance = np.zeros((height, width), np.int32)
    collision = np.zeros_like(instance, dtype=bool)
    for i, lane in enumerate(lanes, 1):
        p = np.asarray(lane, dtype=np.float32)
        if len(p) < 2:
            continue
        mask = np.zeros_like(instance, dtype=np.uint8)
        cv2.polylines(mask, [np.rint(p).astype(np.int32)], False, 1, thickness=thickness)
        occupied = mask.astype(bool)
        collision |= occupied & (instance > 0) & (instance != i)
        instance[occupied] = i
    instance[collision] = -1
    return torch.from_numpy(instance.astype(np.int64))


def bev_targets(lanes: list, grid: GridConfig) -> dict[str, torch.Tensor]:
    h, w = grid.shape
    instance = np.zeros((h, w), dtype=np.int64)
    offset, height = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    height_valid = np.zeros((h, w), dtype=bool)
    collision = np.zeros((h, w), dtype=bool)
    ys = grid.x_min + (np.arange(h) + 0.5) * grid.resolution
    for identity, lane in enumerate(lanes, 1):
        p = np.asarray(lane, dtype=np.float64)
        # Lane is represented as y(x), z(x); branching/nonmonotonic lanes need another head.
        order = np.argsort(p[:, 0], kind="stable")
        p = p[order]
        _, unique_index = np.unique(p[:, 0], return_index=True)
        p = p[unique_index]
        if len(p) < 2:
            continue
        rows = np.where((ys >= p[0, 0]) & (ys <= p[-1, 0]))[0]
        lateral = np.interp(ys[rows], p[:, 0], p[:, 1])
        z = np.interp(ys[rows], p[:, 0], p[:, 2])
        columns = np.floor((lateral - grid.y_min) / grid.resolution).astype(int)
        good = (columns >= 0) & (columns < w) & (z >= grid.z_min) & (z < grid.z_max)
        rows, columns, lateral, z = rows[good], columns[good], lateral[good], z[good]
        collision[rows, columns] |= instance[rows, columns] > 0
        instance[rows, columns] = identity
        offset[rows, columns] = lateral - (grid.y_min + (columns + 0.5) * grid.resolution)
        height[rows, columns] = z
        height_valid[rows, columns] = True
    instance[collision] = -1
    height_valid[collision] = False
    return {
        "instance": torch.from_numpy(instance),
        "offset": torch.from_numpy(offset)[None],
        "height": torch.from_numpy(height)[None],
        "height_valid": torch.from_numpy(height_valid)[None],
    }
