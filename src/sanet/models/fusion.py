"""Lane-selected camera-z lifting, voxel point encoding, and sparse union fusion."""

import torch
from torch import nn
from torch.nn import functional as F

from sanet.config import ModelConfig
from sanet.geometry import backproject
from sanet.models.sparse import SparseBEV, SubmanifoldConv, aggregate, union_concat


class SparseLift(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("depths", torch.linspace(cfg.depth_min, cfg.depth_max, cfg.depth_bins))

    def forward(
        self,
        features,
        segmentation,
        depth_logits,
        intrinsics,
        transforms,
        image_shape,
        warmup: bool = False,
    ):
        b, views, channels, h, w = features.shape
        g, cfg = self.cfg.grid, self.cfg
        coords, values, masses = [], [], []
        selected_count = 0
        ih, iw = image_shape
        for bi in range(b):
            for vi in range(views):
                lane_p = segmentation[bi, vi, 0].sigmoid().flatten()
                if warmup or not cfg.sfm:
                    selected = torch.arange(h * w, device=features.device)
                else:
                    selected = torch.where(lane_p >= cfg.lane_threshold)[0]
                    n = min(h * w, max(cfg.min_pixels, min(len(selected), cfg.max_pixels)))
                    # Scores sorted in a deterministic order, without random fallback pixels.
                    selected = torch.argsort(lane_p, descending=True, stable=True)[:n]
                selected_count += len(selected)
                uv = torch.stack(
                    [((selected % w) + 0.5) * iw / w - 0.5, ((selected // w) + 0.5) * ih / h - 0.5],
                    -1,
                )
                probs = depth_logits[bi, vi].flatten(1)[:, selected].T.softmax(-1)
                k = cfg.top_k if cfg.sfm else cfg.depth_bins
                mass, depth_index = probs.topk(k, dim=-1)
                xyz = backproject(
                    uv[:, None].expand(-1, k, -1),
                    self.depths[depth_index],
                    intrinsics[bi, vi],
                    transforms[bi, vi],
                )
                row = torch.floor((xyz[..., 0] - g.x_min) / g.resolution).long()
                col = torch.floor((xyz[..., 1] - g.y_min) / g.resolution).long()
                gh, gw = g.shape
                valid = (
                    (row >= 0)
                    & (row < gh)
                    & (col >= 0)
                    & (col < gw)
                    & (xyz[..., 2] >= g.z_min)
                    & (xyz[..., 2] < g.z_max)
                )
                c = torch.stack([torch.full_like(row, bi), row, col], -1)[valid]
                f = features[bi, vi].flatten(1).T[selected]
                v = f[:, None].expand(-1, k, -1)[valid]
                weight = (mass * lane_p[selected, None])[valid]
                coords.append(c)
                values.append(v)
                masses.append(weight)
        result = aggregate(torch.cat(coords), torch.cat(values), b, g.shape, torch.cat(masses))
        return result, {
            "selected_pixels": selected_count,
            "total_pixels": b * views * h * w,
            "camera_cells": len(result.coords),
        }


class VoxelPointEncoder(nn.Module):
    """Mean-centered point MLP → voxel max → BEV mean, sparse until fusion.

    This is a portable voxel feature encoder, not a full VoxelNet sparse backbone.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(10, cfg.channels),
            nn.LayerNorm(cfg.channels),
            nn.SiLU(),
            nn.Linear(cfg.channels, cfg.channels),
            nn.SiLU(),
        )

    def forward(self, clouds: list[torch.Tensor], batch_size: int) -> SparseBEV:
        g, c = self.cfg.grid, self.cfg.channels
        coords, values = [], []
        anchor = next(self.parameters())
        gh, gw = g.shape
        nz = round((g.z_max - g.z_min) / g.voxel_height)
        for bi, points in enumerate(clouds):
            if points.ndim != 2 or points.shape[1] != 4:
                raise ValueError("Point cloud must be N x 4 [x,y,z,intensity]")
            valid = (
                torch.isfinite(points).all(1)
                & (points[:, 0] >= g.x_min)
                & (points[:, 0] < g.x_max)
                & (points[:, 1] >= g.y_min)
                & (points[:, 1] < g.y_max)
                & (points[:, 2] >= g.z_min)
                & (points[:, 2] < g.z_max)
            )
            p = points[valid]
            if len(p) == 0:
                continue
            scale = p.new_tensor([g.resolution, g.resolution, g.voxel_height])
            origin = p.new_tensor([g.x_min, g.y_min, g.z_min])
            indices = ((p[:, :3] - origin) / scale).floor().long()
            key = (indices[:, 0] * gw + indices[:, 1]) * nz + indices[:, 2]
            keys, inverse, count = torch.unique(
                key, sorted=True, return_inverse=True, return_counts=True
            )
            mean = p.new_zeros((len(keys), 3)).index_add(0, inverse, p[:, :3])
            mean = mean / count[:, None]
            center = (indices + 0.5) * scale + origin
            feature = self.net(torch.cat([p, p[:, :3] - mean[inverse], p[:, :3] - center], 1))
            voxel = feature.new_full((len(keys), c), -torch.inf)
            voxel = voxel.scatter_reduce(
                0, inverse[:, None].expand(-1, c), feature, reduce="amax", include_self=True
            )
            xykey = keys // nz
            coords.append(torch.stack([torch.full_like(xykey, bi), xykey // gw, xykey % gw], 1))
            values.append(voxel)
        if not coords:
            return aggregate(
                torch.empty((0, 3), dtype=torch.long, device=anchor.device),
                anchor.new_zeros((0, c)) + anchor.sum() * 0,
                batch_size,
                g.shape,
            )
        return aggregate(torch.cat(coords), torch.cat(values), batch_size, g.shape)


class SparseFusion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.lift = SparseLift(cfg)
        self.lidar = VoxelPointEncoder(cfg)
        self.conv = SubmanifoldConv(cfg.channels * 2, cfg.channels)
        self.norm = nn.LayerNorm(cfg.channels)

    def forward(self, features, segmentation, depths, batch, warmup=False):
        camera, stats = self.lift(
            features,
            segmentation,
            depths,
            batch["intrinsics"],
            batch["transforms"],
            batch["images"].shape[-2:],
            warmup,
        )
        lidar = self.lidar(batch["points"], len(features))
        union = union_concat(camera, lidar)
        fused = self.conv(union)
        fused.values = F.silu(self.norm(fused.values))
        dense, active = fused.dense()
        stats.update(lidar_cells=len(lidar.coords), union_cells=len(union.coords))
        return dense, active, stats
