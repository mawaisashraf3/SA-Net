"""Portable active-cell tensors and differentiable sparse submanifold convolution.

Integer coordinate sorting is discrete; feature aggregation and kernel application
remain differentiable. This reference backend favors correctness over CUDA throughput.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SparseBEV:
    coords: torch.Tensor  # N,3: batch,row,column; unique, sorted by linear key
    values: torch.Tensor  # N,C
    batch_size: int
    shape: tuple[int, int]

    def keys(self) -> torch.Tensor:
        h, w = self.shape
        return (self.coords[:, 0] * h + self.coords[:, 1]) * w + self.coords[:, 2]

    def dense(self) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = self.shape
        out = self.values.new_zeros((self.batch_size * h * w, self.values.shape[-1]))
        out = out.index_copy(0, self.keys(), self.values)
        active = torch.zeros(self.batch_size * h * w, dtype=torch.bool, device=self.values.device)
        active[self.keys()] = True
        return (
            out.view(self.batch_size, h, w, -1).permute(0, 3, 1, 2),
            active.view(self.batch_size, 1, h, w),
        )


def aggregate(
    coords: torch.Tensor,
    values: torch.Tensor,
    batch_size: int,
    shape: tuple[int, int],
    weights: torch.Tensor | None = None,
) -> SparseBEV:
    """Weighted cell mean, including correct empty input gradients."""
    h, w = shape
    keys = (coords[:, 0] * h + coords[:, 1]) * w + coords[:, 2]
    unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    if weights is None:
        weights = values.new_ones(len(values))
    # Keep deposited mass and sums in fp32 under AMP; tiny probabilities must
    # not underflow the denominator and turn empty-mass candidates into NaNs.
    original_dtype = values.dtype
    if values.dtype in (torch.float16, torch.bfloat16):
        values, weights = values.float(), weights.float()
    sums = values.new_zeros((len(unique), values.shape[-1]))
    sums = sums.index_add(0, inverse, values * weights[:, None])
    mass = weights.new_zeros(len(unique)).index_add(0, inverse, weights)
    mean = sums / mass[:, None].clamp_min(1e-8)
    c = torch.stack([unique // (h * w), (unique // w) % h, unique % w], 1)
    return SparseBEV(c, mean.to(original_dtype), batch_size, shape)


def union_concat(camera: SparseBEV, lidar: SparseBEV) -> SparseBEV:
    if camera.shape != lidar.shape or camera.batch_size != lidar.batch_size:
        raise ValueError("Sparse tensors must share batch size and grid")
    keys = torch.unique(torch.cat([camera.keys(), lidar.keys()]), sorted=True)
    c = camera.values.new_zeros((len(keys), camera.values.shape[-1]))
    lidar_values = lidar.values.new_zeros((len(keys), lidar.values.shape[-1]))
    c = c.index_copy(0, torch.searchsorted(keys, camera.keys()), camera.values)
    lidar_values = lidar_values.index_copy(0, torch.searchsorted(keys, lidar.keys()), lidar.values)
    h, w = camera.shape
    coords = torch.stack([keys // (h * w), (keys // w) % h, keys % w], 1)
    return SparseBEV(coords, torch.cat([c, lidar_values], 1), camera.batch_size, camera.shape)


class SubmanifoldConv(nn.Module):
    """3x3 sparse cross-correlation, preserving the union's active coordinates."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(3, 3, in_channels, out_channels))
        nn.init.normal_(self.weight, std=(9 * in_channels) ** -0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: SparseBEV) -> SparseBEV:
        keys = x.keys()
        h, w = x.shape
        out = x.values.new_zeros((len(keys), self.bias.numel()))
        if len(keys) == 0:
            out = out + x.values.sum() * 0 + self.weight.sum() * 0
        else:
            for i, dr in enumerate([-1, 0, 1]):
                for j, dc in enumerate([-1, 0, 1]):
                    row, col = x.coords[:, 1] + dr, x.coords[:, 2] + dc
                    wanted = (x.coords[:, 0] * h + row) * w + col
                    pos = torch.searchsorted(keys, wanted).clamp(max=len(keys) - 1)
                    valid = (row >= 0) & (row < h) & (col >= 0) & (col < w) & (keys[pos] == wanted)
                    out = out + (x.values[pos] @ self.weight[i, j]) * valid[:, None]
        return SparseBEV(x.coords, out + self.bias, x.batch_size, x.shape)
