"""AMPM dynamic kernel mixtures and ERA parallel directional/channel attention."""

import torch
from torch import nn
from torch.nn import functional as F

from sanet.models.backbone import norm


def active_descriptor(x, active, beta):
    mask = active.to(x.dtype)
    mean = (x * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1)
    maximum = x.masked_fill(~active, -torch.inf).amax((-2, -1))
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return beta * mean + (1 - beta) * maximum


class DynamicBranch(nn.Module):
    def __init__(self, channels: int, base_kernels: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.kernels = nn.Parameter(torch.randn(base_kernels, channels, 3, 3) / 3)
        self.router = nn.Linear(channels, base_kernels)
        self.pointwise = nn.Conv2d(channels, channels, 1)

    def forward(self, x, descriptor):
        b, c, h, w = x.shape
        gamma = self.router(descriptor).softmax(-1)
        weights = torch.einsum("bm,mcij->bcij", gamma, self.kernels).reshape(b * c, 1, 3, 3)
        result = F.conv2d(
            x.reshape(1, b * c, h, w),
            weights,
            groups=b * c,
            padding=self.dilation,
            dilation=self.dilation,
        ).reshape(b, c, h, w)
        return self.pointwise(F.silu(result))


class AMPM(nn.Module):
    def __init__(self, channels, dilations, base_kernels):
        super().__init__()
        self.beta_logit = nn.Parameter(torch.tensor(0.0))
        self.router = nn.Linear(channels, len(dilations))
        self.branches = nn.ModuleList([DynamicBranch(channels, base_kernels, d) for d in dilations])
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            norm(channels * 2),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x, active):
        descriptor = active_descriptor(x, active, self.beta_logit.sigmoid())
        alpha = self.router(descriptor).softmax(-1)
        expert = sum(
            alpha[:, i, None, None, None] * branch(x, descriptor)
            for i, branch in enumerate(self.branches)
        )
        return x + self.ffn(expert), alpha


class ERA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(4, channels // 4)
        self.height = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))
        self.width = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.channel = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        spatial = x * self.height(x.mean(3, keepdim=True)).sigmoid()
        spatial = spatial * self.width(x.mean(2, keepdim=True)).sigmoid()
        channel = x * self.channel(x.mean((2, 3), keepdim=True))
        return x + self.fuse(torch.cat([spatial, channel], 1))
