"""Integrated image-plane and calibrated BEV SA-Net modes."""

import torch
from torch import nn
from torch.nn import functional as F

from sanet.config import ModelConfig
from sanet.models.attention import AMPM, ERA
from sanet.models.backbone import CameraFPN, block
from sanet.models.fusion import SparseFusion


class LaneHead(nn.Module):
    def __init__(self, channels, embedding_dim):
        super().__init__()
        self.shared = block(channels, channels)
        self.confidence = nn.Conv2d(channels, 1, 1)
        self.embedding = nn.Conv2d(channels, embedding_dim, 1)
        self.offset = nn.Conv2d(channels, 1, 1)
        self.height = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        x = self.shared(x)
        return {
            "confidence": self.confidence(x),
            "embedding": self.embedding(x),
            "offset": self.offset(x),
            "height": self.height(x),
        }


class SANet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.camera = CameraFPN(cfg.backbone, cfg.channels, cfg.pretrained)
        self.segmentation = nn.Conv2d(cfg.channels, 1, 1)
        self.depth = nn.Conv2d(cfg.channels, cfg.depth_bins, 1)
        self.fusion = SparseFusion(cfg) if cfg.mode == "bev" else None
        self.ampm = AMPM(cfg.channels, cfg.dilations, cfg.base_kernels) if cfg.ampm else None
        self.era = ERA(cfg.channels) if cfg.era else nn.Identity()
        self.head = LaneHead(cfg.channels, cfg.embedding_dim)

    def forward(self, batch: dict, warmup: bool = False) -> dict:
        images = batch["images"]
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("Expected B,V,3,H,W RGB images")
        b, v, _, ih, iw = images.shape
        flat = self.camera(images.flatten(0, 1))
        _, c, h, w = flat.shape
        features = flat.view(b, v, c, h, w)
        segmentation = self.segmentation(flat).view(b, v, 1, h, w)
        depths = self.depth(flat).view(b, v, -1, h, w)
        if self.fusion is not None:
            x, active, stats = self.fusion(features, segmentation, depths, batch, warmup)
        else:
            if v != 1:
                raise ValueError("Image-plane mode requires exactly one camera")
            x = features[:, 0]
            active = torch.ones((b, 1, h, w), device=x.device, dtype=torch.bool)
            stats = {}
        routing = None
        if self.ampm is not None:
            x, routing = self.ampm(x, active)
        maps = self.head(self.era(x))
        if self.cfg.mode == "image":
            maps = {
                k: F.interpolate(t, (ih, iw), mode="bilinear", align_corners=False)
                for k, t in maps.items()
            }
        maps.update(segmentation=segmentation, depth=depths, routing=routing, sparsity=stats)
        return maps
