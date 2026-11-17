"""Missingness-aware manuscript objectives with stable BCE and instance margins."""

import torch
from torch import nn
from torch.nn import functional as F

from sanet.config import Config


def masked_mean(values, valid):
    mask = valid.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def confidence_loss(logits, target, valid, balance=True, cap=20.0):
    if balance:
        pos = (target * valid).sum()
        neg = ((1 - target) * valid).sum()
        weight = (neg / pos.clamp_min(1)).clamp(1, cap).detach()
    else:
        weight = logits.new_tensor(1.0)
    raw = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight, reduction="none")
    return masked_mean(raw, valid)


def discriminative_embedding(embedding, instances, variance_margin=0.5, distance_margin=1.5):
    """Equal weighting per lane/frame; deterministic spatial subsampling bounds memory."""
    per_frame = []
    for emb, labels in zip(embedding, instances):
        means, variance = [], []
        for identity in torch.unique(labels):
            if identity <= 0:
                continue
            values = emb[:, labels == identity].T
            if len(values) > 256:
                positions = torch.linspace(0, len(values) - 1, 256, device=values.device).long()
                values = values[positions]
            center = values.mean(0)
            means.append(center)
            variance.append(
                F.relu(torch.linalg.vector_norm(values - center, dim=1) - variance_margin)
                .square()
                .mean()
            )
        if not means:
            per_frame.append(emb.sum() * 0)
            continue
        centers = torch.stack(means)
        var = torch.stack(variance).mean()
        dist = emb.sum() * 0
        if len(centers) > 1:
            distance = torch.pdist(centers)
            dist = F.relu(2 * distance_margin - distance).square().mean()
        per_frame.append(var + dist + 0.001 * torch.linalg.vector_norm(centers, dim=1).mean())
    return torch.stack(per_frame).mean()


class LaneLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "bins", torch.linspace(cfg.model.depth_min, cfg.model.depth_max, cfg.model.depth_bins)
        )

    def forward(self, output: dict, target: dict) -> dict:
        cfg = self.cfg
        labels = target["instance"]
        valid = (labels >= 0)[:, None]
        foreground = (labels > 0)[:, None]
        losses = {
            "confidence": confidence_loss(
                output["confidence"],
                foreground.float(),
                valid,
                cfg.train.balanced_confidence,
                cfg.train.max_positive_weight,
            ),
            "embedding": discriminative_embedding(
                output["embedding"], labels, cfg.train.variance_margin, cfg.train.distance_margin
            ),
        }
        zero = output["confidence"].sum() * 0
        if cfg.model.mode == "bev":
            losses["offset"] = masked_mean(
                (output["offset"] - target["offset"]).square(), foreground
            )
            losses["height"] = masked_mean(
                (output["height"] - target["height"]).square(), foreground & target["height_valid"]
            )
        else:
            losses.update(offset=zero, height=zero)
        shape = output["segmentation"].shape
        seg = output["segmentation"].flatten(0, 1)
        image_labels = target["image_instance"].flatten(0, 1)[:, None]
        # Any lane pixel contributes when reducing thin markings to feature resolution.
        seg_target = F.adaptive_max_pool2d((image_labels > 0).float(), shape[-2:])
        seg_valid = F.adaptive_max_pool2d((image_labels < 0).float(), shape[-2:]) == 0
        losses["segmentation"] = confidence_loss(
            seg, seg_target, seg_valid, cfg.train.balanced_confidence, cfg.train.max_positive_weight
        )
        depth = target["depth"].flatten(0, 1)[:, None]
        valid_depth = (depth >= self.bins[0]) & (depth <= self.bins[-1]) & torch.isfinite(depth)
        depth = torch.where(valid_depth, depth, torch.inf)
        nearest = -F.adaptive_max_pool2d(-depth, shape[-2:])[:, 0]
        good = torch.isfinite(nearest)
        logits = output["depth"].flatten(0, 1).permute(0, 2, 3, 1)
        if good.any():
            indices = (nearest[good, None] - self.bins).abs().argmin(-1)
            losses["depth"] = F.cross_entropy(logits[good], indices)
        else:
            losses["depth"] = output["depth"].sum() * 0
        losses["total"] = sum(getattr(cfg.train, name) * value for name, value in losses.items())
        return losses
