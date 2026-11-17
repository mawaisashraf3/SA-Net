"""ResNet-34/FPN and an explicit compact CPU experiment alternative."""

import torch
from torch import nn
from torch.nn import functional as F


def norm(channels: int) -> nn.GroupNorm:
    # At least two channels per group for stable tiny maps.
    groups = min(8, max(1, channels // 2))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        norm(cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        norm(cout),
        nn.SiLU(),
    )


class CameraFPN(nn.Module):
    def __init__(self, kind: str = "resnet34", channels: int = 64, pretrained: bool = False):
        super().__init__()
        if kind == "resnet34":
            from torchvision.models import ResNet34_Weights, resnet34

            net = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
            widths = [64, 128, 256, 512]
        elif kind == "compact":
            if pretrained:
                raise ValueError("Compact backbone has no pretrained weights")
            self.stem = block(3, 16, 2)
            self.stages = nn.ModuleList([block(16, 24, 2), block(24, 40, 2), block(40, 64, 2)])
            widths = [24, 40, 64]
        else:
            raise ValueError(f"Unknown backbone {kind}")
        self.lateral = nn.ModuleList([nn.Conv2d(w, channels, 1) for w in widths])
        self.smooth = block(channels, channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        features = []
        for stage, lateral in zip(self.stages, self.lateral):
            x = stage(x)
            features.append(lateral(x))
        fused = features[-1]
        for low in reversed(features[:-1]):
            fused = low + F.interpolate(fused, low.shape[-2:], mode="bilinear", align_corners=False)
        return self.smooth(fused)
