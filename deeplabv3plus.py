"""
Team Rashmi — DeepLabV3+ with Monte Carlo Dropout.

Implementation strategy:
  * Backbone: ResNet-50 (torchvision pre-trained on ImageNet) — easy to install.
  * Decoder: ASPP module + low-level skip + 4x bilinear upsample.
  * Dropout layers in the decoder are kept ACTIVE at inference time when MC mode is on.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — captures multi-scale context."""

    def __init__(self, in_ch: int, out_ch: int = 256,
                 dilations=(1, 6, 12, 18), dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            kernel = 3 if d > 1 else 1
            padding = d if d > 1 else 0
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, padding=padding,
                          dilation=d, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(dilations) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),  # <-- MC Dropout point
        )

    def forward(self, x):
        size = x.shape[-2:]
        branches = [b(x) for b in self.branches]
        pool = self.image_pool(x)
        pool = F.interpolate(pool, size=size, mode="bilinear", align_corners=False)
        return self.project(torch.cat(branches + [pool], dim=1))


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with a ResNet-50 encoder.

    Set `mc_dropout=True` (during eval) to keep dropout layers active and
    enable Monte Carlo uncertainty estimation.
    """

    def __init__(self, n_classes: int = 6, in_channels: int = 8,
                 dropout: float = 0.1, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # Adapt first conv to accept N input channels (instead of 3)
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
            with torch.no_grad():
                # Copy RGB weights to first 3 channels; replicate mean for the rest
                w = old_conv.weight  # (64, 3, 7, 7)
                new_w = new_conv.weight
                new_w[:, :min(3, in_channels)] = w[:, :min(3, in_channels)]
                if in_channels > 3:
                    new_w[:, 3:] = w.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
            backbone.conv1 = new_conv

        # Expose intermediate features
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # low-level features (1/4 resolution)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4   # high-level features (1/32 resolution)

        self.aspp = ASPP(2048, 256, dropout=dropout)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),  # <-- MC Dropout point
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),  # <-- MC Dropout point
        )
        self.classifier = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.stem(x)
        c1 = self.layer1(x)        # low-level (1/4)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)       # high-level (1/32)

        aspp = self.aspp(c4)       # (B, 256, H/32, W/32)
        aspp = F.interpolate(aspp, size=c1.shape[-2:],
                             mode="bilinear", align_corners=False)
        low = self.low_level_proj(c1)
        merged = torch.cat([aspp, low], dim=1)
        decoded = self.decoder(merged)
        logits = self.classifier(decoded)
        return F.interpolate(logits, size=(h, w),
                             mode="bilinear", align_corners=False)


def enable_mc_dropout(model: nn.Module) -> None:
    """Put model in eval mode but keep all dropout layers active.
    Call this before MC inference.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
