"""Lightweight CNN architectures for mobile cataract triage."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from src.config import DROPOUT_P

SUPPORTED_BACKBONES = ("efficientnet_b0", "mobilenet_v3_small")


class CataractClassifier(nn.Module):
    """Lightweight CNN with Global Average Pooling -> Dropout(0.3) -> Linear(1).

    Outputs raw logits during training (for BCEWithLogitsLoss) and supports
    `predict_proba` with Sigmoid for [0, 1] cataract probability.
    """

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = True,
        dropout_p: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone '{backbone_name}'. Choose from {SUPPORTED_BACKBONES}")

        self.backbone_name = backbone_name

        if backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            self.features = base.features
            in_features = base.classifier[1].in_features
        else:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
            self.features = base.features
            # MobileNetV3-Small features output 576 channels before its classifier
            in_features = base.classifier[0].in_features

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.head(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
