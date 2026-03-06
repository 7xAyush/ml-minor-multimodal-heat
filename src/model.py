from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalNet(nn.Module):
    def __init__(self, tabular_input_dim: int, num_classes: int = 3):
        super().__init__()

        # Image branch: ResNet18 backbone
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Remove final FC layer (keep up to avgpool)
        modules = list(backbone.children())[:-1]
        self.image_encoder = nn.Sequential(*modules)  # (N, 512, 1, 1)

        # Tabular branch
        self.tabular_encoder = TabularEncoder(input_dim=tabular_input_dim)

        image_feature_dim = 512
        tabular_feature_dim = 64
        fusion_input_dim = image_feature_dim + tabular_feature_dim  # 576

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        img_feats = self.image_encoder(images)              # (N, 512, 1, 1)
        img_feats = img_feats.view(img_feats.size(0), -1)   # (N, 512)

        tab_feats = self.tabular_encoder(tabular)           # (N, 64)

        fused = torch.cat([img_feats, tab_feats], dim=1)    # (N, 576)
        logits = self.fusion_head(fused)                    # (N, num_classes)
        return logits
