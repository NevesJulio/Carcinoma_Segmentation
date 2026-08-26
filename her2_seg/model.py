from __future__ import annotations

import torch
import torch.nn as nn


def create_model(architecture="unet", encoder="resnet50", channels=3, encoder_weights="imagenet"):
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError("Instale as dependências com: pip install -r requirements.txt") from exc
    models = {"unet": smp.Unet, "fpn": smp.FPN, "deeplabv3plus": smp.DeepLabV3Plus}
    if architecture not in models:
        raise ValueError(f"Arquitetura inválida: {architecture}; opções: {', '.join(models)}")
    return models[architecture](encoder_name=encoder, encoder_weights=encoder_weights, in_channels=channels, classes=1)


class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=.5, focal_weight=.5, gamma=2.0):
        super().__init__()
        self.dw, self.fw, self.gamma = dice_weight, focal_weight, gamma

    def forward(self, logits, target):
        prob = logits.sigmoid()
        intersection = (prob * target).sum((1, 2, 3))
        dice = 1 - ((2 * intersection + 1) / (prob.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = prob * target + (1 - prob) * (1 - target)
        focal = (((1 - pt) ** self.gamma) * bce).mean()
        return self.dw * dice + self.fw * focal


@torch.no_grad()
def dice_score(logits, target, threshold=.5):
    pred = (logits.sigmoid() >= threshold).float()
    return ((2 * (pred * target).sum() + 1) / (pred.sum() + target.sum() + 1)).item()
