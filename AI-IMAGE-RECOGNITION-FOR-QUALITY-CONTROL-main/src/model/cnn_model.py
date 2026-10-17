"""
model/cnn_model.py
-------------------
ResNet-50 based CNN for product defect classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Dict, Optional
import yaml
from pathlib import Path


class QualityControlCNN(nn.Module):
    """
    Fine-tuned ResNet-50 for binary quality classification.
    Outputs: (logits, feature_maps) — feature maps enable Grad-CAM.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Keep all layers except the final FC
        self.features = nn.Sequential(*list(backbone.children())[:-2])   # → [B, 2048, 7, 7]
        self.pool = nn.AdaptiveAvgPool2d(1)                               # → [B, 2048, 1, 1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )
        self._freeze_early_layers()

    def _freeze_early_layers(self):
        """Freeze first two ResNet layer-groups; train the rest."""
        children = list(self.features.children())
        for child in children[:6]:
            for param in child.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_maps = self.features(x)          # [B, 2048, 7, 7]
        pooled = self.pool(feat_maps)         # [B, 2048, 1, 1]
        logits = self.classifier(pooled)      # [B, num_classes]
        return logits, feat_maps


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Maps for visualising defect regions."""

    def __init__(self, model: QualityControlCNN):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Hook onto the last conv block
        last_block = list(self.model.features.children())[-1]
        last_block.register_forward_hook(forward_hook)
        last_block.register_full_backward_hook(backward_hook)

    def generate(self, tensor: torch.Tensor, class_idx: int) -> torch.Tensor:
        """
        Returns a (H, W) heatmap tensor (values in [0,1]).
        tensor: [1, C, H, W]
        """
        self.model.eval()
        logits, _ = self.model(tensor)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pool the gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, 7, 7]
        cam = torch.relu(cam).squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── Model I/O ─────────────────────────────────────────────────────────────────

def build_model(config: Dict) -> QualityControlCNN:
    return QualityControlCNN(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    )


def save_checkpoint(model: QualityControlCNN,
                    optimizer,
                    epoch: int,
                    metrics: Dict,
                    path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }, path)
    print(f"Checkpoint saved -> {path}")


def load_checkpoint(model: QualityControlCNN,
                    path: str,
                    device: torch.device,
                    optimizer=None) -> Dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    return checkpoint.get("metrics", {})
