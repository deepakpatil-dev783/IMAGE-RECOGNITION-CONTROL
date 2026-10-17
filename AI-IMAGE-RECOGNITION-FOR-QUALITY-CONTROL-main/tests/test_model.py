"""
tests/test_model.py
"""
import torch
import pytest
from src.model.cnn_model import QualityControlCNN, GradCAM


def test_model_output_shape():
    model = QualityControlCNN(num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    logits, feat_maps = model(x)
    assert logits.shape == (2, 2), f"Expected (2,2), got {logits.shape}"
    assert feat_maps.shape[1] == 2048, "Expected 2048 channels in feature maps"


def test_model_probabilities():
    import torch.nn.functional as F
    model = QualityControlCNN(num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    logits, _ = model(x)
    probs = F.softmax(logits, dim=1)
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_grad_cam_output():
    model = QualityControlCNN(num_classes=2, pretrained=False)
    grad_cam = GradCAM(model)
    x = torch.randn(1, 3, 224, 224)
    cam = grad_cam.generate(x, class_idx=1)
    assert cam.ndim == 2
    assert cam.min().item() >= 0.0
    assert cam.max().item() <= 1.0 + 1e-5


def test_unfreeze():
    model = QualityControlCNN(num_classes=2, pretrained=False)
    model.unfreeze_all()
    for p in model.parameters():
        assert p.requires_grad
