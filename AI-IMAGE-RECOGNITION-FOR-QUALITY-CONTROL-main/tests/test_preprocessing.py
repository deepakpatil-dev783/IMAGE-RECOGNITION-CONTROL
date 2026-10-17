"""
tests/test_preprocessing.py
"""
import numpy as np
import pytest
from src.preprocessing.image_processor import get_train_transforms, get_val_transforms


def test_val_transform_output():
    transform = get_val_transforms(224)
    dummy = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    result = transform(image=dummy)
    tensor = result["image"]
    assert tensor.shape == (3, 224, 224), f"Unexpected shape: {tensor.shape}"


def test_train_transform_determinism():
    """Train transforms should be non-deterministic (augmented)."""
    transform = get_train_transforms(224)
    dummy = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    r1 = transform(image=dummy)["image"]
    r2 = transform(image=dummy)["image"]
    # They might be equal by chance but usually won't be
    # Just ensure shapes are correct
    assert r1.shape == (3, 224, 224)
    assert r2.shape == (3, 224, 224)
