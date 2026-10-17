"""
preprocessing/image_processor.py
---------------------------------
Handles all image loading, augmentation, and dataset preparation.
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Albumentations pipelines ─────────────────────────────────────────────────

def get_train_transforms(input_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(input_size, input_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(input_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class ProductDataset(Dataset):
    """
    Expects directory layout:
        data_dir/
            PASS/    *.jpg | *.png
            DEFECT/  *.jpg | *.png
    """

    CLASS_MAP = {"PASS": 0, "DEFECT": 1}

    def __init__(self, data_dir: str, transform=None, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        for class_name, label in self.CLASS_MAP.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            for path in class_dir.iterdir():
                if path.suffix.lower() in extensions:
                    self.samples.append((path, label))
        print(f"[{self.split}] Loaded {len(self.samples)} samples "
              f"({sum(1 for _, l in self.samples if l==0)} PASS, "
              f"{sum(1 for _, l in self.samples if l==1)} DEFECT)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, label


# ── Single-image preprocessor ─────────────────────────────────────────────────

class ImagePreprocessor:
    """Preprocess a single image for inference."""

    def __init__(self, input_size: int = 224):
        self.input_size = input_size
        self.transform = get_val_transforms(input_size)

    def preprocess(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """Returns (tensor for model, original RGB numpy array)."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image_rgb)
        tensor = augmented["image"].unsqueeze(0)   # [1, C, H, W]
        return tensor, image_rgb

    def preprocess_batch(self, image_paths: List[str]) -> Tuple[torch.Tensor, List[np.ndarray]]:
        tensors, originals = [], []
        for p in image_paths:
            t, orig = self.preprocess(p)
            tensors.append(t)
            originals.append(orig)
        return torch.cat(tensors, dim=0), originals


# ── Utility helpers ───────────────────────────────────────────────────────────

def build_dataloaders(data_dir: str,
                      input_size: int = 224,
                      batch_size: int = 32,
                      val_split: float = 0.2,
                      num_workers: int = 4,
                      seed: int = 42
                      ) -> Tuple[DataLoader, DataLoader]:
    """Split dataset into train/val and return DataLoaders."""
    from torch.utils.data import random_split

    full_dataset = ProductDataset(data_dir, transform=get_train_transforms(input_size))
    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    torch.manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    # Override val transform
    val_ds.dataset = ProductDataset(data_dir, transform=get_val_transforms(input_size), split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def get_class_weights(data_dir: str) -> torch.Tensor:
    """Compute class weights to handle imbalanced datasets."""
    dataset = ProductDataset(data_dir)
    labels = [s[1] for s in dataset.samples]
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float)
