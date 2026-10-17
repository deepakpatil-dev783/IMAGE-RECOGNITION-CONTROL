"""
model/trainer.py
-----------------
Training loop with metrics, early stopping, and TensorBoard logging.
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    def __init__(self, model, config: Dict, device: torch.device,
                 class_weights: Optional[torch.Tensor] = None):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Loss
        w = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=w)

        # Optimiser
        tc = config["training"]
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=tc["learning_rate"],
            weight_decay=tc["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=tc["epochs"], eta_min=1e-6
        )
        self.early_stopping = EarlyStopping(patience=tc["early_stopping_patience"])

        # TensorBoard
        self.writer = SummaryWriter("logs/") if TB_AVAILABLE else None
        self.best_f1 = 0.0
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "val_f1": [], "val_precision": [], "val_recall": [],
        }

    # ── Epoch helpers ──────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, list, list]:
        self.model.train(train)
        total_loss, all_preds, all_labels = 0.0, [], []

        with torch.set_grad_enabled(train):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits, _ = self.model(images)
                loss = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        return avg_loss, all_preds, all_labels

    def _compute_metrics(self, preds, labels) -> Dict:
        return {
            "accuracy":  accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall":    recall_score(labels, preds, zero_division=0),
            "f1":        f1_score(labels, preds, zero_division=0),
            "confusion": confusion_matrix(labels, preds).tolist(),
        }

    # ── Main training loop ────────────────────────────────────────────────────

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            checkpoint_dir: str = "models/") -> Dict:
        epochs = self.config["training"]["epochs"]
        print(f"\n{'='*60}")
        print(f"  Training for {epochs} epochs on {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, tr_preds, tr_labels = self._run_epoch(train_loader, train=True)
            val_loss, vl_preds, vl_labels   = self._run_epoch(val_loader,   train=False)
            self.scheduler.step()

            tr_m = self._compute_metrics(tr_preds, tr_labels)
            vl_m = self._compute_metrics(vl_preds, vl_labels)

            # History
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(tr_m["accuracy"])
            self.history["val_acc"].append(vl_m["accuracy"])
            self.history["val_f1"].append(vl_m["f1"])
            self.history["val_precision"].append(vl_m["precision"])
            self.history["val_recall"].append(vl_m["recall"])

            # TensorBoard
            if self.writer:
                self.writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
                self.writer.add_scalars("Accuracy", {"train": tr_m["accuracy"], "val": vl_m["accuracy"]}, epoch)
                self.writer.add_scalar("Val/F1",        vl_m["f1"],        epoch)
                self.writer.add_scalar("Val/Precision", vl_m["precision"], epoch)
                self.writer.add_scalar("Val/Recall",    vl_m["recall"],    epoch)

            elapsed = time.time() - t0
            print(f"Epoch [{epoch:3d}/{epochs}] | {elapsed:.1f}s | "
                  f"Train Loss: {train_loss:.4f}  Acc: {tr_m['accuracy']:.3f} | "
                  f"Val Loss: {val_loss:.4f}  Acc: {vl_m['accuracy']:.3f}  "
                  f"F1: {vl_m['f1']:.3f}")

            # Best model
            if vl_m["f1"] > self.best_f1:
                self.best_f1 = vl_m["f1"]
                from src.model.cnn_model import save_checkpoint
                save_checkpoint(self.model, self.optimizer, epoch, vl_m,
                                f"{checkpoint_dir}/best_model.pth")

            # Early stopping
            if self.early_stopping.step(vl_m["f1"]):
                print(f"\n⚠  Early stopping triggered at epoch {epoch}.")
                break

        print(f"\n✓ Training complete. Best Val F1: {self.best_f1:.4f}")
        if self.writer:
            self.writer.close()
        return self.history
