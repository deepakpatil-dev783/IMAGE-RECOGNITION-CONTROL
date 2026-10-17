"""
train.py
---------
Entry point for training the quality control CNN.

Usage:
    python train.py
    python train.py --data_dir data/raw --epochs 30 --batch_size 16
"""

import argparse
import yaml
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.model.cnn_model import build_model
from src.model.trainer import Trainer
from src.preprocessing.image_processor import build_dataloaders, get_class_weights


def parse_args():
    p = argparse.ArgumentParser(description="Train Quality Control CNN")
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--data_dir",   default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--checkpoint_dir", default="models/")
    return p.parse_args()


def plot_history(history: dict, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend()

    axes[2].plot(history["val_f1"],        label="F1")
    axes[2].plot(history["val_precision"], label="Precision")
    axes[2].plot(history["val_recall"],    label="Recall")
    axes[2].set_title("Val Metrics"); axes[2].legend()

    plt.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}/training_history.png", dpi=120)
    print(f"Training plot saved -> {out_dir}/training_history.png")
    plt.close()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.data_dir:   config["paths"]["data_dir"]      = args.data_dir
    if args.epochs:     config["training"]["epochs"]     = args.epochs
    if args.batch_size: config["training"]["batch_size"] = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = config["paths"]["data_dir"]
    tc       = config["training"]

    train_loader, val_loader = build_dataloaders(
        data_dir,
        input_size=config["model"]["input_size"],
        batch_size=tc["batch_size"],
        val_split=tc["val_split"],
        seed=tc["seed"],
    )

    class_weights = get_class_weights(data_dir)
    model   = build_model(config)
    trainer = Trainer(model, config, device, class_weights)

    history = trainer.fit(train_loader, val_loader,
                          checkpoint_dir=args.checkpoint_dir)

    plot_history(history, args.checkpoint_dir)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
