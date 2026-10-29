import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from residual_mlp_baseline.model import TASKS, TASK_SETTINGS, ResidualMLP
from residual_mlp_baseline.train import build_model, prepare
from residual_mlp_baseline.augmentation import augment_batch

# Keep the original baseline recipe; the best model reads its own settings.
BASELINE_SETTINGS = {
    "epochs": 30, "batch_size": 256, "lr": 0.05,
    "optimizer": "sgd", "momentum": 0.0, "weight_decay": 0.0,
    "scheduler": "none", "dropout": 0.0, "augmentation": {},
}
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data" / "cifar10"
OUTPUT_DIR = ROOT / "cifar_comparison"


def train_model(name, dataset):
    # Start each model from fresh weights and the same seed.
    torch.manual_seed(SEED)
    if name == "Original baseline":
        settings = BASELINE_SETTINGS
        model = ResidualMLP(
            **TASKS["cifar10"], width=128, hidden=256, depth=30,
            activation="tanh",
        )
    else:
        settings = TASK_SETTINGS["cifar10"]
        model = build_model("cifar10")
    model = model.to(DEVICE)
    model.train()

    # Use a separate generator so initialization does not change the shuffle.
    loader = DataLoader(
        dataset, batch_size=settings["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    if settings["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=settings["lr"], momentum=settings["momentum"],
            weight_decay=settings["weight_decay"],
        )
    scheduler = None
    if settings["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=settings["epochs"], eta_min=0.00001,
        )
    elif settings["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, settings["epochs"] // 4), gamma=0.5,
        )
    elif settings["scheduler"] != "none":
        raise ValueError(f"Unknown scheduler: {settings['scheduler']}")
    criterion = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(1, settings["epochs"] + 1):
        total_loss = 0.0
        for images, labels in loader:
            images = prepare(images.to(DEVICE))
            labels = labels.to(DEVICE)
            images = augment_batch(images, settings.get("augmentation", {}))
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        # Weight batches by their size, including the smaller final batch.
        losses.append(total_loss / len(dataset))
        print(f"{name}: epoch {epoch}/{settings['epochs']}, train loss {losses[-1]:.4f}", flush=True)
        if scheduler is not None:
            scheduler.step()
    return losses


def main():
    # Use all cached training images; conversion to float happens per batch.
    images = torch.from_numpy(np.load(DATA_DIR / "X_train.npy"))
    labels = torch.from_numpy(np.load(DATA_DIR / "y_train.npy")).long()
    dataset = TensorDataset(images, labels)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}; training images: {len(dataset):,}", flush=True)

    curves = {}
    for name in ("Original baseline", "Current best"):
        curves[name] = train_model(name, dataset)
        # Save actual measurements after each model finishes.
        with (OUTPUT_DIR / "training_losses.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["model", "epoch", "train_loss"])
            for run, losses in curves.items():
                writer.writerows((run, epoch, loss) for epoch, loss in enumerate(losses, 1))

    # Record the settings used, including the architecture options from model.py.
    settings = {
        "seed": SEED, "device": DEVICE, "training_images": len(dataset),
        "baseline_architecture": {"width": 128, "hidden": 256, "depth": 30, "activation": "tanh"},
        "baseline_training": BASELINE_SETTINGS,
        "current_best": TASK_SETTINGS["cifar10"],
    }
    (OUTPUT_DIR / "settings.json").write_text(json.dumps(settings, indent=2))

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name, losses in curves.items():
        axis.plot(range(1, len(losses) + 1), losses, label=f"{name} ({len(losses)} epochs)")
    axis.set(
        title="CIFAR-10: original baseline versus current best recipe",
        xlabel="Epoch", ylabel="Training cross-entropy loss",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curve_comparison.png", dpi=300)
    plt.close(figure)
    print(f"Saved curves and graph to: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
