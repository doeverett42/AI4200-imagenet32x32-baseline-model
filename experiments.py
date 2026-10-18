from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  
training = import_module("residual_mlp_baseline.train")

DATA_DIR = ROOT / "data"  
rng = np.random.default_rng(0)

for task in ("mnist", "cifar10", "imagenet32"):
    print(f"\nChecking {task}...")

    # Read selected images without loading the entire dataset into RAM
    X = np.load(DATA_DIR / task / "X_train.npy", mmap_mode="r")
    y = np.load(DATA_DIR / task / "y_train.npy", mmap_mode="r")

    indices = rng.choice(len(y), size=160, replace=False)
    train_indices = indices[:128]
    check_indices = indices[128:]

    saved_epochs = training.EPOCHS[task]
    saved_batch_size = training.BATCH_SIZE

    try:
        #Temporary local settings: one epoch with four batches
        training.EPOCHS[task] = 1
        training.BATCH_SIZE = 32

        model = training.train(
            task,
            X[train_indices],
            y[train_indices],
            seed=0,
            device="cuda"
        )

        probabilities = training.predict(model, X[check_indices])

        #Check that predictions have the correct format and valid values
        assert probabilities.shape == (
            32, training.TASKS[task]["num_classes"]
        )
        assert torch.isfinite(probabilities).all()
        assert (probabilities >= 0).all()
        assert torch.allclose(
            probabilities.sum(dim=1), torch.ones(32), atol=1e-5
        )

        #Choose the class with the highest probability for each image
        predicted_labels = probabilities.argmax(dim=1)

        #Get the correct labels for the held-out images
        true_labels = torch.as_tensor(y[check_indices], dtype=torch.long)

        #Accuracy = number correct / number checked
        correct = (predicted_labels == true_labels).sum().item()
        total = len(check_indices)
        accuracy = correct / total

        print(
            f"{task}: checks passed. "
            f"Held-out accuracy: {accuracy:.2%} ({correct}/{total} correct)"
        )

    finally:
        #Restore settings even if training or a check fails
        training.EPOCHS[task] = saved_epochs
        training.BATCH_SIZE = saved_batch_size