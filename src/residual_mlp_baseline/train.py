import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import TASKS, TASK_SETTINGS, ResidualMLP, configure_normalization, configure_head, configure_blocks
from .augmentation import augment_batch
from .patch_mlp import PatchResidualMLP


# Baseline settings. Deliberately plain so they are easy to beat:
# plain SGD, constant learning rate, no dropout, no schedule, no augmentation.
LEARNING_RATE = 0.05
BATCH_SIZE = 256
EPOCHS = {"mnist": 10, "cifar10": 30, "imagenet32": 10}


def prepare(X):
    # Convert uint8 images (0-255) to floats in roughly [-1, 1].
    return (torch.as_tensor(X).float() / 255.0 - 0.5) / 0.5


# ---------------------------------------------------------------------------
# The two functions below are the REQUIRED INTERFACE for every submission.
# The grading notebook calls them like this:
#
#     model = train("cifar10", X_train, y_train, seed=0, device="cuda")
#     probs = predict(model, X_test)
#
# Change anything inside them, but keep their names and arguments.
# ---------------------------------------------------------------------------

# Build a fresh model using the task's architecture settings.
def build_model(task):
    settings = TASK_SETTINGS[task]
    dropout = settings.get("dropout", 0.0)
    if not 0 <= dropout <= 1:
        raise ValueError("dropout must be between 0 and 1")

    patch_settings = settings.get("patch_model")
    if patch_settings is not None:
        return PatchResidualMLP(
            **TASKS[task],
            **patch_settings,
            activation=settings["activation"],
            normalization=settings.get("normalization", "layer"),
            block_type=settings.get("block_type", "mlp"),
            maxout_pieces=settings.get("maxout_pieces", 2),
            head_type=settings.get("head_type", "linear"),
            dropout=dropout
        )

    # Preserve the existing flat-model setup.
    model = ResidualMLP(
        **TASKS[task], width=settings["width"],
        hidden=settings["hidden"], depth=settings["depth"],
        activation=settings["activation"]
    )
    model = configure_normalization(
        model, settings.get("normalization", "layer")
    )
    model = configure_head(model, settings.get("head_type", "mlp"))
    model = configure_blocks(
        model, block_type=settings.get("block_type", "mlp"),
        maxout_pieces=settings.get("maxout_pieces", 2)
    )
    if dropout > 0:
        for block in model.blocks:
            block.activation = nn.Sequential(
                block.activation, nn.Dropout(dropout)
            )
    return model


def train(task, X_train, y_train, seed=0, device="cuda"):
    # Build a new model and train it from scratch. Returns the trained model
    torch.manual_seed(seed)                   # makes the run reproducible

    #Get passed in settings 
    settings = TASK_SETTINGS[task]

    # Build the selected architecture before creating the optimizer.
    model = build_model(task).to(device)

    # 2. Wrap the data so we can loop over shuffled mini-batches.
    dataset = TensorDataset(prepare(X_train), torch.as_tensor(y_train).long())
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True)

    # 3. Choose the loss and the optimizer.
    loss_fn = nn.CrossEntropyLoss()           # expects raw logits
    if settings["optimizer"] == "adamw": 
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    else: 
        optimizer = torch.optim.SGD(model.parameters(), lr=settings["lr"], momentum=settings["momentum"], weight_decay=settings["weight_decay"])
    
    #option: define scheduler if passed in
    # Fall back to the existing settings for the other tasks.
    scheduler = None
    scheduler_name = settings.get("scheduler")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"], eta_min=0.00001)

    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, settings["epochs"] // 4), gamma=0.5)

    elif scheduler_name != "none":
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    # 4. The training loop.
    model.train()
    for epoch in range(settings["epochs"]):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            #Use this task's augmentation settings
            X_batch = augment_batch(X_batch, settings.get("augmentation", {}))

            logits = model(X_batch)           # forward pass
            loss = loss_fn(logits, y_batch)   # how wrong are we?

            optimizer.zero_grad()             # clear old gradients
            loss.backward()                   # backward pass: compute gradients
            optimizer.step()                  # update the weights

            total_loss += loss.item() * len(X_batch)

        print(f"epoch {epoch + 1}: train loss {total_loss / len(dataset):.4f}")
        #if scheduler is implemeted then run it 
        if scheduler is not None: 
            scheduler.step()

    return model


def predict(model, X):
    # Return class probabilities for X, one row per example, on the CPU.
    device = next(model.parameters()).device
    X = prepare(X)
    # Predict in chunks so a large test set does not run out of GPU memory.
    chunks = [model.predict(X[i:i + 1024].to(device)).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(chunks)
