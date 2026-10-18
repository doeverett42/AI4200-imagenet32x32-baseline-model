import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import TASKS, TASK_SETTINGS, ResidualMLP


# Baseline settings. Deliberately plain so they are easy to beat:
# plain SGD, constant learning rate, no dropout, no schedule, no augmentation.
LEARNING_RATE = 0.05
BATCH_SIZE = 256
EPOCHS = {"mnist": 10, "cifar10": 30, "imagenet32": 10}


def prepare(X):
    """Convert uint8 images (0-255) to floats in roughly [-1, 1]."""
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

def train(task, X_train, y_train, seed=0, device="cuda"):
    """Build a new model and train it from scratch. Returns the trained model."""
    torch.manual_seed(seed)                   # makes the run reproducible

    #Get passed in settings 
    settings = TASK_SETTINGS[task]

    # 1. Build the model for this task. Only input size and class count differ.
    model = ResidualMLP(**TASKS[task], width=settings["width"], hidden=settings["hidden"], depth=settings["depth"], activation=settings["activation"]).to(device)

    # 2. Wrap the data so we can loop over shuffled mini-batches.
    dataset = TensorDataset(prepare(X_train), torch.as_tensor(y_train).long())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Choose the loss and the optimizer.
    loss_fn = nn.CrossEntropyLoss()           # expects raw logits
    if settings["optimizer"] == "adamw": 
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    else: 
        optimizer = torch.optim.SGD(model.parameters(), lr=settings["lr"], momentum=settings["momentum"], weight_decay=settings["weight_decay"])
    
    #option: define scheduler if passed in
    scheduler = None 
    if settings["step_decay"]: 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, EPOCHS[task] // 3), gamma=0.5) #keep initial steps few

    # 4. The training loop.
    model.train()
    for epoch in range(EPOCHS[task]):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

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
    """Return class probabilities for X, one row per example, on the CPU."""
    device = next(model.parameters()).device
    X = prepare(X)
    # Predict in chunks so a large test set does not run out of GPU memory.
    chunks = [model.predict(X[i:i + 1024].to(device)).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(chunks)
