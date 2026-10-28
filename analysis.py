# ImageNet training dynamics and branch ablation
#
# I train the current ImageNet model from scratch and measure how it changes.
# The same clean batch is used at initialization, early, midway, and at the end.
# Measurements run in eval mode on a copy, so they do not change training.
# This is a separate analysis run; the graded train() function is unchanged.

import copy
import csv
import hashlib
import json
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# Use the source files beside this script instead of an older installed package.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from residual_mlp_baseline.model import TASK_SETTINGS, TASKS
from residual_mlp_baseline.train import build_model, prepare
from residual_mlp_baseline.augmentation import augment_batch


# ---------------------------------------------------------------------------
# 1. Analysis settings
# ---------------------------------------------------------------------------
# Model and training choices still come from TASK_SETTINGS in model.py.
TASK = "imagenet32"
SEED = 0
VALIDATION_PER_CLASS = 5       # 5 x 1000 classes gives 5000 validation images.
PROBE_BATCH_SIZE = 128        # These same images are used at every measurement.
DATA_DIR = ROOT / "data" / TASK
RUN_NAME = "imagenet32_final_updated"
OUTPUT_DIR = ROOT / "analysis_results" / RUN_NAME
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPSILON = 1e-12               # Avoid dividing by zero when a stream is all zero.


def write_csv(path, rows):
    # Save after each measurement so completed results survive an interruption.
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2), encoding="utf-8")


def rms(tensor):
    # Square, average, and take the square root to measure typical magnitude.
    return tensor.detach().float().square().mean().sqrt().item()


def variance(tensor):
    # Population variance over all batch, patch, and feature entries.
    return tensor.detach().float().var(unbiased=False).item()


def rng_devices(device):
    # fork_rng saves CPU randomness and the selected GPU's randomness.
    device = torch.device(device)
    if device.type != "cuda":
        return []
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return [index]


@contextmanager
def evaluation_mode(model):
    # Restore each module's original mode after measuring or validating.
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, was_training in states:
            module.training = was_training


# ---------------------------------------------------------------------------
# 2. Load the existing cache and create a fixed validation split
# ---------------------------------------------------------------------------
class CachedImages(Dataset):
    # Copy one uint8 image at a time instead of making a huge float image array.
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        selected = self.indices[index]
        image = torch.from_numpy(np.array(self.images[selected], copy=True))
        return image, int(self.labels[selected])


def split_indices(labels, per_class=VALIDATION_PER_CLASS, seed=SEED):
    # Hold out the same number from each class, using only the training cache.
    generator = np.random.default_rng(seed)
    validation = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) <= per_class:
            raise ValueError(f"Class {label} needs more than {per_class} images")
        validation.extend(generator.permutation(indices)[:per_class])

    validation = np.sort(np.asarray(validation, dtype=np.int64))
    train_mask = np.ones(len(labels), dtype=bool)
    train_mask[validation] = False
    return np.flatnonzero(train_mask), validation


def load_cache():
    for filename in ("X_train.npy", "y_train.npy"):
        path = DATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Use the existing harness cache or edit DATA_DIR."
            )

    # Memory mapping reads images as needed instead of loading the whole cache.
    images = np.load(DATA_DIR / "X_train.npy", mmap_mode="r")
    labels = np.load(DATA_DIR / "y_train.npy", mmap_mode="r")
    if images.dtype != np.uint8 or images.shape[1:] != (3, 32, 32):
        raise ValueError("Expected uint8 RGB images shaped (N, 3, 32, 32)")
    if labels.ndim != 1 or len(images) != len(labels):
        raise ValueError("Expected one label for each training image")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected integer class labels")
    expected_labels = np.arange(TASKS[TASK]["num_classes"])
    if not np.array_equal(np.unique(labels), expected_labels):
        raise ValueError("Cache labels do not match the task's class numbers")
    return images, labels


# ---------------------------------------------------------------------------
# 3. Match the current optimizer and scheduler in train.py
# ---------------------------------------------------------------------------
def make_optimizer(model, settings):
    if settings["optimizer"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=settings["lr"],
            weight_decay=settings["weight_decay"],
        )
    if settings["optimizer"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=settings["lr"],
            momentum=settings["momentum"],
            weight_decay=settings["weight_decay"],
        )
    raise ValueError(f"Unknown optimizer: {settings['optimizer']}")


def make_scheduler(optimizer, settings):
    name = settings.get("scheduler", "none")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=settings["epochs"], eta_min=0.00001
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, settings["epochs"] // 4), gamma=0.5
        )
    if name != "none":
        raise ValueError(f"Unknown scheduler: {name}")
    return None


def residual_branches(model):
    # Each entry identifies a branch and whether its output includes the skip.
    branches = []
    for depth, block in enumerate(model.blocks, start=1):
        if hasattr(block, "token_branch"):
            # Patch blocks add token mixing first, then channel mixing.
            for kind in ("token", "channel"):
                name = f"blocks.{depth - 1}.{kind}_branch"
                module = getattr(block, f"{kind}_branch")
                branches.append((depth, kind, name, module, True))
        else:
            # A flat residual block returns its input plus its branch output.
            name = f"blocks.{depth - 1}"
            branches.append((depth, "branch", name, block, False))
    return branches


# ---------------------------------------------------------------------------
# 4. Measure the network on the same clean batch
# ---------------------------------------------------------------------------
class TrainingDynamics:
    def __init__(self, fixed_X, fixed_y, output_dir):
        self.fixed_X = fixed_X
        self.fixed_y = fixed_y
        self.output_dir = Path(output_dir)
        self.stream_rows = []
        self.branch_rows = []
        self.layer_rows = []
        self.saturation_rows = []
        self.checkpoint_rows = []

    def stream_hook(self, epoch, depth, name):
        # A hook reads a layer's output without changing that output.
        def hook(module, inputs, output):
            self.stream_rows.append({
                "epoch": epoch,
                "depth": depth,
                "module": name,
                "variance": variance(output),
                "stream_rms": rms(output),
            })
        return hook

    def branch_hook(self, epoch, depth, kind, name, returns_delta):
        def hook(module, inputs, output):
            stream = inputs[0].detach()
            # MixingBranch returns the addition; a full block includes the skip.
            delta = output.detach() if returns_delta else output.detach() - stream
            input_rms = rms(stream)
            branch_rms = rms(delta)
            self.branch_rows.append({
                "epoch": epoch,
                "depth": depth,
                "kind": kind,
                "module": name,
                "input_rms": input_rms,
                "branch_rms": branch_rms,
                "ratio": branch_rms / max(input_rms, EPSILON),
            })
        return hook

    def saturation_hook(self, epoch, name):
        def hook(module, inputs, output):
            row = {
                "epoch": epoch,
                "module": name,
                "activation": type(module).__name__,
                "tanh_input_abs_gt_2": None,
                "relu_zero_fraction": None,
                "relu_units_always_zero": None,
                "relu_units_always_nonzero": None,
            }
            if isinstance(module, nn.Tanh):
                large_inputs = inputs[0].detach().abs() > 2
                row["tanh_input_abs_gt_2"] = large_inputs.float().mean().item()
            else:
                # Each position/feature is one unit; compare it across images.
                values = output.detach().flatten(start_dim=1)
                zero = values == 0
                row["relu_zero_fraction"] = zero.float().mean().item()
                row["relu_units_always_zero"] = zero.all(dim=0).float().mean().item()
                row["relu_units_always_nonzero"] = (~zero).all(dim=0).float().mean().item()
            self.saturation_rows.append(row)
        return hook

    def measure(self, model, optimizer, epoch):
        print(f"  Measuring the fixed batch at epoch {epoch}...", flush=True)
        # Copy both together so copied momentum buffers refer to copied weights.
        # fork_rng prevents measurement from changing future training randomness.
        with torch.random.fork_rng(devices=rng_devices(self.fixed_X.device)):
            probe, probe_optimizer = copy.deepcopy((model, optimizer))
            probe.eval()
            probe_optimizer.zero_grad(set_to_none=True)
            handles = []
            linear_outputs = {}

            def linear_hook(name):
                def hook(module, inputs, output):
                    linear_outputs[name] = (variance(output), rms(output))
                return hook

            try:
                projection = "patch_projection" if hasattr(probe, "patch_projection") else "input_layer"
                hook = self.stream_hook(epoch, 0, projection)
                handles.append(getattr(probe, projection).register_forward_hook(hook))

                for depth, block in enumerate(probe.blocks, start=1):
                    name = f"blocks.{depth - 1}"
                    handles.append(block.register_forward_hook(
                        self.stream_hook(epoch, depth, name)
                    ))
                    # The combined change includes token and channel additions.
                    handles.append(block.register_forward_hook(
                        self.branch_hook(epoch, depth, "whole_block", name, False)
                    ))

                for depth, kind, name, module, delta_only in residual_branches(probe):
                    handles.append(module.register_forward_hook(
                        self.branch_hook(epoch, depth, kind, name, delta_only)
                    ))

                linears = []
                for name, module in probe.named_modules():
                    if isinstance(module, nn.Linear):
                        linears.append((name, module))
                        handles.append(module.register_forward_hook(linear_hook(name)))
                    if isinstance(module, (nn.Tanh, nn.ReLU)):
                        handles.append(module.register_forward_hook(
                            self.saturation_hook(epoch, name)
                        ))

                # eval() disables dropout, but gradients still work normally.
                with torch.enable_grad():
                    logits = probe(self.fixed_X)
                    loss = nn.functional.cross_entropy(logits, self.fixed_y)
                    loss.backward()

                before = {}
                gradients = {}
                for name, module in linears:
                    before[name] = module.weight.detach().clone()
                    gradient = module.weight.grad
                    gradients[name] = None if gradient is None else rms(gradient)

                # Use a real copied optimizer step, including momentum or Adam state.
                probe_optimizer.step()
                for order, (name, module) in enumerate(linears):
                    weight_rms = rms(before[name])
                    update_rms = rms(module.weight.detach() - before[name])
                    output_variance, output_rms = linear_outputs[name]
                    self.layer_rows.append({
                        "epoch": epoch,
                        "order": order,
                        "module": name,
                        "gradient_status": "missing" if gradients[name] is None else "present",
                        "gradient_rms": gradients[name],
                        "weight_rms": weight_rms,
                        "update_rms": update_rms,
                        "update_to_weight_ratio": update_rms / max(weight_rms, EPSILON),
                        "activation_variance": output_variance,
                        "activation_rms": output_rms,
                    })
                self.checkpoint_rows.append({
                    "epoch": epoch,
                    "probe_loss": loss.item(),
                    "probe_accuracy": (logits.argmax(dim=1) == self.fixed_y).float().mean().item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                })
            finally:
                # Remove hooks even if a measurement raises an error.
                for handle in handles:
                    handle.remove()
        # The copied model and optimizer are discarded; training keeps its state.
        self.save()

    def save(self):
        write_csv(self.output_dir / "activation_variance.csv", self.stream_rows)
        write_csv(self.output_dir / "branch_ratios.csv", self.branch_rows)
        write_csv(self.output_dir / "linear_metrics.csv", self.layer_rows)
        write_csv(self.output_dir / "saturation.csv", self.saturation_rows)
        write_csv(self.output_dir / "probe_checkpoints.csv", self.checkpoint_rows)


# ---------------------------------------------------------------------------
# 5. Validate and remove one residual branch at a time
# ---------------------------------------------------------------------------
def evaluate(model, loader, device, return_predictions=False):
    loss_sum = 0.0
    count = 0
    correct_parts = []
    prediction_parts = []
    # No augmentation is used, and validation does not update BatchNorm statistics.
    with torch.random.fork_rng(devices=rng_devices(device)):
        with evaluation_mode(model), torch.no_grad():
            for images, labels in loader:
                images = prepare(images.to(device))
                labels = labels.to(device)
                logits = model(images)
                loss_sum += nn.functional.cross_entropy(
                    logits, labels, reduction="sum"
                ).item()
                predictions = logits.argmax(dim=1)
                correct_parts.append((predictions == labels).cpu())
                if return_predictions:
                    prediction_parts.append(predictions.cpu())
                count += len(labels)
    correct = torch.cat(correct_parts)
    if return_predictions:
        # Exact class predictions let the ablation check verify complete restoration.
        return correct.float().mean().item(), loss_sum / count, correct, torch.cat(prediction_parts)
    return correct.float().mean().item(), loss_sum / count, correct


def ablate_branches(model, loader, device, output_dir):
    baseline, baseline_loss, baseline_correct, baseline_predictions = evaluate(
        model, loader, device, return_predictions=True
    )
    print(f"Full model validation accuracy: {baseline:.2%}", flush=True)
    targets = residual_branches(model)
    if hasattr(model.blocks[0], "token_branch"):
        # Also remove both additions together for one point per complete block.
        for depth, block in enumerate(model.blocks, start=1):
            targets.append((depth, "whole_block", f"blocks.{depth - 1}", block, False))

    rows = []
    for depth, kind, name, module, returns_delta in targets:
        def skip_branch(selected, inputs, output):
            # A MixingBranch returns only its addition, so zeroing it keeps the skip.
            # A full block includes its skip, so return its original input instead.
            return torch.zeros_like(output) if returns_delta else inputs[0]

        handle = module.register_forward_hook(skip_branch)
        try:
            accuracy, _, correct = evaluate(model, loader, device)
        finally:
            # Every trial starts from the same full model; removals never accumulate.
            handle.remove()

        difference = baseline_correct.float() - correct.float()
        paired_error = difference.std(unbiased=False).item() / len(difference) ** 0.5
        rows.append({
            "depth": depth,
            "kind": kind,
            "module": name,
            "full_accuracy": baseline,
            "ablated_accuracy": accuracy,
            "accuracy_drop_pp": (baseline - accuracy) * 100,
            "lost_correct": int((baseline_correct & ~correct).sum()),
            "gained_correct": int((~baseline_correct & correct).sum()),
            "paired_standard_error_pp": paired_error * 100,
        })
        write_csv(Path(output_dir) / "ablation.csv", rows)
        print(f"  {name}: accuracy drop {(baseline - accuracy) * 100:+.3f} points", flush=True)

    _, _, _, restored_predictions = evaluate(model, loader, device, return_predictions=True)
    if not torch.equal(baseline_predictions, restored_predictions):
        raise RuntimeError("Validation predictions changed after removing ablation hooks")
    return rows, baseline, baseline_loss


# ---------------------------------------------------------------------------
# 6. Turn the saved measurements into report figures
# ---------------------------------------------------------------------------
def make_plots(output_dir, dynamics, curve, ablations):
    import matplotlib
    matplotlib.use("Agg")        # Save figures without opening a desktop window.
    import matplotlib.pyplot as plt

    figures = Path(output_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    def save(figure, name):
        figure.tight_layout()
        figure.savefig(figures / name, dpi=180, bbox_inches="tight")
        plt.close(figure)

    def profile(rows, x_key, y_key, ylabel, filename, logarithmic=False):
        figure, axis = plt.subplots(figsize=(8, 3.8))
        for epoch in sorted({row["epoch"] for row in rows}):
            selected = [row for row in rows if row["epoch"] == epoch]
            values = [np.nan if row[y_key] is None else row[y_key] for row in selected]
            axis.plot(
                [row[x_key] for row in selected], values,
                marker=".", label=f"Epoch {epoch}",
            )
        if logarithmic:
            # Symlog displays very small values and true zeros on the same plot.
            axis.set_yscale("symlog", linthresh=1e-8)
        if y_key == "update_to_weight_ratio":
            axis.axhline(1e-3, color="gray", linestyle="--", label="Rough 1e-3 reference")
        xlabel = "Linear layer order" if x_key == "order" else "Block depth"
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.legend(ncol=2)
        axis.grid(alpha=0.2)
        save(figure, filename)

    profile(dynamics.stream_rows, "depth", "variance",
            "Residual stream variance", "activation_variance.png")
    profile(dynamics.layer_rows, "order", "gradient_rms",
            "Weight gradient RMS", "gradient_rms.png", True)
    profile(dynamics.layer_rows, "order", "update_to_weight_ratio",
            "Update RMS / weight RMS", "update_to_weight_ratio.png", True)
    branch_titles = {
        "whole_block": "Whole block",
        "token": "Token branch",
        "channel": "Channel branch",
        "branch": "Residual branch",
    }
    for kind, title in branch_titles.items():
        rows = [row for row in dynamics.branch_rows if row["kind"] == kind]
        if rows:
            filename = "branch_to_stream_ratio.png" if kind == "whole_block" else f"branch_ratio_{kind}.png"
            profile(rows, "depth", "ratio", f"{title} RMS / input RMS", filename)

    # Training uses augmented images; validation uses unchanged held-out images.
    epochs = [row["epoch"] for row in curve]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(epochs, [row["train_loss"] for row in curve], label="Augmented training")
    axes[0].plot(epochs, [row["validation_loss"] for row in curve], label="Clean validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross entropy")
    axes[0].legend(fontsize=8)
    axes[1].plot(epochs, [row["validation_accuracy"] * 100 for row in curve])
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy (%)")
    save(figure, "training_curve.png")

    figure, axis = plt.subplots(figsize=(7, 3.4))
    axis.plot(epochs, [row["learning_rate_used"] for row in curve], marker=".")
    axis.set(xlabel="Epoch", ylabel="Learning rate used")
    save(figure, "learning_rate.png")

    kinds = list(dict.fromkeys(row["kind"] for row in ablations))
    figure, axis = plt.subplots(figsize=(8, 3.8))
    for kind in kinds:
        selected = [row for row in ablations if row["kind"] == kind]
        axis.plot(
            [row["depth"] for row in selected],
            [row["accuracy_drop_pp"] for row in selected],
            marker="o", label=branch_titles[kind],
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Block depth", ylabel="Validation accuracy drop (percentage points)")
    axis.legend()
    save(figure, "ablation_accuracy_drop.png")

    # Compare each ablation against the matching branch's final magnitude ratio.
    final_epoch = max(row["epoch"] for row in dynamics.branch_rows)
    ratios = {}
    for row in dynamics.branch_rows:
        if row["epoch"] == final_epoch:
            ratios[(row["depth"], row["kind"])] = row["ratio"]
    figure, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 3.8), squeeze=False)
    for axis, kind in zip(axes[0], kinds):
        selected = [row for row in ablations if row["kind"] == kind]
        for row in selected:
            x = ratios[(row["depth"], kind)]
            y = row["accuracy_drop_pp"]
            axis.scatter(x, y, color="tab:blue")
            axis.annotate(str(row["depth"]), (x, y), fontsize=8)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=branch_titles[kind], xlabel="Final branch / input RMS")
    axes[0, 0].set_ylabel("Accuracy drop (percentage points)")
    save(figure, "branch_ratio_vs_ablation.png")

    # Save applicable activation diagnostics only; GELU does not use these tests.
    metrics = {
        "tanh_input_abs_gt_2": "Fraction of tanh inputs with magnitude above 2",
        "relu_zero_fraction": "Fraction of ReLU outputs equal to zero",
        "relu_units_always_zero": "Fraction of units zero for the whole batch",
        "relu_units_always_nonzero": "Fraction of units nonzero for the whole batch",
    }
    for metric, label in metrics.items():
        rows = []
        for epoch in sorted({row["epoch"] for row in dynamics.saturation_rows}):
            selected = [row for row in dynamics.saturation_rows
                        if row["epoch"] == epoch and row[metric] is not None]
            for order, row in enumerate(selected):
                rows.append(dict(row, order=order))
        if rows:
            profile(rows, "order", metric, label, f"saturation_{metric}.png")


def write_summary(output_dir, dynamics, ablations, metadata):
    # Save measured results; interpret them after the run.
    final_epoch = max(row["epoch"] for row in dynamics.stream_rows)
    streams = [row for row in dynamics.stream_rows if row["epoch"] == final_epoch]
    layers = [row for row in dynamics.layer_rows
              if row["epoch"] == final_epoch and row["gradient_rms"] is not None]
    smallest_gradient = min(layers, key=lambda row: row["gradient_rms"])
    largest_gradient = max(layers, key=lambda row: row["gradient_rms"])
    text = [
        "# ImageNet analysis results",
        f"Validation accuracy: {metadata['validation_accuracy']:.2%} on "
        f"{metadata['validation_count']:,} held-out training images.",
        "## Measured observations",
        f"At epoch {final_epoch}, stream variance was {streams[0]['variance']:.6g} "
        f"after projection and {streams[-1]['variance']:.6g} after the last block.",
        f"Smallest final gradient RMS: {smallest_gradient['module']} = "
        f"{smallest_gradient['gradient_rms']:.6g}.",
        f"Largest final gradient RMS: {largest_gradient['module']} = "
        f"{largest_gradient['gradient_rms']:.6g}.",
    ]
    for kind in dict.fromkeys(row["kind"] for row in ablations):
        rows = [row for row in ablations if row["kind"] == kind]
        largest = max(rows, key=lambda row: row["accuracy_drop_pp"])
        smallest = min(rows, key=lambda row: abs(row["accuracy_drop_pp"]))
        text.extend([
            f"For {kind.replace('_', ' ')} removals, the largest drop was at depth "
            f"{largest['depth']}: {largest['accuracy_drop_pp']:+.3f} percentage points.",
            f"The smallest absolute drop was at depth {smallest['depth']}: "
            f"{smallest['accuracy_drop_pp']:+.3f} percentage points.",
        ])
    text.append(f"Saturation: {metadata['saturation_note']}")
    (Path(output_dir) / "analysis_summary.md").write_text("\n\n".join(text), encoding="utf-8")


# ---------------------------------------------------------------------------
# 7. Save enough context to explain and rerun these measurements
# ---------------------------------------------------------------------------
def save_sources(output_dir):
    source_dir = Path(output_dir) / "source"
    files = [Path("analysis.py")]
    for filename in ("__init__.py", "model.py", "train.py", "patch_mlp.py", "augmentation.py"):
        files.append(Path("src") / "residual_mlp_baseline" / filename)
    hashes = {}
    for relative in files:
        original = ROOT / relative
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        hashes[str(relative)] = hashlib.sha256(original.read_bytes()).hexdigest()
    return hashes


def saturation_note(model):
    # Inspect the actual modules instead of assuming a settings flag is active.
    applies = any(isinstance(module, (nn.Tanh, nn.ReLU)) for module in model.modules())
    if applies:
        return "See saturation.csv for tanh/ReLU measurements."
    return "Not applicable: no tanh or ReLU layers."


def package_results(output_dir):
    # Include measurements, figures, splits, and source code, but no images or weights.
    output_dir = Path(output_dir)
    archive = output_dir.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(f"Results archive already exists: {archive}")
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as file:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                file.write(path, path.relative_to(output_dir.parent))
    return archive


def train_one_epoch(model, optimizer, loader, settings, device):
    model.train()
    total_loss = 0.0
    total_images = 0
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for images, labels in loader:
        # Only this batch becomes float; the complete cache stays uint8 on disk.
        images = prepare(images.to(device))
        labels = labels.to(device)
        images = augment_batch(images, settings.get("augmentation", {}))
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total_images += len(labels)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return total_loss / total_images, time.perf_counter() - started


# ---------------------------------------------------------------------------
# 8. Run training, measure the checkpoints, then run the ablation study
# ---------------------------------------------------------------------------
def main():
    # Check plotting support before spending time training on the cluster.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = copy.deepcopy(TASK_SETTINGS[TASK])
    images, labels = load_cache()
    archive_path = OUTPUT_DIR.with_suffix(".zip")
    if (OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir())) or archive_path.exists():
        raise FileExistsError(
            f"Results already exist for {RUN_NAME}. Change RUN_NAME for another run."
        )
    train_indices, validation_indices = split_indices(labels)
    if len(train_indices) < PROBE_BATCH_SIZE:
        raise ValueError("The training split is smaller than PROBE_BATCH_SIZE")
    generator = np.random.default_rng(SEED + 1)
    probe_indices = generator.choice(train_indices, size=PROBE_BATCH_SIZE, replace=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "split_indices.npz",
        train=train_indices, validation=validation_indices, probe=probe_indices,
    )
    source_hashes = save_sources(OUTPUT_DIR)
    torch.manual_seed(SEED)
    model = build_model(TASK).to(DEVICE)
    optimizer = make_optimizer(model, settings)
    scheduler = make_scheduler(optimizer, settings)

    # Separate generators keep validation from changing the training shuffle.
    train_loader = DataLoader(
        CachedImages(images, labels, train_indices),
        batch_size=settings["batch_size"], shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    validation_loader = DataLoader(
        CachedImages(images, labels, validation_indices),
        batch_size=settings["batch_size"], shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(SEED + 2),
    )
    fixed_X = torch.from_numpy(np.array(images[probe_indices], copy=True)).to(DEVICE)
    fixed_X = prepare(fixed_X)
    fixed_y = torch.as_tensor(np.array(labels[probe_indices], copy=True), device=DEVICE).long()
    dynamics = TrainingDynamics(fixed_X, fixed_y, OUTPUT_DIR)
    epochs = settings["epochs"]
    checkpoints = sorted({0, max(1, epochs // 10), max(1, epochs // 2), epochs})

    metadata = {
        "status": "running",
        "task": TASK,
        "settings": settings,
        "all_task_settings": copy.deepcopy(TASK_SETTINGS),
        "seed": SEED,
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name() if torch.device(DEVICE).type == "cuda" else None,
        "torch_version": torch.__version__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "validation_per_class": VALIDATION_PER_CLASS,
        "probe_batch_size": PROBE_BATCH_SIZE,
        "checkpoints": checkpoints,
        "source_sha256": source_hashes,
        "saturation_note": saturation_note(model),
        "split_note": "Validation is held out from training; official test files are never loaded.",
        "probe_note": (
            "Fixed clean training batch, eval mode, actual step on a copied model and optimizer. "
            "Live weights, gradients, optimizer, module modes, and global RNG remain unchanged."
        ),
        "update_rate_note": (
            "Probes run after the epoch and before scheduler.step(), using the completed "
            "epoch's rate. Epoch 0 uses the initial rate with no accumulated optimizer state."
        ),
        "comparison_note": (
            "This retrains the final recipe on a training-validation split. Its score is "
            "not a reproduction of the full-data harness score or exact stochastic trajectory."
        ),
    }
    metadata_path = OUTPUT_DIR / "run_config.json"
    write_json(metadata_path, metadata)
    print(f"Device: {DEVICE}; training: {len(train_indices):,}; validation: {len(validation_indices):,}", flush=True)
    print(f"Measurement epochs: {checkpoints}; output: {OUTPUT_DIR}", flush=True)
    print(f"Architecture: {len(model.blocks)} blocks; {metadata['parameters']:,} parameters", flush=True)

    started = time.perf_counter()
    dynamics.measure(model, optimizer, 0)
    curve = []
    for epoch in range(1, epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_seconds = train_one_epoch(
            model, optimizer, train_loader, settings, DEVICE
        )
        accuracy, validation_loss, _ = evaluate(model, validation_loader, DEVICE)
        curve.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": accuracy,
            "learning_rate_used": learning_rate,
            "train_seconds": train_seconds,
        })
        write_csv(OUTPUT_DIR / "training_curve.csv", curve)
        print(
            f"epoch {epoch}: train loss {train_loss:.4f}; val loss {validation_loss:.4f}; "
            f"val accuracy {accuracy:.2%}; lr {learning_rate:.6g}",
            flush=True,
        )
        if epoch in checkpoints:
            dynamics.measure(model, optimizer, epoch)
        # Change the learning rate only after measuring the completed epoch.
        if scheduler is not None:
            scheduler.step()

    ablations, accuracy, validation_loss = ablate_branches(
        model, validation_loader, DEVICE, OUTPUT_DIR
    )
    metadata.update(
        validation_accuracy=accuracy,
        validation_loss=validation_loss,
        training_minutes=sum(row["train_seconds"] for row in curve) / 60,
        saturation_applicable=bool(dynamics.saturation_rows),
    )
    make_plots(OUTPUT_DIR, dynamics, curve, ablations)
    write_summary(OUTPUT_DIR, dynamics, ablations, metadata)
    metadata.update(status="complete", total_analysis_minutes=(time.perf_counter() - started) / 60)
    write_json(metadata_path, metadata)
    archive = package_results(OUTPUT_DIR)
    print(f"Analysis complete. Send back this results archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
