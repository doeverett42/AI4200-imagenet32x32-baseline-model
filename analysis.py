#ImageNet training dynamics and residual branch ablation

import copy
import csv
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

#Use the source files beside this script instead of an older installed package
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from residual_mlp_baseline.model import TASK_SETTINGS
from residual_mlp_baseline.train import build_model, prepare
from residual_mlp_baseline.augmentation import augment_batch


#Settings
TASK = "imagenet32"
SEED = 0
VALIDATION_PER_CLASS = 5
PROBE_BATCH_SIZE = 128
DATA_DIR = ROOT / "data" / TASK
RUN_NAME = "imagenet32_final"
OUTPUT_DIR = ROOT / "analysis_results" / RUN_NAME
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPSILON = 1e-12


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(tensor):
    return tensor.detach().float().square().mean().sqrt().item()


def variance(tensor):
    #Population variance over every batch, patch, and feature entry
    return tensor.detach().float().var(unbiased=False).item()


@contextmanager
def evaluation_mode(model):
    #Restore every module's previous mode, including any mixed train/eval modes
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, was_training in states:
            module.training = was_training


def rng_devices(device):
    device = torch.device(device)
    return [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []


class CachedImages(Dataset):
    #Keep the full image array on disk and copy only one selected uint8 image
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
    #Only the training split is divided; official test files are never opened
    generator = np.random.default_rng(seed)
    validation = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) <= per_class:
            raise ValueError(f"Class {label} needs more than {per_class} images")
        validation.extend(generator.permutation(indices)[:per_class])
    validation = np.sort(np.asarray(validation, dtype=np.int64))
    mask = np.ones(len(labels), dtype=bool)
    mask[validation] = False
    return np.flatnonzero(mask), validation


def make_optimizer(model, settings):
    if settings["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    if settings["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=settings["lr"], momentum=settings["momentum"], weight_decay=settings["weight_decay"])
    raise ValueError(f"Unknown optimizer: {settings['optimizer']}")


def make_scheduler(optimizer, settings):
    name = settings.get("scheduler", "none")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"], eta_min=0.00001)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, settings["epochs"] // 4), gamma=0.5)
    if name != "none":
        raise ValueError(f"Unknown scheduler: {name}")
    return None


def residual_branches(model):
    #Patch blocks have two separate additions; flat blocks have one addition
    branches = []
    for depth, block in enumerate(model.blocks, start=1):
        if hasattr(block, "token_branch"):
            for kind in ("token", "channel"):
                name = f"blocks.{depth - 1}.{kind}_branch"
                branches.append((depth, kind, name, getattr(block, f"{kind}_branch"), True))
        else:
            branches.append((depth, "branch", f"blocks.{depth - 1}", block, False))
    return branches


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

    def measure(self, model, optimizer, epoch):
        device = self.fixed_X.device
        #Copy the pair together so momentum buffers still refer to copied weights
        with torch.random.fork_rng(devices=rng_devices(device)):
            probe, probe_optimizer = copy.deepcopy((model, optimizer))
            probe.eval()
            probe_optimizer.zero_grad(set_to_none=True)
            handles = []
            linear_outputs = {}

            def stream_hook(depth, name):
                def hook(module, inputs, output):
                    self.stream_rows.append({"epoch": epoch, "depth": depth, "module": name, "variance": variance(output), "stream_rms": rms(output)})
                return hook

            def branch_hook(depth, kind, name, returns_delta):
                def hook(module, inputs, output):
                    stream = inputs[0].detach()
                    delta = output.detach() if returns_delta else output.detach() - stream
                    input_rms, branch_rms = rms(stream), rms(delta)
                    self.branch_rows.append({"epoch": epoch, "depth": depth, "kind": kind, "module": name, "input_rms": input_rms, "branch_rms": branch_rms, "ratio": branch_rms / max(input_rms, EPSILON)})
                return hook

            def linear_hook(name):
                def hook(module, inputs, output):
                    linear_outputs[name] = (variance(output), rms(output))
                return hook

            def saturation_hook(name):
                def hook(module, inputs, output):
                    row = {"epoch": epoch, "module": name, "activation": type(module).__name__, "tanh_input_abs_gt_2": None, "relu_zero_fraction": None, "relu_units_always_zero": None, "relu_units_always_nonzero": None}
                    if isinstance(module, nn.Tanh):
                        row["tanh_input_abs_gt_2"] = (inputs[0].detach().abs() > 2).float().mean().item()
                    else:
                        values = output.detach().reshape(-1, output.shape[-1])
                        row["relu_zero_fraction"] = (values == 0).float().mean().item()
                        row["relu_units_always_zero"] = (values == 0).all(dim=0).float().mean().item()
                        row["relu_units_always_nonzero"] = (values != 0).all(dim=0).float().mean().item()
                    self.saturation_rows.append(row)
                return hook

            try:
                projection_name = "patch_projection" if hasattr(probe, "patch_projection") else "input_layer"
                handles.append(getattr(probe, projection_name).register_forward_hook(stream_hook(0, projection_name)))
                for depth, block in enumerate(probe.blocks, start=1):
                    handles.append(block.register_forward_hook(stream_hook(depth, f"blocks.{depth - 1}")))
                    handles.append(block.register_forward_hook(branch_hook(depth, "whole_block", f"blocks.{depth - 1}", False)))
                for depth, kind, name, module, returns_delta in residual_branches(probe):
                    handles.append(module.register_forward_hook(branch_hook(depth, kind, name, returns_delta)))
                linears = [(name, module) for name, module in probe.named_modules() if isinstance(module, nn.Linear)]
                for name, module in linears:
                    handles.append(module.register_forward_hook(linear_hook(name)))
                for name, module in probe.named_modules():
                    if isinstance(module, (nn.Tanh, nn.ReLU)):
                        handles.append(module.register_forward_hook(saturation_hook(name)))

                # Eval mode disables dropout; it does not disable differentiation.
                logits = probe(self.fixed_X)
                loss = nn.functional.cross_entropy(logits, self.fixed_y)
                loss.backward()
                before = {name: module.weight.detach().clone() for name, module in linears}
                gradients = {name: None if module.weight.grad is None else rms(module.weight.grad) for name, module in linears}
                probe_optimizer.step()
                for order, (name, module) in enumerate(linears):
                    weight_rms = rms(before[name])
                    update_rms = rms(module.weight.detach() - before[name])
                    output_variance, output_rms = linear_outputs[name]
                    self.layer_rows.append({"epoch": epoch, "order": order, "module": name, "gradient_status": "missing" if gradients[name] is None else "present", "gradient_rms": gradients[name], "weight_rms": weight_rms, "update_rms": update_rms, "update_to_weight_ratio": update_rms / max(weight_rms, EPSILON), "activation_variance": output_variance, "activation_rms": output_rms})
                self.checkpoint_rows.append({"epoch": epoch, "probe_loss": loss.item(), "probe_accuracy": (logits.argmax(dim=1) == self.fixed_y).float().mean().item(), "learning_rate": optimizer.param_groups[0]["lr"]})
            finally:
                for handle in handles:
                    handle.remove()
        self.save()

    def save(self):
        write_csv(self.output_dir / "activation_variance.csv", self.stream_rows)
        write_csv(self.output_dir / "branch_ratios.csv", self.branch_rows)
        write_csv(self.output_dir / "linear_metrics.csv", self.layer_rows)
        write_csv(self.output_dir / "saturation.csv", self.saturation_rows)
        write_csv(self.output_dir / "probe_checkpoints.csv", self.checkpoint_rows)


def evaluate(model, loader, device):
    loss_sum, count, correct_parts = 0.0, 0, []
    #Validation must not change training modes, running statistics, or RNG state
    with torch.random.fork_rng(devices=rng_devices(device)), evaluation_mode(model), torch.no_grad():
        for images, labels in loader:
            images, labels = prepare(images.to(device)), labels.to(device)
            logits = model(images)
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            correct_parts.append((logits.argmax(dim=1) == labels).cpu())
            count += len(labels)
    correct = torch.cat(correct_parts)
    return correct.float().mean().item(), loss_sum / count, correct


def ablate_branches(model, loader, device, output_dir):
    baseline, baseline_loss, baseline_correct = evaluate(model, loader, device)
    print(f"Full model validation accuracy: {baseline:.2%}", flush=True)
    #Include whole-block removals as an additional one-point-per-block view
    targets = residual_branches(model)
    if targets and targets[0][-1]:
        targets += [(depth, "whole_block", f"blocks.{depth - 1}", block, False) for depth, block in enumerate(model.blocks, start=1)]
    rows = []
    for depth, kind, name, module, returns_delta in targets:
        def skip_branch(selected, inputs, output):
            #A MixingBranch returns only its correction, so zero that correction
            return torch.zeros_like(output) if returns_delta else inputs[0]
        handle = module.register_forward_hook(skip_branch)
        try:
            accuracy, _, correct = evaluate(model, loader, device)
        finally:
            handle.remove()
        difference = baseline_correct.float() - correct.float()
        rows.append({"depth": depth, "kind": kind, "module": name, "full_accuracy": baseline, "ablated_accuracy": accuracy, "accuracy_drop_pp": (baseline - accuracy) * 100, "lost_correct": int((baseline_correct & ~correct).sum()), "gained_correct": int((~baseline_correct & correct).sum()), "paired_standard_error_pp": difference.std(unbiased=False).item() / len(difference) ** 0.5 * 100})
        write_csv(Path(output_dir) / "ablation.csv", rows)
        print(f"  {name}: drop {(baseline - accuracy) * 100:+.3f} percentage points", flush=True)
    restored, _, restored_correct = evaluate(model, loader, device)
    if not torch.equal(baseline_correct, restored_correct):
        raise RuntimeError("Full model predictions changed after removing ablation hooks")
    return rows, baseline, baseline_loss


def make_plots(output_dir, dynamics, curve, ablations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = Path(output_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(figure, name):
        figure.tight_layout()
        figure.savefig(figures / name, dpi=180, bbox_inches="tight")
        plt.close(figure)

    def profile(rows, x_key, y_key, ylabel, name, logarithmic=False):
        figure, axis = plt.subplots(figsize=(8, 3.8))
        for epoch in sorted({row["epoch"] for row in rows}):
            selected = [row for row in rows if row["epoch"] == epoch]
            values = [np.nan if row[y_key] is None else row[y_key] for row in selected]
            axis.plot([row[x_key] for row in selected], values, marker=".", label=f"Epoch {epoch}")
        if logarithmic:
            # Symlog preserves true zeros instead of hiding vanishing gradients.
            axis.set_yscale("symlog", linthresh=1e-8)
        if y_key == "update_to_weight_ratio":
            axis.axhline(1e-3, color="gray", linestyle="--", label="Rough 1e-3 reference")
        axis.set(xlabel="Linear layer order" if x_key == "order" else "Block depth", ylabel=ylabel)
        axis.legend(ncol=2)
        axis.grid(alpha=0.2)
        save(figure, name)

    profile(dynamics.stream_rows, "depth", "variance", "Residual stream variance", "activation_variance.png")
    profile(dynamics.layer_rows, "order", "gradient_rms", "Weight gradient RMS", "gradient_rms.png", True)
    profile(dynamics.layer_rows, "order", "update_to_weight_ratio", "Update RMS / weight RMS", "update_to_weight_ratio.png", True)
    for kind, title in (("whole_block", "Whole block"), ("token", "Token branch"), ("channel", "Channel branch"), ("branch", "Residual branch")):
        rows = [row for row in dynamics.branch_rows if row["kind"] == kind]
        if rows:
            profile(rows, "depth", "ratio", f"{title} RMS / input RMS", "branch_to_stream_ratio.png" if kind == "whole_block" else f"branch_ratio_{kind}.png")

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot([row["epoch"] for row in curve], [row["train_loss"] for row in curve], label="Training with augmentation")
    axes[0].plot([row["epoch"] for row in curve], [row["validation_loss"] for row in curve], label="Clean validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross entropy")
    axes[0].legend(fontsize=8)
    axes[1].plot([row["epoch"] for row in curve], [row["validation_accuracy"] * 100 for row in curve])
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy (%)")
    save(figure, "training_curve.png")

    figure, axis = plt.subplots(figsize=(8, 3.8))
    for kind in dict.fromkeys(row["kind"] for row in ablations):
        selected = [row for row in ablations if row["kind"] == kind]
        axis.plot([row["depth"] for row in selected], [row["accuracy_drop_pp"] for row in selected], marker="o", label=kind.replace("_", " "))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Block depth", ylabel="Validation accuracy drop (percentage points)")
    axis.legend()
    save(figure, "ablation_accuracy_drop.png")

    final_epoch = max(row["epoch"] for row in dynamics.branch_rows)
    ratios = {(row["depth"], row["kind"]): row["ratio"] for row in dynamics.branch_rows if row["epoch"] == final_epoch}
    figure, axis = plt.subplots(figsize=(7, 4))
    for kind in dict.fromkeys(row["kind"] for row in ablations if row["kind"] != "whole_block"):
        selected = [row for row in ablations if row["kind"] == kind]
        x = [ratios[(row["depth"], kind)] for row in selected]
        y = [row["accuracy_drop_pp"] for row in selected]
        axis.scatter(x, y, label=kind)
        for row, xpos, ypos in zip(selected, x, y):
            axis.annotate(str(row["depth"]), (xpos, ypos), fontsize=8)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Final branch RMS / input RMS", ylabel="Accuracy drop (percentage points)")
    axis.legend()
    save(figure, "branch_ratio_vs_ablation.png")
    if dynamics.saturation_rows:
        metric = "tanh_input_abs_gt_2" if any(row["activation"] == "Tanh" for row in dynamics.saturation_rows) else "relu_units_always_zero"
        rows = [dict(row, order=index) for epoch in sorted({row["epoch"] for row in dynamics.saturation_rows}) for index, row in enumerate(row for row in dynamics.saturation_rows if row["epoch"] == epoch and row[metric] is not None)]
        profile(rows, "order", metric, "Fraction of activation units", "saturation.png")


def write_summary(output_dir, dynamics, ablations, metadata):
    final_epoch = max(row["epoch"] for row in dynamics.stream_rows)
    streams = [row for row in dynamics.stream_rows if row["epoch"] == final_epoch]
    layers = [row for row in dynamics.layer_rows if row["epoch"] == final_epoch and row["gradient_rms"] is not None]
    individual = [row for row in ablations if row["kind"] != "whole_block"]
    largest = max(individual, key=lambda row: row["accuracy_drop_pp"])
    smallest = min(individual, key=lambda row: abs(row["accuracy_drop_pp"]))
    gradient_min = min(layers, key=lambda row: row["gradient_rms"])
    gradient_max = max(layers, key=lambda row: row["gradient_rms"])
    text = ["# ImageNet analysis results", "", f"Validation accuracy: {metadata['validation_accuracy']:.2%} on {metadata['validation_count']:,} held-out training images.", "This is a separate analysis run with a training-validation split, not the historical harness test score.", "", "## Measured observations", f"At epoch {final_epoch}, residual stream variance changed from {streams[0]['variance']:.6g} after projection to {streams[-1]['variance']:.6g} after the last block.", f"Smallest final gradient RMS: {gradient_min['module']} = {gradient_min['gradient_rms']:.6g}.", f"Largest final gradient RMS: {gradient_max['module']} = {gradient_max['gradient_rms']:.6g}.", f"Largest individual branch ablation drop: {largest['module']} = {largest['accuracy_drop_pp']:+.3f} percentage points.", f"Individual branch with the smallest absolute measured drop: {smallest['module']} = {smallest['accuracy_drop_pp']:+.3f} percentage points.", "", "## How to interpret the figures", "Compare all checkpoints before calling gradients vanishing or exploding; absolute RMS also depends on loss scaling and layer shape.", "LayerNorm controls branch inputs but does not force the residual stream variance to stay at one.", "Compare branch RMS and input RMS alongside their ratio. A falling ratio can reflect a growing input stream.", "The update ratios come from one actual optimizer step on a discarded copy in eval mode, retaining the checkpoint's optimizer state. They are not the last stochastic training update.", "A ratio around 1e-3 is a rough reference, not a target every layer must meet.", "Negative ablation drops mean removal improved measured validation accuracy. Small drops do not prove branches are useless or that several branches can be removed together.", "Compare corresponding token/channel points in branch_ratio_vs_ablation.png; a large correction need not be important to the final prediction.", "", "## Saturation", metadata["saturation_note"], "", "## Reporting", "Use the CSVs for exact values and the PNGs for report figures. Keep the full-model validation baseline fixed for every ablation.", "Training loss uses augmented images; validation loss uses clean images, so their gap is not a pure measurement of overfitting."]
    (Path(output_dir) / "analysis_summary.md").write_text("\n\n".join(text), encoding="utf-8")


def main():
    #Fail before training if plotting dependencies or the existing cache are missing
    import matplotlib
    settings = copy.deepcopy(TASK_SETTINGS[TASK])
    for filename in ("X_train.npy", "y_train.npy"):
        if not (DATA_DIR / filename).is_file():
            raise FileNotFoundError(f"Missing {DATA_DIR / filename}. Use the harness dataset cell first, or edit DATA_DIR above.")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"{OUTPUT_DIR} already contains results. Set RUN_NAME to a new name before rerunning; existing results are preserved.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = np.load(DATA_DIR / "X_train.npy", mmap_mode="r")
    labels = np.load(DATA_DIR / "y_train.npy", mmap_mode="r")
    if images.dtype != np.uint8 or images.shape[1:] != (3, 32, 32) or len(images) != len(labels):
        raise ValueError("Expected the harness's uint8 ImageNet NCHW training cache")
    if len(np.unique(labels)) != 1000:
        raise ValueError("Expected 1000 ImageNet classes")
    train_indices, validation_indices = split_indices(labels)
    probe_indices = np.random.default_rng(SEED + 1).choice(train_indices, size=PROBE_BATCH_SIZE, replace=False)
    np.savez(OUTPUT_DIR / "split_indices.npz", train=train_indices, validation=validation_indices, probe=probe_indices)
    torch.manual_seed(SEED)
    model = build_model(TASK).to(DEVICE)
    optimizer = make_optimizer(model, settings)
    scheduler = make_scheduler(optimizer, settings)
    train_loader = DataLoader(CachedImages(images, labels, train_indices), batch_size=settings["batch_size"], shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(SEED))
    validation_loader = DataLoader(CachedImages(images, labels, validation_indices), batch_size=settings["batch_size"], shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(SEED + 2))
    fixed_X = prepare(torch.from_numpy(np.array(images[probe_indices], copy=True)).to(DEVICE))
    fixed_y = torch.as_tensor(np.array(labels[probe_indices], copy=True), device=DEVICE).long()
    dynamics = TrainingDynamics(fixed_X, fixed_y, OUTPUT_DIR)
    epochs = settings["epochs"]
    checkpoints = sorted({0, max(1, epochs // 10), max(1, epochs // 2), epochs})
    source_hashes = {name: hashlib.sha256((ROOT / "src" / "residual_mlp_baseline" / name).read_bytes()).hexdigest() for name in ("model.py", "patch_mlp.py", "train.py", "augmentation.py")}
    metadata = {"status": "running", "task": TASK, "settings": settings, "seed": SEED, "device": DEVICE, "gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else None, "torch_version": torch.__version__, "parameters": sum(parameter.numel() for parameter in model.parameters()), "train_count": len(train_indices), "validation_count": len(validation_indices), "validation_per_class": VALIDATION_PER_CLASS, "probe_batch_size": PROBE_BATCH_SIZE, "checkpoints": checkpoints, "source_sha256": source_hashes, "saturation_note": "GELU has no tanh |input| > 2 or ReLU zero-unit diagnostic. These requested saturation statistics are not applicable to this final model." if settings.get("activation") == "gelu" and settings.get("block_type", "mlp") == "mlp" else "See saturation.csv if the model contains Tanh or ReLU modules; inactive on this batch does not establish a permanently dead neuron.", "split_note": "Fixed stratified validation is carved only from the cached training split. Official test images are never loaded.", "probe_note": "Eval-mode fixed clean training batch; actual optimizer step on a copied model-optimizer pair; live state and RNG are unchanged."}
    metadata_path = OUTPUT_DIR / "run_config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Device: {DEVICE}; training images: {len(train_indices):,}; validation images: {len(validation_indices):,}", flush=True)
    print(f"Measurement epochs: {checkpoints}; output: {OUTPUT_DIR}", flush=True)
    started = time.perf_counter()
    dynamics.measure(model, optimizer, 0)
    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        total_loss = 0.0
        learning_rate = optimizer.param_groups[0]["lr"]
        for batch_X, batch_y in train_loader:
            # Normalize only the current batch, avoiding full-dataset float copies.
            batch_X, batch_y = prepare(batch_X.to(DEVICE)), batch_y.to(DEVICE)
            batch_X = augment_batch(batch_X, settings.get("augmentation", {}))
            loss = nn.functional.cross_entropy(model(batch_X), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - epoch_start
        if scheduler is not None:
            scheduler.step()
        accuracy, validation_loss, _ = evaluate(model, validation_loader, DEVICE)
        curve.append({"epoch": epoch, "train_loss": total_loss / len(train_indices), "validation_loss": validation_loss, "validation_accuracy": accuracy, "learning_rate_used": learning_rate, "train_seconds": train_seconds})
        write_csv(OUTPUT_DIR / "training_curve.csv", curve)
        print(f"epoch {epoch}: train loss {curve[-1]['train_loss']:.4f}; validation accuracy {accuracy:.2%}", flush=True)
        if epoch in checkpoints:
            dynamics.measure(model, optimizer, epoch)
    ablations, accuracy, validation_loss = ablate_branches(model, validation_loader, DEVICE, OUTPUT_DIR)
    metadata.update(validation_accuracy=accuracy, validation_loss=validation_loss, training_minutes=sum(row["train_seconds"] for row in curve) / 60, saturation_applicable=bool(dynamics.saturation_rows))
    make_plots(OUTPUT_DIR, dynamics, curve, ablations)
    write_summary(OUTPUT_DIR, dynamics, ablations, metadata)
    metadata.update(status="complete", total_analysis_minutes=(time.perf_counter() - started) / 60)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Analysis complete. Figures and CSVs saved to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()