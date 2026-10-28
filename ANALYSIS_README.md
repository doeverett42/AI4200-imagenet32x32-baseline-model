# ImageNet training dynamics and ablation

`analysis.py` trains the current ImageNet model from scratch, measures its internal behavior, and tests how much each residual branch matters. It reads the model and training choices from `TASK_SETTINGS` in `src/residual_mlp_baseline/model.py`.

This is a separate analysis run. It does not change the graded `train()` or `predict()` functions and does not train CIFAR again. CIFAR's settings are included in the saved configuration so the report has a record of all three final recipes.

## What is being analyzed

The current ImageNet recipe has 12 patch blocks, 128 features per patch, GELU, LayerNorm, and a linear classifier. It uses SGD with momentum 0.9, a starting learning rate of 0.05, cosine decay, 20 epochs, and batch size 256. Training augmentation uses horizontal flips, RGB jitter 0.05, and pixel noise 0.02.

These values are not copied into a second settings dictionary in the analysis file. `build_model()` creates the model from your current source, and the analysis reads the same task settings. If you change that source before running, the analysis follows the new settings. The saved configuration and source snapshots show exactly what was used.

## Run it

Run this from the repository folder on the cluster, using the environment that runs your harness:

```bash
python analysis.py
```

In Jupyter, `%run analysis.py` also runs the file. Restart the kernel first if you changed or reinstalled the package; already imported modules can otherwise remain stale.

The script expects the existing harness cache:

```text
data/imagenet32/X_train.npy
data/imagenet32/y_train.npy
```

If your cache is somewhere else, change only `DATA_DIR` near the top of `analysis.py`. It never downloads data or opens the official test files. Besides PyTorch, it needs NumPy and Matplotlib. If those are missing in the cluster environment, install them with `python -m pip install numpy matplotlib`.

The script uses CUDA when available and prints the selected device before training. It reads the image cache as uint8 and converts one batch at a time. This avoids creating a full floating-point copy of ImageNet in RAM.

## What happens in order

1. Set aside five images from each class. This gives 5,000 validation images taken only from the training split. Their indices are fixed by the seed and saved.
2. Select 128 different images from the remaining training data. These are the fixed measurement batch. They are reused without augmentation.
3. Build a new model and optimizer. No saved weights are loaded.
4. Measure the model before training, then after epochs 2, 10, and 20. The checkpoint positions adjust automatically if the epoch count changes.
5. Print training loss, clean validation loss, validation accuracy, and learning rate after every epoch.
6. After training, evaluate each branch removal separately on the same validation split. Restore the full model after each removal.
7. Save the measurements, figures, configuration, source snapshots, and a ZIP file to send back for the report.

The analysis uses slightly fewer training examples and its own fixed split. Its validation accuracy is a new result; it is not expected to reproduce the logged 25.42% harness score exactly.

## How the measurements work

### Activation variance

Variance measures how spread out the stream values are around their mean. The script measures all entries in the batch, patches, and features together. It records the stream after the patch projection and after every complete mixing block.

Depth 0 is the projection. Depths 1 through 12 are the mixing blocks. Python names such as `blocks.0` still start at zero, so `blocks.0` means depth 1.

Look for growth, shrinkage, or abrupt changes with depth. LayerNorm normalizes branch inputs; it does not force the accumulated residual stream to have variance one.

### Branch to stream ratio

RMS means square the values, average them, and take the square root. It describes a signal's typical size.

The branch ratio is `RMS(branch correction) / RMS(branch input)`. The script records token mixing, channel mixing, and the complete block correction separately. The channel branch's input already includes the token branch's addition.

A ratio near zero means the correction is small relative to that input. Check both RMS values before deciding why: the numerator may be small, or the input stream may have grown. This measures size, not usefulness by itself.

### RMS gradient

The script calculates cross-entropy loss on the fixed batch and measures the RMS gradient of every linear layer's weight matrix. There are 50 such matrices in the current ImageNet model: one patch projection, four per mixing block, and one classifier.

Eval mode does not disable gradients. It disables dropout and uses inference behavior for normalization, making measurements more comparable between checkpoints.

### Update to weight ratio

The script copies the model and optimizer together. It calculates the fixed-batch gradient on that copy, takes one actual optimizer step, and measures `RMS(weight change) / RMS(weights before the step)`.

Copying the optimizer preserves momentum. Simply multiplying the gradient by the learning rate would miss momentum's contribution. The copy is discarded, leaving the live weights, optimizer, gradients, module modes, and global random state unchanged.

Measurements happen before the scheduler advances, so each trained checkpoint uses the learning rate from the epoch just completed. Epoch 0 uses the initial learning rate without accumulated momentum. With the current cosine schedule, later updates should become smaller partly because the learning rate decreases. The `1e-3` line is a rough comparison, not a target every layer must hit.

### Saturation

The current ImageNet model uses GELU. The assignment's tanh and ReLU statistics are therefore marked not applicable rather than filled with misleading zeros.

If a later model contains tanh, the script measures the fraction of inputs whose magnitude exceeds two. For ReLU, it records individual zero outputs and units that are always zero or always nonzero across the fixed batch. Each non-batch position/feature is treated as a unit. Inactive on this batch does not mean permanently inactive.

## How the ablation works

A patch block makes two additions: token mixing and then channel mixing. The current model therefore has 24 individual residual branches across 12 blocks.

The script first measures the full validation accuracy. It then removes one individual branch at a time by returning a zero correction. The surrounding skip addition still passes the input through, and the other branch remains active. It also tests removing each complete block by returning that block's input. This gives 12 whole-block points against depth in addition to the 24 individual-branch results.

No ablated model is retrained. No removals accumulate. Hooks are removed even if an evaluation raises an error, and a final check verifies that the full model's class predictions have been restored.

Accuracy drop is full accuracy minus ablated accuracy, expressed in percentage points. For example, 25% to 23% is a drop of 2 points. Negative drops are retained because removal may improve measured accuracy.

Compare the matching kinds: token ratio against token removal, channel ratio against channel removal, and total block ratio against whole-block removal. A large correction may be redundant; a small one may influence difficult decisions. Individually unimportant blocks cannot automatically be removed together.

Five examples per class keeps repeated validation practical, but small drops have limited precision. The CSV includes a paired standard-error estimate based on which validation predictions changed. This is a rough sampling measure, not variation across independently trained seeds or a proof of statistical significance.

## Output files

Results go to `analysis_results/imagenet32_final_updated/`.

| File | What it contains |
| --- | --- |
| `run_config.json` | Settings, source hashes, device, split sizes, checkpoint epochs, and run status |
| `training_curve.csv` | Every epoch's training loss, validation loss/accuracy, learning rate, and training time |
| `probe_checkpoints.csv` | Fixed-batch loss, accuracy, and learning rate at each measurement |
| `activation_variance.csv` | Stream variance and RMS after projection and each block |
| `branch_ratios.csv` | Input RMS, correction RMS, and their ratio for each branch and complete block |
| `linear_metrics.csv` | Weight RMS, gradient RMS, actual update RMS, update ratio, and linear-output variance |
| `ablation.csv` | Full and ablated accuracy, accuracy drops, and changed-correctness counts |
| `saturation.csv` | Created only when tanh or ReLU is actually present |
| `split_indices.npz` | The training, validation, and fixed-batch indices |
| `source/` | A snapshot of the analysis and model source used for this run |
| `figures/` | Ready-to-use PNG plots for the report |
| `analysis_summary.md` | Measured observations and reminders for interpreting them |

The plots include activation variance, token/channel/whole-block ratios, gradient RMS, update ratios, training curves, learning rate, ablation drops, and ratio-versus-ablation comparisons. Linear plots use layer order; `linear_metrics.csv` maps each index to its exact module name.

The script refuses to overwrite an existing run. For a repeat, change `RUN_NAME` at the top to a new name before running. Do not delete the previous measurements just to reuse the name.

## What to send back

Send `analysis_results/imagenet32_final_updated.zip` after the script prints that analysis is complete. It contains the measurements and figures needed to write the remaining report section, without images or trained weights.

We can then describe which layers grow or shrink the signal, whether gradients and updates remain useful, which branches matter most, and whether their magnitude predicts their ablation impact. The current Word report already explains the method and the logged experiments; its measurement prompts still need the actual results from this run.
