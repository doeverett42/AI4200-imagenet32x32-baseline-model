# ImageNet training dynamics and ablation guide

This script measures the final ImageNet recipe in `model.py`. It is separate from the graded `train()` function, so the harness gets no extra hooks, validation passes, or ablations.

## What I need to run

1. Put `analysis.py` in the repository root, beside `src` and the harness notebook.
2. Use the existing HPC Python environment with PyTorch, NumPy, and Matplotlib. If Matplotlib is missing, install only it with `python -m pip install matplotlib`.
3. The script expects the harness cache at `data/imagenet32/X_train.npy` and `data/imagenet32/y_train.npy`. If the HPC cache is elsewhere, change the hardcoded `DATA_DIR` near the top of `analysis.py` to that folder. It does not download data.
4. Run `python analysis.py` in the HPC environment. There are no command-line arguments. This is a full analysis training run, not a small local experiment.
5. Keep the resulting `analysis_results/imagenet32_final` folder. To rerun, choose a new hardcoded `RUN_NAME`; the script refuses to overwrite a folder containing earlier results.

The code reads the current final settings: 12 patch blocks, 20 epochs, GELU, SGD learning rate 0.05, momentum 0.9, horizontal flipping 0.5, and noise standard deviation 0.02. It trains from scratch. Printed losses cannot recover missing initialization or midway measurements from the old run, so one instrumented run is necessary unless those measurements were already saved.

## What the script does

It holds out five training images from every ImageNet class: 5,000 validation images in total. The split is deterministic and saved in `split_indices.npz`. The official harness test images are never read. The remaining training images are still memory-mapped uint8 data; only the current batch is converted to floating point. This avoids creating full-size floating-point copies of ImageNet in RAM.

It chooses 128 fixed, clean training images for the measurement batch. Those image indices, their labels, and their prepared pixel values stay fixed across snapshots. At 20 epochs, measurements happen at epochs 0, 2, 10, and 20.

At a snapshot, `TrainingDynamics.measure()` copies the model and optimizer together. The copied optimizer retains the actual momentum buffers. The copied model uses eval mode, but gradients are enabled. It runs the fixed batch, computes cross entropy, records gradients, and takes one optimizer step on the copy. The live training model, optimizer, modes, and RNG state are preserved. The reported update ratio is therefore a real **diagnostic step on the fixed eval-mode batch**, not an approximation and not the last stochastic training update.

At each epoch the script also records loss and accuracy on the same clean validation split. This produces a training curve for the final recipe. The analysis run uses a slightly smaller training set than the harness, so its validation score should be reported separately from the historical 24.80% harness score.

## Understanding the measurements

**Activation variance:** The population variance over all entries of the residual stream after the patch projection and after each complete block. Depth zero means the projection. A growing curve means the values spread out; a shrinking curve means they become less varied. LayerNorm normalizes the branch input, so the skip stream itself does not have to maintain variance one.

**Branch-to-stream ratio:** The RMS of a branch correction divided by the RMS of that branch's input. RMS is the square root of the mean squared values. Token and channel branches have separate denominators: channel mixing receives the token-updated stream. The CSV also saves their absolute RMS values. A smaller ratio can result from a larger stream, even if the branch output stays similar. The whole-block ratio measures the combined change after both additions.

**Gradient RMS:** The RMS of each Linear layer's weight gradient from the fixed-batch loss. It covers the patch projection, four matrices per patch block, and the final classifier. The current model has 50 Linear layers. Use their CSV names to match plot positions to depth. A missing gradient is labeled separately from a genuine zero. Compare the profile across checkpoints before describing gradients as vanishing or exploding.

**Update-to-weight ratio:** The RMS of the copied optimizer step's actual weight change divided by the RMS of the weights immediately before the step. This includes momentum or AdamW behavior. Around 0.001 is a rough comparison point, not a rule that every layer must satisfy. The plot keeps true zeros visible using a symmetric logarithmic scale.

**Saturation:** The final model uses GELU, so the assignment's tanh and ReLU diagnostics are not applicable. The run configuration states this explicitly. If the model is changed to tanh, the code records the fraction of activation inputs with absolute value above two. For ReLU, it records the zero-output fraction and the fraction of hidden units zero or nonzero everywhere over the fixed batch and shared spatial positions. Being inactive on this batch does not prove a unit is permanently dead.

## How the ablation works

Each of the 12 blocks has a token branch and a channel branch. That gives **24 individual branch ablations**. The script also tests **12 whole-block ablations** as a separate comparison.

The important distinction is where the skip addition happens. A `MixingBranch` returns just its correction, so a hook on that branch must return `torch.zeros_like(output)`. The parent then computes the original skip stream plus zero. Returning the branch input would add that stream twice. A hook on the whole `PatchBlock` returns `inputs[0]`, removing both corrections.

Every trial starts from the same trained model, disables only one target, and evaluates the same 5,000 clean validation images. No ablated model is retrained. A `finally` block removes the hook, and the full-model predictions are checked again at the end.

The plotted drop is full-model accuracy minus ablated-model accuracy, in percentage points. Positive means removal hurt accuracy. Negative means removal improved the measured accuracy. The CSV also records how many previously correct predictions were lost and how many became correct. Its paired standard error describes sampling uncertainty on this validation subset; it is not variation across training seeds.

Compare the token and channel series separately by block depth. Whole-block removal removes both, so it answers a different question. Then use the scatterplot to compare final branch ratios with the corresponding accuracy drops. A large correction need not be indispensable; a small correction can still change a difficult decision. Individually removable branches cannot automatically be removed together.

## Files to use in the report

| File | What it contains |
| --- | --- |
| `run_config.json` | Exact settings, seed, software, parameter count, source hashes, split sizes, timing, and completion status |
| `training_curve.csv` | One training loss and clean validation loss/accuracy per epoch |
| `activation_variance.csv` | Stream variance and RMS at depth zero and after every block |
| `branch_ratios.csv` | Token, channel, and whole-block RMS measurements at every snapshot |
| `linear_metrics.csv` | Weight-gradient RMS, actual step RMS, update ratio, and Linear output variance |
| `probe_checkpoints.csv` | Fixed-batch loss, accuracy, and current learning rate at each snapshot |
| `saturation.csv` | Only created when tanh or ReLU modules are present |
| `ablation.csv` | Full and ablated validation scores, drops, and paired correctness counts |
| `figures/` | Ready-to-insert PNG plots |
| `analysis_summary.md` | Measured observations and interpretation reminders |

`run_config.json` says `complete` only after the CSVs, figures, summary, and ablation checks have finished successfully. An interrupted run can retain partial results without being presented as complete.

## Completing the editable report

The Word report already contains the model recipes, recorded harness results, experiment explanations, failures, and an actual comparison of historical ImageNet training curves. The internal dynamics and ablation sections have specific prompts instead of invented findings.

After the HPC run, insert the matching figures and replace those prompts using the CSV values. Include the projection-to-last-block variance change, gradient and update extremes by layer, the most damaging individual branch removals, and whether branch ratios follow the same ranking as accuracy drops. Keep the new analysis validation accuracy distinct from the historical harness score.

One additional verification remains: MNIST's historical 99.27% result was logged before an unrecorded hidden-width change from 1568 to the current 1024. Verify a matching current-width harness result before labeling 99.27% as its exact final score. The current CIFAR and ImageNet recipes have matching recorded results.

## Small code map

`CachedImages` reads selected images without loading all float tensors. `split_indices` builds the saved validation split. `make_optimizer` and `make_scheduler` follow the existing training recipe. `TrainingDynamics` manages the fixed-batch hooks and copied step. `evaluate` computes clean validation metrics while restoring mode and RNG state. `ablate_branches` removes one correction at a time. `make_plots` creates the figures, and `write_summary` records measured observations. `main` connects those pieces in training order.

The analysis is intentionally a separate experiment: the graded `train()` and `predict()` functions remain unchanged. Do not use the tiny synthetic verification outputs as project results.
