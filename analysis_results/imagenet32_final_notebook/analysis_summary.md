# ImageNet analysis results

Validation accuracy: 27.86% on 5,000 held-out training images.

## Measured observations

At epoch 20, stream variance was 2.97916 after projection and 57.2737 after the last block.

Smallest final gradient RMS: blocks.11.token_branch.mlp.3 = 0.000698572.

Largest final gradient RMS: patch_projection = 0.00777551.

For token removals, the largest drop was at depth 1: +23.860 percentage points.

The smallest absolute drop was at depth 9: +0.000 percentage points.

For channel removals, the largest drop was at depth 1: +27.720 percentage points.

The smallest absolute drop was at depth 6: +1.480 percentage points.

For whole block removals, the largest drop was at depth 1: +27.740 percentage points.

The smallest absolute drop was at depth 10: +1.820 percentage points.

Saturation: Not applicable: no tanh or ReLU layers.