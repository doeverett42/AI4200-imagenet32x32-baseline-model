### Change 0 — mnist — 2026-10-16

**Change:** Baseline
**Why:** NA
**Expected:** Baseline results 
**Result:** Test Accuracy: 95.95% 1.1 min to train on lmy local machine with 1 GPU. 

### Change 1 - mnist - 2026-10-21

**Change:** Reduced MNIST depth from 30 residual blocks to 6. All other settings stayed the same.
**Why:** MNIST may not need 30 blocks to separate ten handwritten digits. I want to see whether a simpler residual model can learn as well with less computation.
**Expected:** Fewer parameters and faster training. Validation accuracy might improve or stay similar if the extra depth was unnecessary; it could fall if six blocks are insufficient.
**Result:** Test accuracy 95.95% -> 96.82%. Train loss 0.1884. 1.1 min to train on my local machine with 1 GPU. The model performed better with less 
depth, so complexity was not improving it but perhaps causing too much training memorization. I will add momentum to possibly escape local minima. 

### Change 2 - mnist - 2026-10-21
**Change:** Changed MNIST's SGD momentum from 0.0 to 0.9, keeping learning rate 0.05 and the accepted architecture unchanged.
**Why:** I want to test whether using recent gradient direction helps the model learn more effectively than plain SGD within the same ten epochs.
**Expected:** Faster useful loss reduction and possibly higher validation accuracy. Larger oscillations or unstable loss are possible if the combined update is too aggressive.
**Result:** Test accuracy 96.82% -> 97.08%. Train loss 0.0418. 0.5 min to train on my local machine with 1 GPU. The model performed significantly better with training loss and fairly better with test accuracy. I will keep the momentum and try ReLU. 

### Change 0 — cifar10 — 2026-10-16

**Change:** Baseline
**Why:** NA
**Expected:** Baseline results 
**Result:** Test Accuracy: 48.62% 4.8 min to train on my local machine with 1 GPU. 

### Change 0 — imagenet32 — 2026-10-16

**Change:** Baseline
**Why:** NA
**Expected:** Baseline results 
**Result:** Test Accuracy: 6.77% 37.2 min to train on my local machine with 1 GPU. 

### Change 1 — cifar10 — 2026-10-

**Change:** 
**Why:** 
**Expected:** 
**Result:** 