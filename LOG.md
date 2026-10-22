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

### Change 3 - mnist - 2026-10-21

**Change:** Replaced MNIST's tanh/Xavier recipe with ReLU and the existing automatic He initialization for matrices feeding ReLU. 
**Why:** I want to test whether a different nonlinear response and its matching starting weight scale improve learning.
**Expected:** Different activation/gradient behavior and possibly better validation accuracy. ReLU may instead hurt if many useful features become inactive.
**Result:** Validation accuracy 97.08% -> 96.82%. Train loss 0.0392. 0.7 min to train on my local machine with 1 GPU. The model learned worse compared with tanh because it performed better on training data but worse on test data. I will remove ReLU with He initialization because it seemed to overfit more. 

### Change 4 - mnist - 2026-10-21

**Change:** Enabled MNIST's existing StepLR schedule. Starting learning rate and ten-epochs stayed unchanged.
**Why:** Smaller later updates may refine the model after larger early updates have made progress.
**Expected:** Better late-epoch validation performance. If decay happens too early, learning may flatten before the model has learned enough.
**Result:** Validation accuracy 97.08% -> 97.86%. Train loss 0.0335. 0.5 min to train on my local machine with 1 GPU. Smaller later updates helped the final result. Next I will try adding more epochs to make the scheduler more impactful. 

### Change 4 - mnist - 2026-10-21

**Change:** Changed number of training epochs from 10 to 30.
**Why:** The learning rate scheduler halves the learning rate every 1/3 epochs, so I wanted to give the model more time for each learning rate variation. 
**Expected:** More epochs means more gradient updates to refine the model, and going from large to smaller learning rates should improve test accuracy without overly memorizing the dataset. 
**Result:** Validation accuracy 97.86% -> 98.22%. Train loss 0.0019. 1.4 min to train on my local machine with 1 GPU. More epochs drastically improved training loss, resulting in signifcant test accuracy as well. I will try a new optimizer next. 

### Change 5 - mnist - 2026-10-21

**Change:** Changed optimizer to AdamW. 
**Why:** Adaptive per-parameter updates may make better progress within the same training budget if SGD is the limiting factor.
**Expected:** Different convergence and possibly higher validation accuracy, but not a guaranteed improvement.
**Result:** Validation accuracy 98.22% -> 93.43%. Train loss 0.2114. 1.6 min to train on my local machine with 1 GPU. The AdamW optimizer with learning rate 0.05 showed train loss that decreased drastically initially but then began to oscillate, which indicated it was too large. I will lower it greatly for next test. 

### Change 6 - mnist - 2026-10-21

**Change:** Changed learning rate from 0.05 to 0.01. 
**Why:** Previous learning rate should training loss oscillations. 
**Expected:** More consisting training loss decrease and convergence to a minimum. 
**Result:** Validation accuracy 93.43% -> 98.02%. Train loss 0.0072. 1.4 min to train on my local machine with 1 GPU. The AdamW optimizer with learning rate 0.001 showed train loss that decreased drastically throughout, even reaching loss of 0.0004 but it increased again at the end. I will try a bigger scheduler decrease. 

### Change 7 - mnist - 2026-10-21
**Change:** Changed scheduler from having every 1/3 epochs to every 1/4 epochs. 
**Why:** Previous learning rate combined with scheduler saw a rise in training loss and oscillation in final epochs.  
**Expected:** More consisting training loss decrease and convergence to a minimum over all epochs. 
**Result:** Validation accuracy 98.02% -> 98.35%. Train loss 0.0002. 0.9 min to train on my local machine with 1 GPU. The AdamW optimizer with learning rate 0.001 and a halving scheduler that occured every 1/4 epochs showed significant decrease in training loss and accompanying substantial increase in test accuracy. 

### Change 8 - mnist - 2026-10-21
**Change:** Added weight decay of 0.001. 
**Why:** The model already has an extremely low training loss, so I want to help it generalize better. 
**Expected:** Weight decay will help the model take into account more data from all the inputs so that is performs better on test data. 
**Result:** Validation accuracy 98.35% -> 98.41%. Train loss 0.0002. 0.9 min to train on my local machine with 1 GPU. The weight decay did not hinder training loss and it marginally improved test accuracy, so I will increase it. 

### Change 8 - mnist - 2026-10-21
**Change:** Changed weight decay from 0.001 to 0.01. 
**Why:** The model already has an extremely low training loss, so I want to help it generalize better. 
**Expected:** Weight decay will help the model take into account more data from all the inputs so that is performs better on test data. 
**Result:** Validation accuracy 98.41% -> 98.39%. Train loss 0.0003. 1.1 min to train on my local machine with 1 GPU. The added weight decay reduced training loss accuracy and validation accuracy, meaning the model may have reached maximum complexity. I will try expanding feature dimensions.

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