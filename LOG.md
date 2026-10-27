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

**Change:** Changed learning rate from 0.05 to 0.001. 
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

### Change 9 - mnist - 2026-10-21

**Change:** Changed width from 128 to 392 and changed hidden from 256 to 1568. 
**Why:** The model seemed to get stuck on training loss, so I wanted to push it further by increasing complexity to learn. 
**Expected:** More feature dimensions would allow the model to reduce training loss but may lead to overfitting. 
**Result:** Validation accuracy 98.39% -> 98.42%. Train loss 0.0001. 1.1 min to train on my local machine with 1 GPU. The increased complexity did allow the model to achive the lowest training loss so far, but the increase in test accuracy was only marginal of +0.01% from the best previous model. 

### Change 10 - mnist - 2026-10-21

**Change:** Changed weight decay from 0.01 to 0.05. 
**Why:** The model already has an extremely low training loss, so I want to help it generalize better given the increased complexity
**Expected:** Weight decay will help the model take into account more data from all the inputs so that is performs better on test data. 
**Result:** Validation accuracy 98.42% -> 98.40%. Train loss 0.0001. 1.1 min to train on my local machine with 1 GPU. The added weight decay kept training loss the same, and test accuracy decreased. I will go back to weight decay of 0.01. 

### Change 11 - mnist -2026-10-21

**Change:** Changed model width from 392 to 522. 
**Why:** Increasing the complexity of features increased the model's ability to learn marginally, so maybe it needs more features to continue to improve with the low training loss already. 
**Expected:** The model should improve in training loss, but it might begin overfitting for test data. 
**Result:** Validation accuracy 98.40% -> 98.39%. Train loss 0.0001. 1.3 min to train on my local machine with 1 GPU. The added weight decay kept training loss the same, and test accuracy decreased. I will go back to width of 392. 

### Change 12 - mnist -2026-10-21

**Change:** Changed model depth from 6 to 10. 
**Why:** The training loss is already increasing low, meaning maybe all the information is in the machine already, but it needs more time to process and reshape.
**Expected:** The model should stay the same in training loss and maybe will show improvement in test accuracy or just take more time to train.  
**Result:** Validation accuracy 98.42% -> 98.46%. Train loss 0.0001. 1.6 min to train on my local machine with 1 GPU. The added depth did not affect training loss but did increase test accuracy along with training time.

### Change 13 - mnist - 2026-10-23

**Change:** Changed model depth from 10 to 50.
**Why:** Deep networks can process the features in more ways to discover patterns, and I want to push the depth as far as I can without exeeding training time. 
**Expected:** The model should stay the same in training loss and maybe will show improvement in test accuracy or just take more time to train.  
**Result:** Validation accuracy 98.46%-> 98.08% Training loss 0.0004. 6.0 min to train on my local machine with 1 GPU. The increased depth actually hurt both validation and training loss, which could be a sign of overfitting and training loss oscillation. I will reduce the epochs to see if it generalizes better at the start. 

### Change 14 - mnist - 2026-10-23

**Change:** Changed model epochs from 30 to 1. 
**Why:** Deep networks can process the features in more ways to discover patterns, but they only generalize well in the beginning before starting to memorize. 
**Expected:** The model might drop some in training loss, but it could show improvement in test accuracy due to better generalization. 
**Result:** Validation accuracy 98.08% -> 84.23%% Training loss 1.8235. 0.3 min to train on my local machine with 1 GPU. The validation accuracy and training loss both drastically decreased because the network is probably not deep or complex enough to train in one epoch. 

### Change 15 - mnist - 2026-10-23

**Change:** Model returned to best previous model in all features. Added label smoothing of 0.05 to MNIST’s cross-entropy loss. 
**Why:** The model already fits the training data extremely well. I want to see whether slightly softer targets help it recognize unseen digits instead of becoming overly confident about training examples.
**Expected:** Higher training loss because the targets changed, but possibly better validation accuracy. Too much smoothing could instead reduce accuracy.
**Result:** Test accuracy 98.46% -> 98.42%. Train loss 0.2830. 1.7 min to train. Label smoothing did not improve accuracy in this run. The higher training loss was expected because smoothing changes the loss calculation; it was close to the minimum possible smoothed loss of 0.2824. I will remove label smoothing and keep my previous best model.

### Change 16 - mnist - 2026-10-23

**Change:** Reduced batch size from 256 to 64. 
**Why:** I want to see whether more frequent updates help the model learn a solution that performs better on unseen digits.
**Expected:** More optimizer updates and possibly better validation accuracy, but longer training and potentially less stable loss.
**Result:** Test accuracy 98.46% -> 98.25%. Train loss 0.0001. 4.9 min to train. The smaller batch size still allowed the model to fit the training data, but accuracy decreased and training took roughly three times longer. I will restore batch size 256 and test mild dropout separately.

### Change 17 - mnist - 2026-10-23

**Change:** Added dropout of 0.1 after the hidden activation in each residual branch. 
**Why:** The model already fits the training data very well. I want to see whether temporarily turning off some hidden activations during training helps it rely less on particular features and recognize unseen digits better.
**Expected:** Training loss may increase because learning becomes harder while dropout is active. Accuracy could improve, but unnecessary dropout could instead make performance worse.
**Result:** Test accuracy 98.46% -> 98.49%. Train loss 0.0009. 1.6 min to train. Dropout produced three more correct predictions on the test set without noticeably increasing training time. I will next test dropout of 0.2 with all other settings unchanged.

### Change 18 - mnist - 2026-10-24

**Change:** Replaced LayerNorm with BatchNorm inside each residual branch and before the classification head.
**Why:** Increasing model size and adjusting regularization produced little improvement. I want to see whether normalizing each feature across training examples works better than normalizing each image’s features separately.
**Expected:** Different training behavior and possibly better accuracy without increasing model size. BatchNorm could also perform worse because it uses batch statistics during training.
**Result:** Test accuracy 98.49% -> 98.27%. Train loss 0.0011. 1.8 min to train. Training loss decreased substantially, but this did not produce better test accuracy. The sharp later loss reductions also followed the scheduled learning-rate decreases, so they don't show that BatchNorm would benefit more from longer training. I will restore LayerNorm and test GELU separately.

### Change 19 - mnist - 2026-10-24

**Change:** Replaced tanh with GELU in the residual branches and classification head.
**Why:** Increasing model size and changing regularization produced little improvement. I want to test whether a different activation helps the model learn representations that work better on unseen digits.
**Expected:** Different learning behavior and possibly better accuracy without adding parameters. GELU could also perform worse, so lower training loss alone would not establish an improvement.
**Result:** Test accuracy 98.49% -> 98.56%. Train loss 0.0003. 2.0 min to train. GELU produced seven more correct predictions than my previous best model, although training took slightly longer. 

### Change 20 - mnist - 2026-10-24

**Change:** Replaced two-layer classification head with a single Linear layer.
**Why:** The residual backbone already transforms each image into learned features, so I want to see whether those features can be classified directly instead of passing through another hidden layer.
**Expected:** Fewer parameters and possibly slightly faster training. Accuracy could improve if the extra head was unnecessary, or decrease if its nonlinear processing was useful.
**Result:** Test accuracy 98.56% -> 98.57%. Train loss 0.0005. 1.9 min to train. The simpler head achieved essentially the same accuracy with fewer parameters. 

### Change 21 - mnist - 2026-10-24

**Change:** Replaced GELU in the residual branches with two-piece maxout. Each branch now calculates two candidates for every hidden feature and keeps the larger. 
**Why:** The previous changes produced only small improvements. Maxout lets each hidden feature select between separately learned responses, which may help the model recognize different handwriting patterns.
**Expected:** Validation accuracy may improve, but training will likely take longer. The additional parameters could also increase overfitting. 
**Result:** Test accuracy 98.57% -> 98.54%. Train loss 0.0005. 1.8 min to train. The more complex parameters of maxout performed slightly worse on test accuracy and achived the same training loss. However, the training loss looks more stagnant in areas, so maybe it needs a smoother scheduler decrease. Also, maxout could do better with info across features since there is more complexity to capture them; I will try batch normalization again. 

### Change 22 - mnist - 2026-10-24

**Change:** Replaced LayerNorm with BatchNorm inside each residual branch and before the linear classification head. 
**Why:** BatchNorm performed worse with my earlier tanh model, but maxout changes the learned features. I want to test whether BatchNorm works better with this new architecture.
**Expected:** Different training behavior and possibly better accuracy. The earlier BatchNorm result does not necessarily predict its effect when combined with maxout.
**Result:** Test accuracy 98.54% -> 98.73%. Train loss 0.0006. 2.0 min to train on my local machine with 1 GPU. This is my highest observed accuracy so far. 

### Change 23 - mnist - 2026-10-24

**Change:** Replaced StepLR with cosine annealing over 30 epochs, starting at learning rate 0.001 and approaching a minimum of 0.00001. 
**Why:** StepLR changes the learning rate abruptly every seven epochs. I want to test whether a gradual reduction and smaller final updates let the model refine its learned weights more effectively.
**Expected:** Smaller learning-rate changes between epochs and possibly lower final training loss. Accuracy might improve, stay similar, or decrease.
**Result:** Test accuracy 98.73% -> 98.77%. Train loss 0.0006 -> 0.0003. 1.5 min to train on my local machine with 1 GPU. Cosine annealing produced a more gradual training-loss decrease and a lower final loss. The accuracy improvement was small, but this is my highest observed result now.

### Change 24 — mnist — 2026-10-24

**Change:** Reduced epochs from 30 to 25. 
**Why:** The last five epochs showed minimal gains in training loss and showed signs of possibly memorizing trianing rather than generalizing. 
**Expected:** Training loss will slightly increase since less updates are made, but test accuracy could increase if this cuts the model's training prior to memorizing. 
**Result:** Test Accuracy: 98.77% -> 98.82%. Train loss 0.0003 -> 0.0004. 1.3 min to train on my local machine with 1 GPU. The model performed better on test accuracy because it was probably stopped from updating before it started memorizing. 

### Change 25 - mnist - 2026-10-24

**Change:** Added training-only random horizontal and vertical shifts of up to two pixels. Rotation, flipping, and noise remained disabled. 
**Why:** The model already fits the original training images very closely. I want to test whether recognizing digits in slightly different positions helps it classify unseen handwriting without relying too strongly on exact pixel locations.
**Expected:** Training loss may increase because the model sees changing versions of the images. Validation accuracy may improve if learning these positional variations helps generalization, but unnecessary shifts or clipping near image edges could reduce accuracy. Prediction and parameter count remain unchanged.
**Result:** Test accuracy 98.82% -> 99.01%. Train loss 0.0004 -> 0.0181. 1.4 min to train on my local machine with 1 GPU. Training loss increased while learning from randomly shifted images, but test accuracy improved by 0.19 percentage points. 

### Change 26 — mnist — 2026-10-24

**Change:** Increased epochs from 25 to 40. 
**Why:** Now that the training loss is higher due to more augmented data to learn from, the model needs more time to generalize the patterns.  
**Expected:** Training loss will decrease significantly and test accuracy may increase as a result from more generalizing time unless it starts to memorize again.
**Result:** Test Accuracy: 99.01% -> 99.19%%. Train loss 0.0181 -> 0.0122. 2.9 min to train on my local machine with 1 GPU. The training loss gradually increased even more along with test accuracy, but so did training time. 

### Change 27 — mnist — 2026-10-25

**Change:** Added 7 degree rotation shift. 
**Why:**  Given that the model improved from data translation, it should also generalize more given rotation. 
**Expected:** Training loss may increase due to more data augmentation, but test accuracy should improve with more data variation to learn from rotation. 
**Result:** Test Accuracy: 99.19% -> 99.27%. Train loss 0.0122 -> 0.0149. 2.5 min to train on my local machine with 1 GPU. Training loss slightly worsened, but test accuracy drastically improved, meaning more augmentationed helped generalization. 

### Change 28 — mnist — 2026-10-25

**Change:** Added 0.02 random noise augmentation.
**Why:**  Given that the model improved from data translation & rotation, it should also generalize more given random noise. 
**Expected:** Training loss may increase due to more data augmentation, but test accuracy should improve with more data variation to learn from random noise. 
**Result:** Test Accuracy: 99.27% -> 99.14%. Train loss 0.0149 -> 0.0160. 2.1 min to train on my local machine with 1 GPU. Adding random noise slighty worsened both training loss and test accuracy. I will remove random noise. 

### Change 0 — cifar10 — 2026-10-16

**Change:** Baseline
**Why:** NA
**Expected:** Baseline results 
**Result:** Test Accuracy: 48.62% 4.8 min to train on my local machine with 1 GPU. 

### Change 1 - cifar10 - 2026-10-25

**Change:** Replaced tanh with SiLU in the residual branches and classification head. 
**Why:** The lecture described SiLU as a smooth, input-dependent gate. I want to test whether this different nonlinear response helps the model learn useful features before making a larger architectural change.
**Expected:** Different training behavior and possibly better validation accuracy without adding parameters. SiLU may also perform worse.
**Result:** Test accuracy 48.62% -> 46.95%. Final epoch train loss 0.2538. 1.8 min to train on my local machine with 1 GPU. SiLU did not improve accuracy under the existing training settings, even though training loss continued decreasing. I will not keep SiLU. 

### Change 2 - cifar10 - 2026-10-25

**Change:** Replaced the tanh residual branches with SwiGLU. Increased width from 128 to 512 and hidden size from 256 to 1024, reduced depth from 30 blocks to 8, and replaced the two-layer classification head with one linear layer. 
**Why:** CIFAR images contain more complicated patterns than MNIST. A wider model gives the network more features to work with, while SwiGLU learns gates that modify the strength of those features. Fewer residual blocks provide a shorter path through the network, and I removed the two-layer classification head because it didn't prove helpful on MNIST. 
**Expected:** Accuracy may improve because the model has more capacity and can learn interactions between features. The model has more parameters overall, so training time could increase. 
**Result:** Test accuracy 48.62% -> 55.10%. Final training loss 0.0007. 1.2 min to train on my local machine with 1 GPU. The wider, shallower SwiGLU model improved accuracy by 6.48. However, the nearly zero training loss suggests it fits the training examples much better than unseen images.

### Change 3 - cifar10 - 2026-10-25

**Change:** Replaced SwiGLU with Maxout using two candidates per hidden feature. 
**Why:** I wanted to compare Maxout against SwiGLU using the same architecture size and number of parameters. Maxout selects the largest learned candidate for each feature, while SwiGLU multiplies learned values by learned gate. 
**Expected:** Accuracy may improve or decrease depending on which approach learns more useful features. I think SwiGLU will perform better since more parameters are being used on any one given example. 
**Result:** Test accuracy 55.10% -> 57.68%. Final training loss 0.0006. 1.0 min to train on my local machine with 1 GPU. Maxout performed better under these settings, so I will keep it for the next experiment. Both models reached nearly zero training loss, so I will focus on augmenting the data. 

### Change 4 - cifar10 - 2026-10-25

**Change:** Added random horizontal flipping with a probability of 0.5 during training.
**Why:** The previous model reached nearly zero training loss but achieved 57.68% test accuracy. Flipping gives it different views of the same objects, which may help it recognize unseen images.
**Expected:** Accuracy may improve because the model learns to recognize objects facing either direction. Training loss may remain higher because the training images vary between epochs.
**Result:** Test accuracy 57.68% -> 56.54%. Final training loss 0.1300. 1.4 min to train on my local machine with 1 GPU. Flipping did not improve accuracy within 30 epochs. Training loss was still decreasing at the end, so I will test whether the augmented model benefits from more training.

### Change 5 - cifar10 - 2026-10-25

**Change:** Increased training from 30 to 60 epochs while keeping horizontal flipping enabled. All other settings stayed unchanged.
**Why:** The augmented training loss was still falling when the previous run ended. The model may need more updates to learn both original and mirrored images.
**Expected:** Lower training loss and possibly higher validation accuracy, with roughly twice the training time. 
**Result:** Test accuracy 56.54% -> 60.38%. Final training loss 0.0003. Training time 1.4 -> 2.2 min. Additional training improved accuracy while flipping stayed unchanged. This also beat the unaugmented 30-epoch model by 2.70 points. 

### Change 6 - cifar10 - 2026-10-25

**Change:** Added random horizontal and vertical shifts of up to two pixels with reflection padding. 
**Why:** The current model fits the training images extremely well. Small shifts may help it recognize objects in different positions instead of depending too heavily on their exact pixel locations.
**Expected:** Test accuracy may improve because the model encounters more variations of each training image. Training loss may stay higher because those images are harder to fit. 
**Result:** Test accuracy 60.38% -> 59.35%. Final training loss 0.4660. Training took 2.2 min. Adding two-pixel shifts did not improve accuracy within 60 epochs. Training loss was still decreasing, so I will try momentum to speed it up. 

### Change 7 - cifar10 - 2026-10-25

**Change:** Increased SGD momentum from 0.0 to 0.9.
**Why:** The translated model's training loss was still decreasing at epoch 60 and ended at 0.4660. Momentum carries information from previous updates, which may help the model learn these varied images more quickly.
**Expected:** Faster reduction in training loss and possibly better validation accuracy than the translation-only run. 
**Result:** Test accuracy 59.35% -> 63.12%. Final training loss 0.5293. 1.6 to train on my local machine with one GPU. The model significantly improved in test accuracy while still being slightly higher with training loss. This suggests the model learned more with momentum but perhaps has more generalizations to learn. 

### Change 8 - cifar10 - 2026-10-25

**Change:** Replaced SGD with AdamW and changed the learning rate from 0.05 to 0.001.
**Why:** Training loss was still decreasing after 60 epochs with SGD. AdamW adjusts update sizes for individual weights, which may help the model learn useful features more quickly.
**Expected:** Faster reduction in training loss and possibly higher validation accuracy. Training time may increase or decrease.
**Result:** Test accuracy 63.12% -> 62.47%. Final training loss 0.5737. Training time 1.6 -> 2.0 min. AdamW did not improve accuracy or training time with these settings, so I will return to SGD with momentum.

### Change 9 - cifar10 - 2026-10-25

**Change:** Replaced LayerNorm with BatchNorm in every residual block and before the classification head. 
**Why:** Increasing the learning rate did not produce the faster learning I wanted. BatchNorm changes how features are normalized across training examples and may make the model easier to optimize.
**Expected:** Faster loss reduction at matching epoch counts and possibly higher validation accuracy. 
**Result:** Test accuracy 63.12% -> 62.22%. Training loss decreased from 0.5293 to 0.2308. Training took 1.8 min. BatchNorm improved training loss but reduced test accuracy, so I will restore LayerNorm.

### Change 10 - cifar10 - 2026-10-25

**Change:** Added rotation data augmentation of 7. 
**Why:** Data augmentation has proven to yield the best results, and on this data set where images are more variable, more changes to the images might help the model generalize better. 
**Expected:** Training loss may increase becasue the model has to see more variation with only the same time to learn it; however, test accuracy may increase becasue the varied images help it generalize better. 
**Result:** Test accuracy 63.12% -> 62.67%. Training loss 0.5375. 2.3 min to train on my local machine with one gpu. Adding rotation augmentation did not help my model in either train loss or test accuracy; maybe because the images are not expected to be rotated like perhaps the written digits could be written slantedly. 

### Change 13 - cifar10 - 2026-10-25

**Change:** Increased the CIFAR model's depth from 8 to 30 residual blocks.
**Why:** Accuracy remained in the low 60% range. More residual blocks give the model additional stages to process features, which may help it recognize more complicated patterns.
**Expected:** Possibly higher accuracy from the additional capacity, with longer training time. More depth could also make optimization harder.
**Result:** Test accuracy 62.92% -> 63.07%. Final training loss 0.6362. Training time 2.3 -> 6.0 min. Accuracy was slightly worse than the previous best run, and training loss increased and training time more than doubled. I will first try giving more augmentation before reverting. 

### Change 14 - cifar10 - 2026-10-25

**Change:** Tested the 30-block model with random rotation between -7 and +7 degrees and Gaussian noise with noise_std = 0.01. 
**Why:** I wanted to see whether rotation and mild pixel noise for a largere augmentation treatment could help the deeper model recognize more variations of the same objects and generalize better. 
**Expected:** Possibly higher test accuracy from the additional image variations. Training loss may remain higher because the augmented examples are harder to fit within the same 60 epochs.
**Result:** Test accuracy 63.07% -> 62.40%. Final training loss 0.7045. 4.3 min to train on my local machine with one gpu. Accuracy decreased and training loss increased. This combination did not improve accuracy. It seems like my current architecture is hitting a limit of about 63% accuracy.

### Change 15 - cifar10 - 2026-10-25

**Change:** Enabled a configurable residual patch MLP with 4x4 patches, 128 features per patch, six blocks, token hidden size 64, and channel hidden size 256. Selected GELU, LayerNorm, ordinary MLP branches, and a linear head. Kept SGD at lr 0.05 with momentum 0.9, 60 epochs, batch size 256, zero weight decay/dropout, no scheduler, and the current flipping, translation, rotation, and noise settings.
**Why:** I want to test whether processing image patches with shared weights helps the model learn useful local patterns. Separate token and channel branches let it combine information between regions and within each region. Keeping the options configurable lets me test them individually.
**Expected:** Possibly better validation accuracy and learning efficiency given more ways of processing patterns. Training time must be measured because fewer parameters do not necessarily mean faster epochs. 
**Result:** Test accuracy (best prev model) 63.12% -> 71.71%. Training loss 0.3877. 3.3 min to run on my local machine with one gpu. The residual patch mlp demonstrated significant improvement in both test accuracy and training loss, showing that processing immages sequentially and mixing their features across patches and channels allows the neurons to discover more local and general patterns to an image. 

### Change 16 - cifar10 - 2026-10-25

**Change:** Increased patch blocks from 6 to 12.  
**Why:** More patch blocks could allow the model to transform and discover more complex features. 
**Expected:** The training loss should decrease given more parameters allow the model to learn the dataset; test accuracy could increase from this or decrease if the model begins to specialize. 
**Result:** Test accuracy 71.71% -> 70.06%. Training loss 0.2529. 6.0 min to run on my local machine with one gpu. The residual patch mlp demonstrated significant improvement in training loss, but it did worse in test accuracy, meaning it began to overfit and specialize on the testing data.

### Change 16 - cifar10 - 2026-10-25

**Change:** Decreased patch blocks from 12 to 10.  
**Why:** Lese patch blocks to hopefully find a sweet spot of lowering training loss and helping the model generalize useful features. 
**Expected:** The training loss will increase given less parameters, but the test accuracy could increase if this depth is before the model begins to specialize. 
**Result:** Test accuracy 70.06% -> 70.88%. Training loss 0.2726. 5.0 min to run on my local machine with one gpu. The model performed slightly worse on training loss and still seemed to over specialize on the training data from test accuracy. I will revert to 6. 

### Change 17 - cifar10 - 2026-10-25

**Change:** Added dropout to the patch mlp with 6 blocks. 
**Why:** The model has shown signs of overfitting, so adding dropout could help the model rely less on certain features and generalize more. 
**Expected:** Training loss will likey increase because it loses some of the inputs it was training on, but test accuracy should increase if the model generalizes better. 
**Result:** Test accuracy 71.71% -> 72.76%. Training loss 0.5332. 3.6 min to run on my local machine with one gpu. The model performed slightly worse on training loss but improved in test accuracy, showing that patch improves and generalizes better with dropout. 

### Change 18 - cifar10 - 2026-10-26

**Change:** Removed noise_std and rotation in data augmentation. 
**Why:** Because the images are already scaled down to 32x32, adding noise or rotation could obscure the target category too much to be useful.  
**Expected:** Training loss will decrease because training images are easier to process, and test accuracy could increase if the model gathers more useful patterns from the noise-free images. 
**Result:** Test accuracy 72.76% -> 75.45%. Training loss 0.4226. 3.4 min to run on my local machine with one gpu. The model performed better on both training loss and test accuracy because the random noise and added rotation augmentation did not provide useful generalizations. 

### Change 19 - cifar10 - 2026-10-26

**Change:** Added a cosine scheduler down to 0.00001 learning rate. 
**Why:** The model has a low learning rate and perhaps if it's keeping the same lr throughout, then it is overshooting directions of the loss plain that could allow the model to reach lower loss and better accuracy. 
**Expected:** Training loss will decrease because the model is able to make finer grained updates and thus should be able to generalize better as well. 
**Result:** Test accuracy 75.45% -> 75.69%. Training loss 0.5104. 4.4 min to run on my local machine with one gpu. The model performed better on both test accuracy but slightly worse on trainin loss, which suggests there may be some overfitting occuring. 

### Change 20 - cifar10 - 2026-10-26

**Change:** Increased epochs from 60 to 80.
**Why:** The model still shows evidence of decreasing significantly in training loss, which could translate into more test accuracy gain from more training. 
**Expected:** Training loss will decrease from more training on the data, and test accuracy could increase as long as the model doesn't begin overfitting; time to train will increase. 
**Result:** Test accuracy 75.69% -> 76.01%. Training loss 0.3548. 4.9 min to on remote hpc cluster with one gpu. The model performed better on both test accuracy and training loss, showing more training time is better. 

### Change 21 - cifar10 - 2026-10-26

**Change:** Added back mlp head after mixer blocks.
**Why:** Because cifar is a complex data set, it could likely benefit from an additional classifier head with more parameters to distinguish between the patterns found by the mixer blocks. 
**Expected:** Training loss will likely decrease due to the increase of parameters to better learn the training set, test accuracy might increase if more complexity at the end of the network improves generalization, and training time will likely increase to backpropagate through the head. 
**Result:** Test accuracy 76.01% -> 77.63%. Training loss 0.2764. 4.1 min to run on remote hpc cluster with one gpu. The model performed better on both test accuracy and training loss, showing more complexity at the end of the network is better. 

### Change 22 - cifar10 - 2026-10-26

**Change:** Increase epochs from 80 to 90. 
**Why:** Training loss is still decreasing and test accuracy is increasing along with it, therefore more time to test could allow the model to generalize better. 
**Expected:**  Training loss will decrease, training time will increase, and test accuracy may increase if the model continues to generalize but could decrease if it memorizes. 
**Result:** Test accuracy 77.63% -> 77.82%. Training loss 0.2156. 4.4 min to run on remote hpc cluster with one gpu. The model performed better on both test accuracy and training loss, demonstrating that more training time allowed it to continue learning.

### Change 10 - cifar10 - 2026-10-27

**Change:** Added RGB channel scaling with rgb_jitter = 0.05. Each training image receives independent red, green, and blue intensity multipliers between 0.95 and 1.05. 
**Why:** Objects can appear under different lighting and color balance. Small changes to channel intensities may help the model recognize useful patterns without depending too heavily on exact colors.
**Expected:** Possibly higher accuracy from learning mild color variations. Training loss may increase because the images vary more. 
**Result:** Test accuracy 77.82% -> 78.11%. Training loss 0.2551. 4.1 min to run on remote hpc cluster with one gpu. The model performed better significantly on test accuracy and only performed slightly worse on training loss, showing that learning given slight variances in each channel per image does increase generalization.

### Change 0 — imagenet32 — 2026-10-16

**Change:** Baseline
**Why:** NA
**Expected:** Baseline results 
**Result:** Test Accuracy: 6.77% 37.2 min to train on my local machine with 1 GPU. 

### Change 1 - imagenet32 - 2026-10-26

**Change:** Replaced the flat residual MLP with a residual patch MLP using 4×4 patches, 128 features per patch, six mixing blocks, token hidden size 64, and channel hidden size 256. The model uses tanh, LayerNorm, ordinary MLP branches, and a linear classifier with 1,000 outputs. Kept 10 epochs, batch size 256, SGD learning rate 0.05, zero momentum, no weight decay or dropout, no scheduler, and no augmentation.
**Why:** Patch mixing produced my strongest CIFAR results, so I want to test whether the same way of processing images helps ImageNet32. Shared patch processing may help the model learn useful local patterns, while token and channel mixing combine information across image regions and learned features.
**Expected:** Possibly higher accuracy than the original model's 6.77%. ImageNet has more classes and more varied images, so the CIFAR improvement may not transfer directly. 
**Result:** Test accuracy 6.77% -> 13.54%. Training loss 4.5708. Training took 12.0 min on remote hpc cluster. The patch model achieved twice the baseline accuracy, suggesting that shared patch processing and mixing learned more useful image features under these settings. Training loss was still decreasing at the end, so I will test momentum next to see whether the model can learn more within the same 10 epochs.

### Change 2 - imagenet32 - 2026-10-26

**Change:** Increased SGD momentum from 0.0 to 0.9. 
**Why:** Training loss was still decreasing after 10 epochs. Momentum carries information from previous updates, which may help the model make faster progress without increasing the number of epochs.
**Expected:** Faster reduction in training loss and possibly higher accuracy within the same 10 epochs, with similar training time.
**Result:** Test accuracy 13.54% -> 17.29%. Final training loss 4.1471. Training took 12.2 min on remote hpc cluster. With momentum, the model reached a lower training loss after five epochs than the previous model reached after ten. Momentum improved both training progress and test accuracy.

### Change 3 - imagenet32 - 2026-10-26

**Change:** Replaced tanh with GELU in the patch model's token-mixing and channel-mixing branches. 
**Why:** Tanh flattens for large inputs, which can weaken the signal used to update weights. GELU does not cap large positive outputs, and it's recommended by the mlp-mixer paper, so I want to include it. 
**Expected:** Possibly faster reduction in training loss and higher accuracy within the same 10 epochs.
**Result:** Test accuracy 17.29% -> 21.42%. Training loss 3.7395. 10.4 min on the remote HPC cluster. GELU improved both training loss and test accuracy, so I will keep it. 

### Change 4 - imagenet32 - 2026-10-26

**Change:** Added random horizontal flipping with a probability of 0.5 during training. 
**Why:** ImageNet objects can appear facing different directions. Mirroring training images may help the model recognize the same category in either orientation.
**Expected:** Possibly higher accuracy from seeing more image variations. Training loss could increase because the varied examples are harder to learn, and augmentation will likely add processing time.
**Result:** Test accuracy 21.42% -> 21.71%. Training loss 3.7244. 14.9 min to train on remote HPC cluster. Flipping produced a small accuracy improvement in this run, but the longer training time makes this seem not as necessary; I will keep for the time being. 

### Change 5 - imagenet32 - 2026-10-26

**Change:** Added random horizontal and vertical shifts of up to two pixels with reflection padding.
**Why:** Small shifts may help the model recognize objects in different positions. They also change where image features fall relative to the 4x4 patch boundaries, which may help the model rely less on exact positioning.
**Expected:** Possibly higher accuracy from learning these positional variations. Training loss may increase because the shifted images are harder to learn. Translation also adds more data variation to process, likely increasing training time. 
**Result:** Test accuracy 21.71% -> 21.58%. Training loss 3.8242. 11.0 min to train on the remote HPC cluster. Adding translation did not improve accuracy within 10 epochs, so I will remove it and return to flipping alone.

### Change 6 - imagenet32 - 2026-10-26

**Change:** Added 0.02 noise_std to slightly obscure images. 
**Why:** Small variation in pixel values might help the model gather information from blurred instances. 
**Expected:** Possibly higher accuracy if variation helps the model generalize better, but more data variation will cause overhead and training time to increase along with training loss. 
**Result:** Test accuracy 21.71% -> 21.99%. Training loss 3.7329. 10.5 min to train on the remote HPC cluster. Adding random noise improved both training loss and test accuracy, showing that perturbing the image allowed the model to generalize better. 

### Change 7 - imagenet32 - 2026-10-26

**Change:** Changed optimizer to AdamW with learning rate 0.001. 
**Why:** Adaptive per-parameter updates may make better progress within the same training budget if SGD is the limiting factor.
**Expected:** Different convergence and possibly higher validation accuracy, but not a guaranteed improvement.
**Result:** Validation accuracy 21.99% -> 20.84%. Train loss 3.8142. 10.6 min to train on remote cluster with one gpu. AdamW proved worse than SGD, which could just be that it randomly for this seed did not fall to a lower minimum. I will not keep it for now.

### Change 8 - imagenet32 - 2026-10-26

**Change:** Increased patch depth from 6 to 12. 
**Why:** Because imagenet is more complex with 1000 categories, more depth of patch blocks could help the model understand patterns with less data as one category may only have one corresponding image; the model needs to learn this quickly.  
**Expected:** More depth should push validation accuracy down, and if the model generalizes better from this, it should perform better on test accuracy as well. Training time will increase because depth is greater and the model has more parameters to train. 
**Result:** Validation accuracy 21.99% -> 23.26%. Train loss 3.5687. 17.8 min to train on remote cluster with one gpu. Increasing depth vastly improved test accuracy, and validation accuracy improved slightly; more depth allowed the model to capture more patterns but also drastically increased training time. 

### Change 9 - imagenet32 - 2026-10-26

**Change:** Increased epochs from 10 to 20. 
**Why:** The training loss is continuing to go down throughout the 10 epochs, so it probably needs more time to learn or more epochs. 
**Expected:** Because the number of epochs have doubled, the training time is likely to double, but the training loss should continue to decrease and test accuracy should improve. 
**Result:** Validation accuracy 23.26% -> 24.80%. Train loss 3.2406. 45.3 min to train on remote cluster with one gpu. With more epochs, the model did learn more and improved in both training loss and test accuracy.

### Change 9 - imagenet32 - 2026-10-26

**Change:** Add mlp head to the architecture with 256 neurons. 
**Why:** The model still has very low accuracy, which suggest it doesn't have enough complexity to capture the signal relating to the many classes, so adding an extra head block should help interpret the output from the mixing blocks.  
**Expected:** The increased number of parameters will drive the training time up slightly, but training loss should decrease given more complexity, and hopefully this translates to useful signal and improving test accuracy. 
**Result:** Validation accuracy 24.80% -> %. Train loss .  min to train on remote cluster with one gpu. 