# kaggle pneumothorax segmentation #
https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation
----
Code for [ods.ai] Dies Irae team from SIIM-ACR Pneumothorax Segmentation challenge.

Features of the solution:
* Loss: BCE + DICE. No weighing or log. Just the sum of the losses.
* Batch 2. Just because
* A mixture of Unet and Linknet with a backbone from se-resnext50, se-resnext101, SENet154
* ConvTranspose in decoder
* Final Convolution 2x2
* No image normalization at all
* Sampling in which only batches containing non-zero samples are taken
* Scheduling: 40 epoch of Adam with CosineAnnealing, 15 epoch of SGD with CyclicLR, 16 epoch of Adam with CosineAnnealing

Logs from 1st stage:

![picture alt](https://github.com/n01z3/kaggle-pneumothorax-segmentation/blob/master/misc/stage1_logs.png)

Logs from 2nd stage:

![picture alt](https://github.com/n01z3/kaggle-pneumothorax-segmentation/blob/master/misc/stage2_logs.png)

