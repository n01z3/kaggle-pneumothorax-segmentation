from glob import glob

import cv2
import numpy as np
# import pickle
from torch import nn
from tqdm import tqdm


def dice_coef_metric_batch(outputs, targets):
    scores = []
    for i in range(outputs.shape[0]):
        output = outputs[i]
        target = targets[i]
        intersection = 2.0 * (target * output).sum()
        union = target.sum() + output.sum()
        if target.sum() == 0 and output.sum() == 0:
            scores.append(1.0)
        else:
            scores.append(intersection / union)
    return np.mean(scores)


def dice_coef_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union


def dice_coef_loss(inputs, target, prints=0):
    smooth = 1.0
    if prints:
        print("inp and tar:", inputs.min(), inputs.max(), target.min(), target.max())

    intersection = 2.0 * ((target * inputs).sum()) + smooth
    union = target.sum() + inputs.sum() + smooth
    if prints:
        print("intersection and union:", intersection, union)
    return 1 - (intersection / union)


def bce_dice_loss(inputs, target):
    dicescore = dice_coef_loss(inputs, target)
    bcescore = nn.BCELoss()
    bceloss = bcescore(inputs, target)

    return bceloss + dicescore


class SOTALoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = dice_coef_loss

    def forward(self, outputs, targets):
        bce_loss = self.bce(outputs, targets)
        dice_loss = self.dice(outputs, targets)
        return bce_loss + dice_loss


def main():
    filenames = sorted(
        glob(
            "/media/n01z3/red3_2/learning_dumps/pneumo/dn121_v5_1/fold_1/predictions/epoch0_metric0.8629_show/*png"
        )
    )
    print(filenames)
    y_true, y_pred = [], []
    for filename in tqdm(filenames, total=len(filenames)):
        raw = cv2.imread(filename, 0)
        print(raw.shape)

        y_t = raw[:, -768:] / 255.0
        y_p = raw[:, 768 + 3: 2 * 768 + 3] / 255.0
        y_pred.append(y_p)
        y_true.append(y_t)

        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(y_true)
        # plt.subplot(1,2,2)
        # plt.imshow(y_pred)
        # plt.show()

    y_true.append(np.zeros(y_true[-1].shape))
    y_pred.append(np.zeros(y_pred[-1].shape))

    scores1 = []
    for y_t, y_p in zip(y_true, y_pred):
        score = dice_coef_metric(y_t, y_p)
        scores1.append(score)

    print(scores1)
    print('per_image:', np.mean(scores1))

    y_true = np.expand_dims(np.array(y_true), 1)
    y_pred = np.expand_dims(np.array(y_pred), 1)

    print(y_pred.shape, y_true.shape)

    print('per_batch:', dice_coef_metric_batch(y_pred, y_true))


if __name__ == "__main__":
    main()
