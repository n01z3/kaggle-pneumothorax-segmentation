__author__ = "n01z3"

from glob import glob
import os
import os.path as osp
import shutil
import numpy as np

# import pickle
import torch
import torch.utils.data


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position : current_position + lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)


def mask_to_rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1
    return " " + " ".join(rle)


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def select_best_checkpoint(folder, fold, model):
    fns = sorted(glob(osp.join(folder, f"*{model}_fold{fold}*.pth")))
    if len(fns) >= 1:
        return fns[-1]
    else:
        print(f'emplty {folder}')
        return None



def select_sota_weights(folder_input, folder_dist, model="se154"):
    os.makedirs(folder_dist, exist_ok=True)

    for fold in range(8):
        top_weight = select_best_checkpoint(folder_input, fold, model)
        print(top_weight)
        shutil.copyfile(top_weight, osp.join(folder_dist, osp.basename(top_weight)))


if __name__ == "__main__":
    select_sota_weights(
        "/media/n01z3/red3_2/learning_dumps/pneumo/sota_weights", "/media/n01z3/red3_2/learning_dumps/pneumo/sota_weights/se154"
    )
