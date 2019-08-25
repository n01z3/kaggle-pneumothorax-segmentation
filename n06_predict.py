__author__ = "n01z3"

import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from multiprocessing import Pool
from n03_loss_metric import dice_coef_metric_batch, dice_coef_metric
from n04_dataset import SIIMDataset_Unet
import numpy as np

DEVICE = torch.device("cuda:0")
from torch.autograd import Variable


def save_img(data):
    filename, image = data
    cv2.imwrite(filename, image)


@torch.no_grad()
def predict_fold(model_name, fold=0, mode="valid", out_folder="outs", weights_dir="outs", validate=True):
    assert mode in ("train", "valid", "test"), mode

    model_ft = torch.load(osp.join(weights_dir, f"{model_name}_fold{fold}_best.pth"))
    model_ft.to(DEVICE)
    model_ft.eval()

    dst = osp.join(out_folder, f"{fold}_{model_name}_{mode}")
    os.makedirs(dst, exist_ok=True)

    dataset_valid = SIIMDataset_Unet(mode=mode, fold=fold)
    vloader = torch.utils.data.DataLoader(dataset_valid, batch_size=2, shuffle=False, num_workers=8)

    progress_bar = tqdm(enumerate(vloader), total=len(vloader), desc=f"Predicting {mode} {fold}")

    outputs, gts, filenames = [], [], []
    for i, batch in progress_bar:
        images, targets, ids = batch

        mirror = torch.flip(images, (3,))
        batch1ch = torch.cat([images, mirror], dim=0)
        batch3ch = torch.FloatTensor(np.empty((batch1ch.shape[0], 3, batch1ch.shape[2], batch1ch.shape[3])))
        for chan_idx in range(3):
            batch3ch[:, chan_idx : chan_idx + 1, :, :] = batch1ch
        images = Variable(batch3ch.cuda())
        targets = targets.data.cpu().numpy()

        preictions = model_ft(images)

        probability = preictions.data.cpu().numpy()

        for j in range(2):
            probabilityTTA = np.mean(
                np.concatenate([probability[0 + j], probability[2 + j][:, :, ::-1]], axis=0), axis=0
            )
            outputs.append(np.uint8(255 * probabilityTTA))
            filenames.append(osp.join(dst, f"{ ids[j]}.png"))
            gts.append(targets[j, 0])

    with Pool() as pool:
        list(
            tqdm(
                pool.imap_unordered(save_img, zip(filenames, outputs)),
                total=len(filenames),
                desc="saving predictions to image",
            )
        )

    scores = []
    if validate:
        for y_pred, y_true in zip(gts, outputs):
            scores.append(dice_coef_metric(y_true >= 128, y_true))

    print(f"\n{model_name} fold{fold} {np.mean(scores):0.4f}")


def main():
    predict_fold("sx50", fold=1)


if __name__ == "__main__":
    main()
