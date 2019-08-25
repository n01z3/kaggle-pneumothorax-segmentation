__author__ = "n01z3"

import gc
import os
import os.path as osp
import random
from multiprocessing import Pool

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from skimage import measure, morphology
from tqdm import tqdm

from n01_config import get_paths
from n03_loss_metric import dice_coef_metric
from n04_dataset import SIIMDataset_Unet

# DEVICE = torch.device("cuda:0")

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

NCORE = 18

def read_prediction(filename):
    return cv2.imread(filename, 0) / 255.0


def get_data(model_name="sx101", fold=0):
    paths = get_paths()
    dumps_dir = osp.join(paths["dumps"]["path"], paths["dumps"]["predictions"])
    dst = osp.join(dumps_dir, f"{fold}_{model_name}_valid")

    dataset_valid = SIIMDataset_Unet(mode="valid", fold=fold)
    vloader = torch.utils.data.DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=NCORE)

    progress_bar = tqdm(enumerate(vloader), total=len(vloader), desc=f"generating masks f{fold}")

    y_trues, ids = [], []
    for i, batch in progress_bar:
        images, targets, batch_ids = batch
        y_trues.append(np.array(targets[0, 0] > 0.5))
        ids.append(batch_ids[0])

    filenames = [osp.join(dst, f"{sample_id}.png") for sample_id in ids]
    with Pool(NCORE) as p:
        y_preds = list(
            tqdm(p.imap_unordered(read_prediction, filenames), total=len(filenames), desc="reading predictions")
        )

    scores = []
    for yp, yt in zip(y_preds, y_trues):
        # print(np.amax(yp), np.amin(yp))
        scores.append(dice_coef_metric(yp > 0.5, yt))

    # print(scores)
    print(np.mean(scores))
    return y_preds, y_trues, scores, ids


def dump_data():
    model = "sx101"

    os.makedirs("tmp", exist_ok=True)
    scores = []
    for i in range(0, 8):
        y_preds, y_trues, score = get_data(model_name=model, fold=i)
        np.savez(f"tmp/{model}_{i}_val", y_preds=y_preds, y_trues=y_trues, score=score)
        scores.append(score)

    print(scores)


def eda_fold(y_preds, y_trues, ids, mask_thresh=0.5, min_size_thresh=1500, max_size_thresh=50000, dilation=2):
    y_binary = []
    for el, gt, sample_id in zip(y_preds, y_trues, ids):
        y_pred = el > mask_thresh
        if np.sum(y_pred) > 0:
            plt.figure(figsize=(15, 25))
            plt.subplot(3, 2, 1)
            plt.title("original")
            plt.imshow(el, cmap="gray")

            plt.subplot(3, 2, 2)
            plt.title(f"binary, score {dice_coef_metric(y_pred, gt):0.4f}")
            plt.imshow(y_pred, cmap="gray")

            labels = measure.label(y_pred)
            for n, region in enumerate(measure.regionprops(labels)):
                if region.area < min_size_thresh:
                    y_pred[labels == n + 1] = 0

            plt.subplot(3, 2, 3)
            plt.title(f"remove small, score {dice_coef_metric(y_pred, gt):0.4f}")
            plt.imshow(y_pred, cmap="gray")

            y_pred = ndimage.binary_fill_holes(y_pred)
            if dilation > 0:
                y_pred = morphology.dilation(y_pred, morphology.disk(dilation))

            plt.subplot(3, 2, 4)
            plt.title(f"morphology, score {dice_coef_metric(y_pred, gt):0.4f}")
            plt.imshow(y_pred, cmap="gray")

            labels = measure.label(y_pred)
            for n, region in enumerate(measure.regionprops(labels)):
                print(region.area)
                # if region.area < min_size_thresh:
                #     y_pred[labels == n + 1] = 0

            score = dice_coef_metric(y_pred, gt)

            plt.subplot(3, 2, 5)
            plt.title(f"gt score {score:0.4f}")
            plt.imshow(gt, cmap="gray")

            image = cv2.imread(f"/mnt/ssd2/dataset/pneumo/train_png/{sample_id}.png", 0)
            plt.subplot(3, 2, 6)
            plt.title(f"image")
            plt.imshow(image, cmap="gray")
            plt.show()


def score_sample(data):
    el, gt, mask_thresh, min_size_thresh, dilation = data
    y_pred = el > mask_thresh
    if np.sum(y_pred) > 0:

        labels = measure.label(y_pred)
        for n, region in enumerate(measure.regionprops(labels)):
            if region.area < min_size_thresh:
                y_pred[labels == n + 1] = 0

        y_pred = ndimage.binary_fill_holes(y_pred)
        if dilation > 0:
            try:
                y_pred = morphology.dilation(y_pred, morphology.disk(dilation))
            except:
                pass

    sample_score = dice_coef_metric(y_pred, gt)
    return sample_score


def score_fold(y_preds, y_trues, mask_thresh=0.5, min_size_thresh=1500, dilation=2):
    total = len(y_trues)
    with Pool(NCORE) as p:
        scores = list(
            tqdm(
                p.imap_unordered(
                    score_sample,
                    zip(y_preds, y_trues, total * [mask_thresh], total * [min_size_thresh], total * [dilation]),
                ),
                total=total,
                desc="scoring",
            )
        )
    return np.mean(scores)


def random_search():
    model = "sx50"
    folds = []
    n_folds = 5
    string_sep = "8===>"

    sizes = [500, 1000, 1500, 2000, 2500]
    mask_threshs = [0.4, 0.45, 0.5, 0.55, 0.60]
    dilations = [0, 1, 2, 3]

    base_scores = []
    for i in range(n_folds):
        y_preds, y_trues, scores, ids = get_data(model, fold=i)
        base_scores.append(np.mean(scores))
        folds.append((y_preds, y_trues))

    best_score = np.mean(base_scores)
    print("base_score", best_score)
    # gc.collect()

    for n in range(1000):
        size = random.choice(sizes)
        mask_thresh = random.choice(mask_threshs)
        dilation = random.choice(dilations)
        if n == 0:
            size, mask_thresh, dilation = 2, 0.5, 1000 # sx101

        iter_scores = []
        for y_preds, y_trues in folds:
            tscore = score_fold(y_preds, y_trues, mask_thresh, size, dilation)
            iter_scores.append(tscore)

        iter_score = np.mean(iter_scores)
        if iter_score > best_score:
            print(iter_score)
            best_score = iter_score
            print(model, mask_thresh, size, dilation)
            print(string_sep)
            string_sep = string_sep[:-1] + "=>"
        else:
            print('not better', iter_scores)

        # gc.collect()
        # score_fold(y_preds, y_trues, ids)


if __name__ == "__main__":
    random_search()
