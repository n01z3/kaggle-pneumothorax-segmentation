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
from n03_loss_metric import dice_coef_metric, dice_coef_metric_batch
from n04_dataset import SIIMDataset_Unet

# DEVICE = torch.device("cuda:0")

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

NCORE = 18
PATHS = get_paths()
PREDICTS = PATHS["dumps"]["predictions"]


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


def get_data_npz(model_name="sx101", fold=0):
    mode = "valid"
    name_pattern = f"{fold}_{model_name}_{mode}"

    filename = osp.join(PREDICTS, model_name, name_pattern, f"{name_pattern}_index.npz")
    tfz = np.load(filename)

    outputs, outputs_mirror, ids, gts = tfz["outputs"], tfz["outputs_mirror"], tfz["ids"], tfz["gts"]
    print(outputs.shape, outputs_mirror.shape, gts.shape)
    y_preds = np.mean(np.concatenate([outputs_mirror, outputs], axis=1), axis=1) / 255.0
    print(y_preds.shape)
    score = dice_coef_metric_batch(y_preds > 0.5, gts > 0.5)
    print(model_name, fold, score)
    gc.collect()
    return y_preds, gts, score, ids


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

    # with Pool() as p:
    #     scores = p.map(score_sample,
    #                    zip(y_preds, y_trues, total * [mask_thresh], total * [min_size_thresh], total * [dilation]))

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
    model = "sx101"
    folds = []
    n_folds = 2
    string_sep = "8===>"

    # sizes = [500, 1000, 1500, 2000, 2500]
    # mask_threshs = [0.4, 0.45, 0.5, 0.55, 0.60]
    dilations = [0, 1, 2, 3]

    y_preds, y_trues, base_scores = [], [], []
    for i in range(n_folds):
        ty_preds, ty_trues, scores, ids = get_data_npz(model, fold=i)
        base_scores.append(np.mean(scores))
        # folds.append((y_preds, y_trues))
        y_preds.append(ty_preds.astype(np.float32))
        y_trues.append(ty_trues.astype(np.float32))

    y_preds = np.concatenate(y_preds, axis=0).astype(np.float32)
    y_trues = np.concatenate(y_trues, axis=0).astype(np.float32)

    best_score = np.mean(base_scores)
    print("base_score", best_score)
    gc.collect()

    best_combo = ()
    for n in range(1000):
        size = random.randint(500, 2500)
        mask_thresh = random.uniform(0.42, 0.58)
        dilation = random.choice(dilations)
        if n == 0:
            size, mask_thresh, dilation = 1000, 0.5, 0  # sx101

        iter_scores = []
        # for y_preds, y_trues in folds:
        iter_score = score_fold(y_preds, y_trues, mask_thresh, size, dilation)
        #

        if iter_score > best_score:
            print(iter_score)
            best_score = iter_score
            best_combo = (model, mask_thresh, size, dilation)
            print(f'best combo', best_combo)
            print(string_sep)
            string_sep = string_sep[:-1] + "=>"
        else:
            print("not better", best_combo)

        # gc.collect()
        # score_fold(y_preds, y_trues, ids)


if __name__ == "__main__":
    random_search()
