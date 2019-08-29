__author__ = "n01z3"

import gc
import os
import os.path as osp
import random
from multiprocessing import Pool

import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from tqdm import tqdm

from n01_config import get_paths
from n03_loss_metric import dice_coef_metric, dice_coef_metric_batch

# DEVICE = torch.device("cuda:0")

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

PATHS = get_paths()
PREDICTS = PATHS["dumps"]["predictions"]


def get_data_npz(model_name="sx101", fold=0, mode="valid"):
    if "+" not in model_name:
        models = [model_name]
    else:
        models = model_name.split("+")

    name_pattern = f"{fold}_{models[0]}_{mode}"
    filename = osp.join(PREDICTS, models[0], name_pattern, f"{name_pattern}_fp32_d.npz")
    tfz = np.load(filename)
    y_preds, ids, gts, disagreements = tfz["outputs"], tfz["ids"], tfz["gts"], tfz['disagreements']
    score = dice_coef_metric_batch(y_preds > 0.5, gts > 0.5)
    print(f"{models[0]} {score:0.5f}")

    if len(models) > 1:
        for n, model in enumerate(models[1:]):
            name_pattern = f"{fold}_{model}_{mode}"
            filename = osp.join(PREDICTS, model, name_pattern, f"{name_pattern}_fp32_d.npz")
            tfz = np.load(filename)
            y_preds += tfz["outputs"]
            score = dice_coef_metric_batch(y_preds > (n + 1) * 0.5, gts > 0.5)
            print(f"add {model} {score:0.5f}")

        y_preds /= len(models)

    print(y_preds.shape, gts.shape)
    del tfz
    gc.collect()
    return y_preds, gts, score, ids, disagreements


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

    with Pool() as p:
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
    print(model)
    n_folds = 4
    string_sep = "8===>"

    dilations = [0, 1, 2, 3]

    y_preds, y_trues, base_scores = [], [], []
    for i in range(n_folds):
        ty_preds, ty_trues, scores, ids, disagreements = get_data_npz(model, fold=i)
        base_scores.append(np.mean(scores))
        y_preds.append(ty_preds)
        y_trues.append(ty_trues)

    y_preds = np.concatenate(y_preds, axis=0)
    del ty_preds
    y_trues = np.concatenate(y_trues, axis=0)

    best_score = np.mean(base_scores)
    print("base_score", best_score)
    gc.collect()

    best_combo = ()
    for n in range(1000):
        size = random.randint(500, 2500)
        mask_thresh = random.uniform(0.45, 0.55)
        dilation = random.choice(dilations)
        if n == 0:
            size, mask_thresh, dilation = 1000, 0.5, 0  # sx101

        iter_score = score_fold(y_preds, y_trues, mask_thresh, size, dilation)

        if iter_score > best_score:
            print(iter_score)
            best_score = iter_score
            best_combo = (model, mask_thresh, size, dilation)
            print(f"best combo", best_combo)
            print(string_sep)
            string_sep = string_sep[:-1] + "=>"
        else:
            print("not better", best_combo)


if __name__ == "__main__":
    random_search()
