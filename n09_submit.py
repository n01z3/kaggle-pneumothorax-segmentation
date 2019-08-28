__author__ = "n01z3"

import gc
import os
import os.path as osp
from glob import glob
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure, morphology
from tqdm import tqdm

from n01_config import get_paths
from n02_utils import mask_to_rle
from n03_loss_metric import dice_coef_metric_batch, dice_coef_metric
from n08_blend import get_data_npz

# DEVICE = torch.device("cuda:0")

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

NCORE = 18


def binarize_sample(data):
    el, mask_thresh, min_size_thresh, dilation = data
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
    return y_pred


def convert_one(sample_id):
    if "+" in model_name:
        model_lst = model_name.split("+")
    else:
        model_lst = [model_name]

    y_pred = []
    for model in model_lst:
        for fold in range(n_fold):
            filename = osp.join(dumps_dir, f"{fold}_{model}_test", f"{sample_id}.png")
            y_pred.append(cv2.imread(filename, 0) / 255.0)

    y_pred = np.mean(np.array(y_pred), axis=0)

    cv2.imwrite(f"/mnt/ssd2/dataset/pneumo/predictions/uint8/mean/{sample_id}.png", np.uint8(255 * y_pred))

    y_bin = binarize_sample((y_pred, mask_thresh, min_size_thresh, dilation))
    if np.sum(y_bin) > 0:
        rle = mask_to_rle(y_bin.T, 1024, 1024)
    else:
        rle = " -1"
    return rle


def convert_mask(mask):
    y_bin = binarize_sample((mask, mask_thresh, min_size_thresh, dilation))
    if np.sum(y_bin) > 0:
        rle = mask_to_rle(y_bin.T, 1024, 1024)
    else:
        rle = " -1"
    return rle


def get_all_data(model="se154", n_fold=2):
    y_preds = np.zeros((1377, 1024, 1024), np.float32)

    for i in range(n_fold):
        ty_preds, gts, _, ids = get_data_npz(model, fold=i, mode="test")
        y_preds += ty_preds.astype(np.float32)
        gc.collect()

    y_preds /= n_fold

    dice_coef_metric_batch(y_preds > 0.5, gts)

    # y_preds = np.mean(np.array(y_preds), axis=0).astype(np.float32)
    print(y_preds.shape, np.amax(y_preds), np.amin(y_preds))
    gc.collect()

    return y_preds, ids


def main():
    global model_name, n_fold, mask_thresh, min_size_thresh, dilation, dumps_dir
    model_name = "se154+sx50+sx101"
    n_fold = 8
    mask_thresh = 0.5
    min_size_thresh = 1000
    dilation = 0

    paths = get_paths()
    dumps_dir = "/mnt/ssd2/dataset/pneumo/predictions/old/sota_predictions"
    # dst = osp.join(dumps_dir, f"{fold}_{model_name}_valid")

    df = pd.read_csv("tables/sample_submission.csv")

    with Pool() as p:
        rles = p.map(convert_one, df["ImageId"])

    df["EncodedPixels"] = rles
    os.makedirs("subm", exist_ok=True)
    df.to_csv(f"subm/{model_name}_nf{n_fold}_{mask_thresh}_{min_size_thresh}_{dilation}.csv", index=False)

    empty = df[df["EncodedPixels"] == " -1"]
    print(f"empty {empty.shape[0] / df.shape[0]}")


def main_bad():
    global model_name, n_fold, mask_thresh, min_size_thresh, dilation, dumps_dir
    model_name = "se154"
    n_fold = 2
    mask_thresh = 0.4892385788890003
    min_size_thresh = 1598
    dilation = 1

    paths = get_paths()
    dumps_dir = osp.join(paths["dumps"]["path"], paths["dumps"]["predictions"])
    # dst = osp.join(dumps_dir, f"{fold}_{model_name}_valid")

    y_preds, ids = get_all_data(model_name, n_fold)
    print(ids[:10])
    print(y_preds.shape)

    #
    df = pd.DataFrame()
    df["ImageId"] = ids

    y_preds_lst = [y_preds[n] for n, sample_id in enumerate(ids)]

    for n, sample_id in enumerate(ids):
        cv2.imwrite(
            f"/mnt/ssd2/dataset/pneumo/predictions/uint8/se154/debug{sample_id}.png", np.uint8(255 * y_preds_lst[n])
        )

    with Pool() as p:
        rles = list(tqdm(p.imap_unordered(convert_mask, y_preds_lst), total=len(y_preds_lst), desc="converting"))

    df["EncodedPixels"] = rles
    df.drop_duplicates("ImageId", inplace=True)
    os.makedirs("subm", exist_ok=True)

    sample = pd.read_csv("tables/sample_submission.csv")
    df = df[df["ImageId"].isin(sample["ImageId"])]

    df.to_csv(f"subm/{model_name}_nf{n_fold}_{mask_thresh}_{min_size_thresh}_{dilation}.csv", index=False)

    empty = df[df["EncodedPixels"] == " -1"]
    print(f"empty {empty.shape[0] / df.shape[0]}")


def check_subm():
    df1 = pd.read_csv("subm/sx50+sx101_nf8_0.5_1000_0.csv")
    df2 = pd.read_csv("subm/se154_nf8_0.4892385788890003_1598_1.csv")

    df1.sort_values("ImageId", inplace=True)
    df2.sort_values("ImageId", inplace=True)

    print(df1.head())
    print(df2.head())


def compare_dice():
    fns1 = sorted(glob("/mnt/ssd2/dataset/pneumo/predictions/uint8/se154/debug2/*png"))
    fns2 = sorted(glob("/mnt/ssd2/dataset/pneumo/predictions/uint8/sx101/debug/*png"))

    scores = []
    for fn1, fn2 in tqdm(zip(fns1, fns2), total=len(fns1)):
        img1 = cv2.imread(fn1) / 255.0
        img2 = cv2.imread(fn2) / 255.0

        scores.append(dice_coef_metric(img1 > 0.5, img2 > 0.5))
    print(np.mean(scores))
    print(scores[:10])


if __name__ == "__main__":
    main()
    # check()

    # check_subm()
    # main_bad()
    # compare_dice()
