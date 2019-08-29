__author__ = "n01z3"

import shutil
import argparse
import os
import os.path as osp
from glob import glob
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure, morphology

from n01_config import get_paths
from n02_utils import mask_to_rle
from n14_blend import get_data_npz

# DEVICE = torch.device("cuda:0")

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


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
    path_pattern = osp.join(predict_dir, "tmp", f"{model_name}*{sample_id}*")
    fns = glob(path_pattern)
    assert len(fns) == 1, path_pattern

    y_pred = np.load(fns[0])
    agreement = float(osp.basename(fns[0]).split("|")[1])
    if agreement < 0.65 and correction:
        y_bin = binarize_sample((y_pred, mask_thresh, 1500, 2))
    else:
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


def parse_args():
    parser = argparse.ArgumentParser(description="pneumo submit")
    parser.add_argument("--mask_thresh", help="binarization threshold", default=0.5, type=float)
    parser.add_argument("--min_size_thresh", help="filter small objets", default=1000, type=int)
    parser.add_argument("--dilation", help="dilation size", default=1, type=int)
    parser.add_argument("--model_name", help="models name to predict", default="se154", type=str)
    parser.add_argument("--correction", help="apply correction", default=True, type=bool)
    args = parser.parse_args()
    return args


def dump_predicts():
    try:
        shutil.rmtree(dst)
    except:
        print(f"no {dst}")
    os.makedirs(dst, exist_ok=True)
    y_preds, y_trues, scores, ids, disagreements = get_data_npz(model_name, fold=0, mode="test")

    for i in range(1, n_folds):
        ty_preds, ty_trues, _, _, tdisagree = get_data_npz(model_name, fold=i, mode="test")
        y_preds += ty_preds
        disagreements += tdisagree

    y_preds /= n_folds
    disagreements /= n_folds

    print(y_preds.shape, np.amax(y_preds), np.amin(y_preds))
    print(np.sort(disagreements)[::-1])
    print(ids)
    print(len(ids))
    print(len(sorted(set(list(ids)))))

    for i, (sample_id, disagree) in enumerate(zip(ids, disagreements)):
        y_pred = y_preds[i]
        np.save(osp.join(dst, f"{model_name}|{disagree:0.5f}|{sample_id}"), y_pred)
        cv2.imwrite(osp.join(debug, f"{model_name}|{disagree:0.5f}|{sample_id}.png"), np.uint8(255 * y_pred))


def make_submite():
    df = pd.read_csv("tables/stage_2_sample_submission.csv")

    with Pool() as p:
        rles = p.map(convert_one, df["ImageId"])

    df["EncodedPixels"] = rles
    os.makedirs("subm", exist_ok=True)
    df.to_csv(
        f"subm/st2_{model_name}_{mask_thresh}_{min_size_thresh}_{dilation}_corr{int(correction)}.csv", index=False
    )

    empty = df[df["EncodedPixels"] == " -1"]
    print(f"empty {empty.shape[0] / df.shape[0]}")


if __name__ == "__main__":
    global model_name, mask_thresh, min_size_thresh, dilation, predict_dir, correction
    n_folds = 8
    paths = get_paths()
    predict_dir = osp.join(paths["dumps"]["path"], paths["dumps"]["predictions"])
    dst = osp.join(predict_dir, "tmp")
    debug = osp.join(predict_dir, "debug")
    os.makedirs(debug, exist_ok=True)
    args = parse_args()

    model_name, mask_thresh, min_size_thresh, dilation, correction = (
        args.model_name,
        args.mask_thresh,
        args.min_size_thresh,
        args.dilation,
        args.correction,
    )

    dump_predicts()
    make_submite()
