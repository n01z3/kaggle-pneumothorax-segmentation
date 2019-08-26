__author__ = "n01z3"

import os
import os.path as osp
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure, morphology

from n01_config import get_paths
from n02_utils import mask_to_rle

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
    # print(y_pred.shape)

    y_bin = binarize_sample((y_pred, mask_thresh, min_size_thresh, dilation))
    if np.sum(y_bin) > 0:
        rle = mask_to_rle(y_bin.T, 1024, 1024)
    else:
        rle = " -1"
    return rle


def main():
    global model_name, n_fold, mask_thresh, min_size_thresh, dilation, dumps_dir
    model_name = "sx50+sx101"
    n_fold = 8
    mask_thresh = 0.5
    min_size_thresh = 1000
    dilation = 0

    paths = get_paths()
    dumps_dir = osp.join(paths["dumps"]["path"], paths["dumps"]["predictions"])
    # dst = osp.join(dumps_dir, f"{fold}_{model_name}_valid")

    df = pd.read_csv("tables/sample_submission.csv")

    with Pool() as p:
        rles = p.map(convert_one, df["ImageId"])

    df["EncodedPixels"] = rles
    os.makedirs("subm", exist_ok=True)
    df.to_csv(f"subm/{model_name}_nf{n_fold}_{mask_thresh}_{min_size_thresh}_{dilation}.csv", index=False)

    empty = df[df["EncodedPixels"] == " -1"]
    print(f"empty {empty.shape[0] / df.shape[0]}")


if __name__ == "__main__":
    main()
