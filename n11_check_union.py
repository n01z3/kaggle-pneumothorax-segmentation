__author__ = "bobbqe"

import os
import os.path as osp
import gc

import numpy as np
from n03_loss_metric import dice_coef_metric_batch
from n09_submit import binarize_sample


def check_mean_fold(fold):

    if "+" in model_name:
        model_lst = model_name.split("+")
    else:
        model_lst = [model_name]

    all_outputs_fold = []
    for model in model_lst:
        model_outputs = []

        filename = (
            f"/mnt/ssd1/dataset/pneumo/sota_predictions/{model}/{fold}_{model}_valid/{fold}_{model}_valid_index.npz"
        )
        tfz = np.load(filename)
        outputs, outputs_mirror, ids, gts = tfz["outputs"], tfz["outputs_mirror"], tfz["ids"], tfz["gts"]

        outputs = outputs / 255.0  # (N,H,W)
        outputs_mirror = outputs_mirror / 255.0
        for jj in range(outputs.shape[0]):
            model_outputs.append((outputs[jj][0]+outputs_mirror[jj][0])/2.)

        all_outputs_fold.append(model_outputs)    # (3,N,H,W)

    all_outputs_fold = np.array(all_outputs_fold)
    fold_mean = []
    for jj in range(all_outputs_fold.shape[1]):
        sample = 0
        for ii in range(all_outputs_fold.shape[0]):
            sample += all_outputs_fold[ii][jj]
        sample /= all_outputs_fold.shape[0]
        fold_mean.append(sample)

    fold_mean = np.array(fold_mean)  # (N,H,W)

    fold_mean_bin = []
    for i in range(fold_mean.shape[0]):
        fold_mean_bin.append(binarize_sample((fold_mean[i], mask_thresh, min_size_thresh, dilation)))
    fold_mean_bin = np.array(fold_mean_bin)

    return dice_coef_metric_batch(fold_mean_bin, gts)


def check_binary_union_fold(fold):

    if "+" in model_name:
        model_lst = model_name.split("+")
    else:
        model_lst = [model_name]


    models_mean = []
    models_mean_bin = []
    models_gt = []
    for model in model_lst:
        model_mean = []

        filename = (
            f"/mnt/ssd1/dataset/pneumo/sota_predictions/{model}/{fold}_{model}_valid/{fold}_{model}_valid_index.npz"
        )
        tfz = np.load(filename)
        outputs, outputs_mirror, ids, gts = tfz["outputs"], tfz["outputs_mirror"], tfz["ids"], tfz["gts"]

        outputs = outputs / 255.0  # (N,H,W)
        outputs_mirror = outputs_mirror / 255.0

        for jj in range(outputs.shape[0]):
            model_mean.append((outputs[jj][0]+outputs_mirror[jj][0])/2.)

        models_mean.append(model_mean)  # (3,N,H,W)
    models_mean = np.array(models_mean)

    for j in range(models_mean.shape[0]):
        model_mean_bin = []
        for k in range(models_mean[j].shape[0]):
            model_mean_bin.append(
                binarize_sample((models_mean[j][k], mask_thresh, min_size_thresh, dilation))
            )
        models_mean_bin.append(model_mean_bin)


    models_mean_bin = np.array(models_mean_bin)
    # print (models_mean_bin.shape)
    models_union_bin = np.amax(models_mean_bin, axis=0)  # (8,N,H,W)
    # print (models_union_bin.shape)

    return dice_coef_metric_batch(models_union_bin, gts)


def main():
    global model_name, n_fold, mask_thresh, min_size_thresh, dilation, dumps_dir
    # model_name = "sx50+sx101+se154"
    model_name = "sx50+sx101"
    n_fold = 8

    mask_thresh = 0.5
    min_size_thresh = 1000
    dilation = 0
    mean_scores, bin_union_scores = [], []

    print(f"Validation DICEs over 2 models :")
    print(f"                    MEAN        :      UNION  ")
    for fold in range(n_fold):
        mean_score = check_mean_fold(fold)
        mean_scores.append(mean_score)
        # mean_score = 0
        bin_union_score = check_binary_union_fold(fold)
        bin_union_scores.append(bin_union_score)
        print(f"Fold {fold} :    {mean_score}   :  {bin_union_score}")

    # for dice_score in mean_scores:
        
    print(f"Mean validation DICE over 2 models: ")
    print(f" {np.mean(mean_scores)}   :  {np.mean(bin_union_scores)}")


if __name__ == '__main__':
    main()