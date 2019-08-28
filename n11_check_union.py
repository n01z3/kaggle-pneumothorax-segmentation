__author__ = "bobbqe"

import os
import os.path as osp
import gc
import operator
from multiprocessing import Pool

import numpy as np
from n03_loss_metric import dice_coef_metric_batch, dice_coef_metric
from n09_submit import binarize_sample

# NCORE = 18

def check_mean_fold(fold, dilation, min_size_thresh):

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
        agree_dice = dice_coef_metric(fold_mean[i], gts[i])
        # print (f"Pic dice on mean: {agree_dice}  and agreement: {agreement_tta}")
        if agree_dice > agreement_tta:
            dilation_c = 0
        else:
            dilation_c = dilation

        fold_mean_bin.append(binarize_sample((fold_mean[i], mask_thresh, min_size_thresh, dilation_c)))
    fold_mean_bin = np.array(fold_mean_bin)

    return dice_coef_metric_batch(fold_mean_bin, gts)


def check_binary_union_fold(fold, dilation, min_size_thresh):

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
            agree_dice = dice_coef_metric(models_mean[j][k], gts[k])
            # print (f"Pic dice on union: {agree_dice}  and agreement: {agreement_tta}")
            if agree_dice > agreement_tta:
                dilation_c = 0
            else:
                dilation_c = dilation
            sample_mean_bin = binarize_sample((models_mean[j][k], mask_thresh, min_size_thresh, dilation_c))

            model_mean_bin.append(sample_mean_bin)
        models_mean_bin.append(model_mean_bin)


    models_mean_bin = np.array(models_mean_bin)
    # print (models_mean_bin.shape)
    models_union_bin = np.amax(models_mean_bin, axis=0)  # (8,N,H,W)
    # print (models_union_bin.shape)

    return dice_coef_metric_batch(models_union_bin, gts)


def main(dilation_in, min_size_thresh_in, agreement_in):
    global model_name, n_fold, mask_thresh, dumps_dir, agreement_tta
    # model_name = "sx50+sx101+se154"
    model_name = "sx50+sx101"
    n_fold = 8

    mask_thresh = 0.5
    min_size_thresh = min_size_thresh_in
    dilation = dilation_in
    agreement_tta = agreement_in
    mean_scores, bin_union_scores = [], []

    for fold in range(n_fold):
        mean_score = check_mean_fold(fold, dilation, min_size_thresh)
        mean_scores.append(mean_score)
        bin_union_score = check_binary_union_fold(fold, dilation, min_size_thresh)
        bin_union_scores.append(bin_union_score)
        # print(f"Fold {fold} :    {mean_score}   :  {bin_union_score}")
        
    print(f"Mean validation DICE over 2 models: dilation: {dilation}  min_size_thresh: {min_size_thresh} agreement_tta: {agreement_tta}")
    print(f" {np.mean(mean_scores)}   :  {np.mean(bin_union_scores)}")

    return np.mean(mean_scores), np.mean(bin_union_scores)


if __name__ == '__main__':
    global stats_mean_dic, stats_union_dic
    stats_mean_dic = {}
    stats_union_dic = {}
    dilations = [1,2]
    # thresh_mins = np.linspace(1000, 1600, num=7)
    min_size_thresh = 1500
    agreement_th = np.linspace(0.7, 0.95, num=6)
    for dilation in dilations:
    #     for min_size_thresh in thresh_mins:
        for agreement in agreement_th:
        # with Pool() as p:
        #     scores = p.map(main, agreement_th)
            scores = main(dilation, min_size_thresh, agreement)

        stats_mean_dic[f'Dil: {dilation}  Min_size: {min_size_thresh} Dice_agree: {agreement}'] = scores[0]
        stats_union_dic[f'Dil: {dilation}  Min_size: {min_size_thresh} Dice_agree: {agreement}'] = scores[1]

    max_stats_mean = max(stats_mean_dic.items(), key=operator.itemgetter(1))
    max_stats_union = max(stats_union_dic.items(), key=operator.itemgetter(1))
    print(f"Best mean validation DICE over 2 models: {max_stats_mean[1]}  with set up: {max_stats_mean[0]}")
    print(f"Best union validation DICE over 2 models: {max_stats_union[1]}  with set up: {max_stats_union[0]}")



