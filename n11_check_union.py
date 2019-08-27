__author__ = "bobbqe"

import os
import os.path as osp
import gc

import numpy as np
from n03_loss_metric import dice_coef_metric_batch
from n09_submit import binarize_sample


def check_binary_union():

    if "+" in model_name:
        model_lst = model_name.split("+")
    else:
        model_lst = [model_name]

    mean_dice_score = []
    folds_models_mean = []
    for fold in range(n_fold):
        all_outputs_fold = []
        fold_models_mean = []
        for model in model_lst:
            model_outputs_fold = []

            filename = (
                f"/mnt/ssd2/dataset/pneumo/predictions/uint8/{model}/{fold}_{model}_valid/{fold}_{model}_valid_index.npz"
            )
            tfz = np.load(filename)
            outputs, outputs_mirror, ids, gts = tfz["outputs"], tfz["outputs_mirror"], tfz["ids"], tfz["gts"]

            outputs = outputs / 255.0  # (N,H,W)
            outputs_mirror = outputs_mirror / 255.0
            all_outputs_fold.append(outputs)
            model_outputs_fold.append(outputs)
            all_outputs_fold.append(outputs_mirror)  # (3*2,N,H,W)
            model_outputs_fold.append(outputs_mirror)

            fold_model_mean = np.mean(np.array(model_outputs_fold), axis=0)  # (N,H,W)
            fold_models_mean.append(fold_model_mean)  # (3,N,H,W)

        folds_models_mean.append(fold_models_mean)  # (8,3,N,H,W)
        folds_gts.append(gts)  # (8,N,H,W)

        fold_mean = np.mean(np.array(all_outputs_fold), axis=0)  # (N,H,W)

        fold_mean_bin = []
        for i in range(fold_mean.shape[0]):
            fold_mean_bin.append(binarize_sample((fold_mean[i], mask_thresh, min_size_thresh, dilation)))
        fold_mean_bin = np.array(fold_mean_bin)

        mean_dice_score.append(dice_coef_metric_batch(fold_mean_bin, gts))

    for dice_score in mean_dice_score:
        print(f"Mean validation DICE over {len(model_lst)} models for fold {fold} :  {dice_score}")
    print(f"Mean validation DICE over {len(model_lst)} models : {np.mean(mean_dice_score)}")

    for i in range(folds_models_mean.shape[0]):
        for j in range(folds_models_mean[i].shape[0]):
            for k in range(folds_models_mean[i][j].shape[0]):
                folds_models_mean_bin.append(
                    binarize_sample((folds_models_mean[i][j][k], mask_thresh, min_size_thresh, dilation))
                )

        folds_models_union_bin = np.amax(folds_models_mean_bin, axis=0)  # (8,N,H,W)
        fold_union_dice_score = dice_coef_metric_batch(folds_models_union_bin, gts[i])
        union_dice_scores.append(fold_union_dice_score)

    for dice_score in union_dice_scores:
        print(f"Union validation DICE over {len(model_lst)} models for fold {fold} :  {dice_score}")
    print(f"Mean Union validation DICE over {len(model_lst)} models : {np.mean(union_dice_scores)}")


def main():
    global model_name, n_fold, mask_thresh, min_size_thresh, dilation, dumps_dir
    # model_name = "sx50+sx101+se154"
    model_name = "sx50+sx101"
    n_fold = 8

    mask_thresh = 0.5
    min_size_thresh = 1000
    dilation = 0

    check_binary_union()

if __name__ == '__main__':
    main()