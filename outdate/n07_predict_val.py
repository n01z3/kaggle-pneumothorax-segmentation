import numpy as np
import pandas as pd
import os
import glob
import sys
import tqdm
from tqdm import tqdm_notebook
from PIL import Image, ImageFile
from scipy.misc import imread, imresize, imsave
from scipy import ndimage

# from mask_functions import rle2mask, mask2rle
ImageFile.LOAD_TRUNCATED_IMAGES = True


def dice_coef_metric(inputs, target):
    # print ("Metrics: ", inputs.min(), inputs.max(), target.min(), target.max())
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    # print ("Metrics inter & union: ", intersection, union)
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position : current_position + lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)


def mask_to_rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1
    return " " + " ".join(rle)


if __name__ == "__main__":

    IMG_SIZE = 1024  # 448
    SMALL_OBJ_THRESHOLD = 2000
    FOLD_ID = 5
    # device = torch.device('cuda:0')

    model_name = "unet_dense121"  # UnetSENet154

    sample_df = pd.read_csv("tables/sample_submission.csv")
    masks_ = sample_df.groupby("ImageId")["ImageId"].count().reset_index(name="N")
    masks_ = masks_.loc[masks_.N > 0].ImageId.values
    print(len(masks_))
    ###
    sample_df = sample_df.drop_duplicates("ImageId", keep="last").reset_index(drop=True)

    folds_df = pd.read_csv("tables/folds_v5.csv")
    fold_df = folds_df.loc[folds_df.fold_id == FOLD_ID]
    print(len(fold_df.drop_duplicates("ImageId", keep="last").reset_index(drop=True)))

    # thresholds = np.linspace(0.05, 0.95, num=19)
    thresholds = [0.5]

    val_dice_list = {}

    for threshold in thresholds:
        cntr = 0
        valdice = []
        for index, row in tqdm.tqdm(fold_df.iterrows(), total=len(fold_df)):
            cntr += 1
            image_id = row["ImageId"]
            annotations = row[" EncodedPixels"].strip()

            val_gt = np.zeros((IMG_SIZE, IMG_SIZE))
            # annotations = [item.strip() for item in annotations]
            # print (image_id)

            # print (annotations)
            if annotations != "-1":
                val_gt += rle2mask(annotations, IMG_SIZE, IMG_SIZE).T
            if IMG_SIZE != 1024:
                val_gt = imresize(val_gt, (1024, 1024), interp="bilinear").astype(float)

            val_gt = (val_gt >= 1).astype("float32")  # for overlap cases

            if os.path.exists("/mnt/ssd2/dataset/pneumo/predictions/sx101_fold5_val/" + image_id + ".png"):
                img_path = os.path.join("/mnt/ssd2/dataset/pneumo/predictions/sx101_fold5_val/", image_id + ".png")
                val_pred = imread(img_path)
                # val_pred = imresize(val_pred, (1024, 1024), interp='bilinear')

                out_cut = np.copy(val_pred / 255.0)
                # print (val_gt.max(),val_pred.max(), out_cut.max())
                out_cut[np.nonzero(out_cut <= threshold)] = 0.0
                out_cut[np.nonzero(out_cut > threshold)] = 1.0

                # out_cut, nr_objects = ndimage.label(out_cut)
                #
                # for ii in range(1, nr_objects + 1):
                #     if (out_cut[out_cut == ii].sum() / ii) < SMALL_OBJ_THRESHOLD:
                #         out_cut[np.nonzero(out_cut == ii)] = 0.0
                #
                # out_cut[np.nonzero(out_cut != 0)] = 1.0
                #
                # # ## fill
                # out_cut = ndimage.binary_fill_holes(out_cut).astype(out_cut.dtype)
                # out_cut = ndimage.binary_dilation(out_cut, iterations=2).astype(out_cut.dtype)

                valdice.append(dice_coef_metric(out_cut, val_gt))
            # print ("Validation loss:", picdice)
            # valdice += picdice
        print("Validation DICE:", np.mean(valdice))
        val_dice_list[threshold] = np.mean(valdice)

    best_val_th = max(val_dice_list, key=val_dice_list.get)
    print("Best threshold: ", best_val_th)
    print(" with val DICE score: ", val_dice_list[best_val_th])

    # sublist = []
    # for index, row in tqdm.tqdm(sample_df.iterrows(), total=len(sample_df)):
    #     image_id = row['ImageId']
    #     if image_id in masks_:
    #
    #         img_path = os.path.join('epoch0_metric0.8629_test/', image_id + '.png')
    #
    #         img = imread(img_path).T
    #         img = imresize(img, (1024, 1024), interp='bilinear')
    #
    #         out_cut = np.copy(img / 255.)
    #         out_cut[np.nonzero(out_cut <= best_val_th)] = 0.
    #         out_cut[np.nonzero(out_cut > best_val_th)] = 1.
    #
    #         # out_cut, nr_objects = ndimage.label(out_cut)
    #
    #         # for ii in range(1, nr_objects+1):
    #         # 	if (out_cut[out_cut==ii].sum() / ii) < SMALL_OBJ_THRESHOLD:
    #         # 		out_cut[np.nonzero(out_cut==ii)] = 0.
    #
    #         # out_cut[np.nonzero(out_cut!=0)] = 1.
    #
    #         # ## fill
    #         # out_cut = ndimage.binary_fill_holes(out_cut).astype(out_cut.dtype)
    #         # out_cut = ndimage.binary_dilation(out_cut, iterations=2).astype(out_cut.dtype)
    #
    #         if out_cut.sum() > 0:
    #             rle = mask_to_rle(out_cut, 1024, 1024)
    #         else:
    #             rle = " -1"
    #         sublist.append([image_id, rle])
    #     else:
    #         rle = " -1"
    #         sublist.append([image_id, rle])
    #
    # submission_df = pd.DataFrame(sublist, columns=sample_df.columns.values)
    # submission_df.to_csv('submission_' + model_name + '_' + str(IMG_SIZE) + '_' + str(best_val_th) + '.csv',
    #                      index=False)
