import os.path as osp
import numpy as np
import pandas as pd
import os
import glob
import sys
import tqdm
from tqdm import tqdm_notebook
import datetime
import time
import torch
from torch.autograd import Variable
import torch.utils.data
from PIL import Image, ImageFile
from scipy.misc import imread, imresize, imsave
from scipy import ndimage
import cv2
import collections
import torchvision
from torchvision import transforms
import random

# from pretrained_models import *
import albumentations
from n01_config import get_paths
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# from mask_functions import rle2mask, mask2rle
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    fold = 5
    model_name = 'UnetSEResNext101'
    paths = get_paths()

    test_dir = os.path.join(paths["dataset"]["path"], paths["dataset"]["test_dir"])

    IMG_SIZE = 1024  # 448
    SMALL_OBJ_THRESHOLD = 2000
    device = torch.device("cuda:0")

    model_ft = torch.load(f"outs/{model_name}_fold{fold}_best.pth")

    dst = 'outs/tmp'
    dst_dir = osp.join(dst, f'{model_name}_fold{fold}')
    os.makedirs(dst_dir, exist_ok=True)

    model_ft.to(device)
    model_ft.eval()

    # if not os.path.exists(model_name+'_out_all/'):
    # 	os.makedirs(model_name+'_out_all/')

    for param in model_ft.parameters():
        param.requires_grad = False

    sample_df = pd.read_csv("tables/sample_submission.csv")

    masks_ = sample_df.groupby("ImageId")["ImageId"].count().reset_index(name="N")
    masks_ = masks_.loc[masks_.N > 0].ImageId.values
    print(len(masks_))
    ###
    sample_df = sample_df.drop_duplicates("ImageId", keep="last").reset_index(drop=True)

    threshold = 0.5
    # thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]

    sublist = []

    for index, row in tqdm.tqdm(sample_df.iterrows(), total=len(sample_df)):
        image_id = row["ImageId"]
        if image_id in masks_:
            tta_img = []
            tta_preds = []
            img_path = os.path.join(test_dir, image_id + ".png")

            img = imread(img_path)
            # print ("data1: ",img.shape)
            width, height = img.shape[0], img.shape[1]
            img = imresize(img, (IMG_SIZE, IMG_SIZE), interp="bilinear")

            aug_HFlip = albumentations.HorizontalFlip(p=1)
            # aug_transp = albumentations.Transpose(p=1)
            augmented1 = aug_HFlip(image=img)
            # augmented2 = aug_transp(image=img)

            img1 = augmented1["image"]

            tta_img.append(img)
            tta_img.append(img1)
            inx = 0

            for img in tta_img:
                img = img[np.newaxis, np.newaxis, :, :]
                img = torch.FloatTensor(img)

                images_3chan = torch.FloatTensor(np.empty((img.shape[0], 3, img.shape[2], img.shape[3])))
                # kostyl
                # print(i, data[0].shape, images_3chan.shape)
                for chan_idx in range(3):
                    images_3chan[:, chan_idx : chan_idx + 1, :, :] = img

                img = Variable(images_3chan.cuda())

                result = model_ft(img)
                result = result.squeeze(0)
                result = result.squeeze(0)

                pred = result.data.cpu().numpy()
                if inx == 1:
                    pred = aug_HFlip(image=pred)["image"]

                pred = pred.T

                tta_preds.append(pred)
                inx += 1
            # print (pred.shape)
            pred = np.mean(tta_preds, axis=0)

            if pred.sum() > 0:
                # print (pred.min(), pred.max())

                res = transforms.ToPILImage()(pred)
                res = np.asarray(res.resize((width, height), resample=Image.BILINEAR))
                # print ("result:    ",res.min(), res.max(), res.sum(), res.shape)
                imsave(osp.join(dst_dir, image_id + "|1.png"), res)

                out_cut = np.copy(res)
                out_cut[np.nonzero(out_cut <= threshold)] = 0.0
                out_cut[np.nonzero(out_cut > threshold)] = 1.0

                out_cut, nr_objects = ndimage.label(out_cut)

                for ii in range(1, nr_objects + 1):
                    if (out_cut[out_cut == ii].sum() / ii) < SMALL_OBJ_THRESHOLD:
                        out_cut[np.nonzero(out_cut == ii)] = 0.0

                out_cut[np.nonzero(out_cut != 0)] = 1.0

                ## fill
                out_cut = ndimage.binary_fill_holes(out_cut).astype(out_cut.dtype)
                out_cut = ndimage.binary_dilation(out_cut, iterations=2).astype(out_cut.dtype)
                postproc = "_fill_dilation2"

                imsave(osp.join(dst_dir, image_id + "|2.png"), res)

                rle = mask_to_rle(out_cut, width, height)
                sublist.append([image_id, rle])

            else:
                rle = " -1"
                sublist.append([image_id, rle])
        else:
            rle = " -1"
            sublist.append([image_id, rle])

    submission_df = pd.DataFrame(sublist, columns=sample_df.columns.values)
    submission_df.to_csv(
        f"outs/fold{fold}_submission_"
        + model_name
        + "_"
        + str(IMG_SIZE)
        + "_"
        + str(threshold)
        + "_TTA_REGIONS_cut"
        + str(SMALL_OBJ_THRESHOLD)
        + postproc
        + ".csv",
        index=False,
    )
