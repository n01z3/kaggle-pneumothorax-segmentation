__author__ = "n01z3"

import os
import warnings
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.utils.data
from albumentations import (
    HorizontalFlip,
    Compose,
    Transpose,
    RandomRotate90,
    OneOf,
    RandomGamma,
    RandomBrightness,
    RandomContrast,
    ShiftScaleRotate,
    Normalize,
)
from scipy.misc import imread, imresize
from tqdm import tqdm

from n01_config import get_paths
from n02_utils import rle2mask

warnings.filterwarnings("ignore", category=DeprecationWarning)

IMAGENET_MEAN = 0.0  # np.array([0.485, 0.456, 0.406])
IMAGENET_STD = 1.0  # np.array([0.229, 0.224, 0.225])


def strong_aug(p=0.5):
    return Compose(
        [
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.4),
            Transpose(p=0.4),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            # OneOf([
            #     ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            #     GridDistortion(),
            #     OpticalDistortion(distort_limit=2, shift_limit=0.3)
            #     ], p=0.2),
            OneOf(
                [
                    RandomContrast(),
                    RandomGamma(),
                    RandomBrightness()
                    # RandomBrightnessContrast(),
                ],
                p=0.3,
            ),
        ],
        p=p,
    )


class SIIMDataset_Unet(torch.utils.data.Dataset):
    def __init__(self, fold=0, mode="train", image_size=1024, normalized=False):
        assert mode in ("train", "valid", "test"), mode
        self.df = pd.read_csv("tables/folds_v6_st2.csv")
        if mode == "train":
            self.df = self.df[self.df["fold_id"] != fold]
        elif mode == "valid":
            self.df = self.df[self.df["fold_id"] == fold]
        else:
            self.df = pd.read_csv("tables/stage_2_sample_submission.csv")
            self.df[" EncodedPixels"] = ["-1"] * self.df.shape[0]

        print(self.df.head())
        print(f"{mode} {self.df.shape[0]}")

        self.gb = self.df.groupby("ImageId")
        self.fnames = list(self.gb.groups.keys())

        paths = get_paths()

        self.paths = paths
        self.height = image_size
        self.width = image_size
        self.image_dir = os.path.join(paths["dataset"]["path"], paths["dataset"]["images_dir"])
        self.mask_dir = os.path.join(paths["dataset"]["path"], paths["dataset"]["masks_dir"])
        if mode == "test":
            self.image_dir = os.path.join(paths["dataset"]["path"], paths["dataset"]["test_dir"])

        self.augs = False
        if mode == "train":
            self.augs = True
        self.transform = strong_aug()
        self.cache = False
        self.norm_transform = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.normalized = normalized

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        df = self.gb.get_group(image_id)
        annotations = df[" EncodedPixels"].tolist()
        img = imread(os.path.join(self.image_dir, f"{image_id}.png"))
        width, height = img.shape[0], img.shape[1]
        if width != self.width:
            img = imresize(img, (self.width, self.height), interp="bilinear")

        mask = np.zeros((width, height))
        annotations = [item.strip() for item in annotations]
        if annotations[0] != "-1":
            # if self.cache:
            #     mask = imread(os.path.join(self.image_dir, f'{image_id}.png'))
            # else:
            for rle in annotations:
                mask += rle2mask(rle, width, height).T

        if width != self.width:
            mask = imresize(mask, (self.width, self.height), interp="bilinear").astype(float)

        mask = (mask >= 1).astype("float32")  # for overlap cases
        mask_save = np.uint8(255 * mask)
        cv2.imwrite(
            os.path.join(self.paths["dataset"]["path"], self.paths["dataset"]["masks_dir"], f"{image_id}.png"),
            mask_save,
        )

        if self.augs:
            augmented = self.transform(image=img, mask=mask)

            img = augmented["image"]
            mask = augmented["mask"]

        if self.normalized:
            img = self.norm_transform(image=img)["image"]

        img = img[np.newaxis, :, :]
        mask = mask[np.newaxis, :, :]

        return torch.FloatTensor(img), torch.FloatTensor(mask), image_id

    def __len__(self):
        return len(self.fnames)


def check_dataset():
    batch_size = 9
    side = int(1.2 * batch_size ** 0.5)
    mode = "train"

    for i in range(10):
        dataset = SIIMDataset_Unet(mode=mode, fold=1, image_size=1024, normalized=True)
        vloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        progress_bar = tqdm(
            enumerate(vloader),
            total=len(vloader),
            desc="Predicting",
            ncols=0,
            postfix=["DICE:", dict(value=0), "loss:", dict(value=0)],
        )

        t0 = time()
        for n, data in progress_bar:
            images, masks, ids = data
            images = images.numpy()
            masks = masks.numpy()

            if n % 20 == 0:
                progress_bar.postfix[1] = n / 2
                progress_bar.postfix[3] = n * 2
                progress_bar.update()
            print(images.shape, masks.shape)
            plt.figure(figsize=(25, 35))
            for i in range(batch_size):
                plt.subplot(side, side, i + 1)
                plt.imshow(images[i, 0], cmap="gray")

                plt.title(f"area:{np.sum(masks[i])}")
                plt.imshow(masks[i, 0], alpha=0.5, cmap="Reds")
            plt.show()

        print(time() - t0)


if __name__ == "__main__":
    check_dataset()
