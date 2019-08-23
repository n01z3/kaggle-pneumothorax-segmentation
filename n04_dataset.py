import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import pickle
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
)
from scipy.misc import imread, imresize
from tqdm import tqdm

from n01_config import get_paths
from n02_utils import rle2mask


def strong_aug(p=0.5):
    return Compose(
        [
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.4),
            Transpose(p=0.4),
            ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2
            ),
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
    def __init__(self, fold=0, mode="train", image_size=1024):
        assert mode in ("train", "valid", "test"), mode
        self.df = pd.read_csv("tables/folds_v5.csv")
        if mode == "train":
            self.df = self.df[self.df["fold_id"] != fold]
        elif mode == "valid":
            self.df = self.df[self.df["fold_id"] != fold]
        else:
            self.df = pd.read_csv("tables/test.csv")

        self.gb = self.df.groupby("ImageId")
        self.fnames = list(self.gb.groups.keys())

        paths = get_paths()

        self.height = image_size
        self.width = image_size
        self.image_dir = os.path.join(
            paths["dataset"]["path"], paths["dataset"]["images_dir"]
        )
        if mode == "test":
            self.image_dir = os.path.join(
                paths["dataset"]["path"], paths["dataset"]["test_dir"]
            )
        self.augs = False
        if mode == "train":
            self.augs = True
        self.transform = strong_aug()

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        df = self.gb.get_group(image_id)
        annotations = df[" EncodedPixels"].tolist()
        image_path = os.path.join(self.image_dir, image_id + ".png")
        img = imread(image_path)
        width, height = img.shape[0], img.shape[1]
        if width != self.width:
            img = imresize(img, (self.width, self.height), interp="bilinear")

        mask = np.zeros((self.width, self.height))
        annotations = [item.strip() for item in annotations]
        if annotations[0] != "-1":
            for rle in annotations:
                mask += rle2mask(rle, width, height).T
            if width != self.width:
                mask = imresize(
                    mask, (self.width, self.height), interp="bilinear"
                ).astype(float)

        mask = (mask >= 1).astype("float32")  # for overlap cases

        if self.augs:
            augmented = self.transform(image=img, mask=mask)

            img = augmented["image"]
            mask = augmented["mask"]

        img = img[np.newaxis, :, :]
        mask = mask[np.newaxis, :, :]

        return torch.FloatTensor(img), torch.FloatTensor(mask)

    def __len__(self):
        return len(self.fnames)


def check_dataset():
    batch_size = 9
    side = int(1.2 * batch_size ** 0.5)
    mode = 'valid'

    dataset = SIIMDataset_Unet(mode=mode)
    tloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    progress_bar = tqdm(enumerate(tloader), total=len(tloader), desc='Predicting', ncols=0)

    for n, data in progress_bar:
        images, masks = data
        images = images.numpy()
        masks = masks.numpy()
        print(images.shape, masks.shape)
        plt.figure(figsize=(25, 35))
        for i in range(batch_size):
            plt.subplot(side, side, i + 1)
            plt.imshow(images[i, 0], cmap='gray')

            plt.title(f'area:{np.sum(masks[i])}')
            plt.imshow(masks[i, 0], alpha=0.5, cmap='Reds')
        plt.show()


if __name__ == '__main__':
    check_dataset()
