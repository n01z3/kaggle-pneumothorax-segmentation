import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import cv2

from n01_config import get_paths

PATHS = get_paths()
image_dir = osp.join(PATHS["dataset"]["path"], PATHS["dataset"]["test_dir"])


def main():
    masks = "/mnt/ssd2/dataset/pneumo/predictions/uint8/mean/study"

    filenames = sorted(glob(osp.join(masks, "*png")))
    print(filenames)

    dfns = []

    plt.figure()
    for i, fn in enumerate(filenames):
        # print(i)

        plt.title(f"{i}")
        image = cv2.imread(osp.join(image_dir, osp.basename(fn)), 0)
        mask = cv2.imread(fn, 0)

        canvas = np.hstack([image, image])
        plt.imshow(canvas, cmap="gray")

        masks = np.hstack([np.zeros(image.shape, np.uint8), mask])

        plt.imshow(masks, alpha=0.5, cmap="Reds")
        plt.axis("off")

    print(dfns)

    plt.show()


if __name__ == "__main__":
    main()
