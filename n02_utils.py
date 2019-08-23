import numpy as np
import pandas as pd
import os
import glob
import sys
import tqdm
from collections import defaultdict, deque
import datetime

# import pickle
import time
import torch.distributed as dist
import errno
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data as D
from PIL import Image, ImageFile
from scipy.misc import imread, imresize, imsave

import collections
import torchvision
from torchvision import transforms
import random


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


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
