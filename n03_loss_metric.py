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
from torch import nn
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data as D
from PIL import Image, ImageFile
from scipy.misc import imread, imresize, imsave

import collections
import torchvision
from torchvision import transforms
import random


def dice_coef_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union


def dice_coef_loss(inputs, target, prints=0):
    smooth = 1.0
    if prints:
        print("inp and tar:", inputs.min(), inputs.max(), target.min(), target.max())

    intersection = 2.0 * ((target * inputs).sum()) + smooth
    union = target.sum() + inputs.sum() + smooth
    if prints:
        print("intersection and union:", intersection, union)
    return 1 - (intersection / union)


def bce_dice_loss(inputs, target):
    dicescore = dice_coef_loss(inputs, target)
    bcescore = nn.BCELoss()
    bceloss = bcescore(inputs, target)

    return bceloss + dicescore


class SOTALoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = dice_coef_loss

    def forward(self, outputs, targets):
        bce_loss = self.bce(outputs, targets)
        dice_loss = self.dice(outputs, targets)
        return bce_loss + dice_loss
