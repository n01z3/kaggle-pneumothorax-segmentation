import numpy as np
import cv2
from albumentations import (
    HorizontalFlip,
    VerticalFlip,    
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    OneOf,
    RandomBrightnessContrast,    
    RandomGamma,
    RandomBrightness,
    RandomContrast,
    ShiftScaleRotate
)

def soft_aug(p=0.5):
    return Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.2),
        OneOf([
            RandomContrast(),
            RandomGamma(),
            RandomBrightness()
            # RandomBrightnessContrast(),
            ], p=0.3)
        ], p=p)


def strong_aug(p=0.5):
    return Compose([
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.4),
        Transpose(p=0.4),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # OneOf([
        #     ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #     GridDistortion(),
        #     OpticalDistortion(distort_limit=2, shift_limit=0.3)
        #     ], p=0.2),

        OneOf([
            RandomContrast(),
            RandomGamma(),
            RandomBrightness()
            # RandomBrightnessContrast(),
            ], p=0.3)
        ], p=p)


def strong_aug2(p=0.5):
    return Compose([
        HorizontalFlip(p=0.5),
        OneOf([
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(),
            ], p=0.5),
        Transpose(p=0.3),
        OneOf([
            RandomContrast(),
            RandomGamma(),
            RandomBrightness()
            # RandomBrightnessContrast(),
            ], p=0.4)
        ], p=p)
