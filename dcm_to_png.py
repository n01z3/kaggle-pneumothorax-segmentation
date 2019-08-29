import os
import cv2
import glob2
import pydicom
from tqdm import tqdm_notebook as tqdm
import numpy as np
from PIL import Image
from skimage import exposure
from multiprocessing import Pool

OUT_SIZE = 1024
CONTRAST_CORRECT = 0
NCORE = 16

PATH_TEST = '/mnt/ssd1/dataset/pneumo/stage_2_dcm/'
PNG_TEST2 = '/mnt/ssd1/dataset/pneumo/test2_png/'

def convert_images(filename):
    ds = pydicom.read_file(filename)
    img = ds.pixel_array
    img = cv2.resize(img, (OUT_SIZE, OUT_SIZE))
    if CONTRAST_CORRECT:
        img = exposure.equalize_adapthist(img) # contrast correction
        img = ((img*255)).clip(0,255).astype(np.uint8)
    
    cv2.imwrite(PNG_TEST2 + filename.split('/')[-1][:-4] + '.png', img)


def main(test):
    # for fname in tqdm(test, total=len(test)):
    #     print (fname)
    with Pool(NCORE) as p:
        p.map(convert_images,test)

if __name__ == "__main__":
    test = glob2.glob(os.path.join(PATH_TEST, '*.dcm'))
    main(test)
