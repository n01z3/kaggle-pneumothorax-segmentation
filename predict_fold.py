import numpy as np
import pandas as pd
import os
import glob
import pydicom
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
from pretrained_models import *
import albumentations

IMG_SIZE = 1024

# from mask_functions import rle2mask, mask2rle
ImageFile.LOAD_TRUNCATED_IMAGES = True

def rle2mask(rle, width, height):
	mask= np.zeros(width* height)
	array = np.asarray([int(x) for x in rle.split()])
	starts = array[0::2]
	lengths = array[1::2]

	current_position = 0
	for index, start in enumerate(starts):
		current_position += start
		mask[current_position:current_position+lengths[index]] = 1
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
			currentPixel+=1
	return " " + " ".join(rle)


if __name__ == "__main__":

	device = torch.device('cuda:0')
	SMALL_OBJ_THRESHOLD = 2000

	model_name = 'UnetSEResNext101'

	sample_df = pd.read_csv("input/sample_submission.csv")

	masks_ = sample_df.groupby('ImageId')['ImageId'].count().reset_index(name='N')
	masks_ = masks_.loc[masks_.N > 0].ImageId.values
	print (len(masks_))
	###
	sample_df = sample_df.drop_duplicates('ImageId', keep='last').reset_index(drop=True)

	total_preds = np.zeros([len(sample_df),1024,1024])

	for fold in range(5):
		model_ft = torch.load('outs/'+model_name+'_'+str(IMG_SIZE)+'_fold'+str(fold)+'.pt')
		model_ft.to(device)
		model_ft.eval()

		for param in model_ft.parameters():
			param.requires_grad = False

		threshold_list = [0.5] #0.6 ? (this one for TTA)

		for threshold in threshold_list:
			sublist = []
			for index, row in tqdm.tqdm(sample_df.iterrows(), total=len(sample_df)):
				image_id = row['ImageId']
				if image_id in masks_:
					tta_img = []
					tta_preds = []
					img_path = os.path.join('input/test_png', image_id + '.png')

					img = imread(img_path)
					# print ("data1: ",img.shape)
					width, height = img.shape[0], img.shape[1]
					img = imresize(img, (IMG_SIZE, IMG_SIZE), interp='bilinear')
					# print ("data2: ",img.shape)


					aug_HFlip = albumentations.HorizontalFlip(p=1)
					# aug_Rot90 = albumentations.Transpose(p=1)
					augmented1 = aug_HFlip(image=img)
					# augmented2 = aug_Rot90(image=img)

					img1 = augmented1['image']
					# img2 = augmented2['image']

					tta_img.append(img)
					tta_img.append(img1)
					# tta_img.append(img2)
					inx = 0
					# print ("data2: ",img.shape)
					for img in tta_img:
						img = img[np.newaxis,np.newaxis,:,:]
						img = torch.FloatTensor(img)

						# print ("img: ",img.shape)

						images_3chan = torch.FloatTensor(np.empty((img.shape[0],3,img.shape[2],img.shape[3])))
						# print(i, data[0].shape, images_3chan.shape)
						for chan_idx in range(3):
							images_3chan[:,chan_idx:chan_idx+1,:,:]=img

						img = Variable(images_3chan.cuda())
						
						result = model_ft(img)
						result = result.squeeze(0)
						result = result.squeeze(0)

						pred = result.data.cpu().numpy()
						if inx == 1:
							pred = aug_HFlip(image=pred)['image']
						# elif inx == 2:
						# 	pred = aug_Rot90(image=pred)['image']

						pred = pred.T

						tta_preds.append(pred)
						inx += 1
					# print (pred.shape)
					pred = np.mean(tta_preds, axis=0)
					
					out_cut = np.copy(pred)
					res = transforms.ToPILImage()(out_cut)
					res = np.asarray(res.resize((1024, 1024), resample=Image.BILINEAR))

					total_preds[index,:,:] += res

	if not os.path.exists(model_name+'_out_all_folds/'):
		os.makedirs(model_name+'_out_all_folds/')


	threshold_list = [1.4]
	# threshold_list = [1.2, 1.4, 1.5, 2]
	# threshold = 1.5

	for threshold in threshold_list:
		
		sublist = []
		for index, row in tqdm.tqdm(sample_df.iterrows(), total=len(sample_df)):
			image_id = row['ImageId']
			pred = total_preds[index]
			# print (pred.shape)

			if pred.sum() > 0:

				out_cut = np.copy(pred)
				out_cut[np.nonzero(out_cut<=threshold)]=0.
				out_cut[np.nonzero(out_cut>threshold)]=1.

				# once again - maybe something appeared again #not really
				out_cut, nr_objects = ndimage.label(out_cut)

				for ii in range(1, nr_objects+1):
					if (out_cut[out_cut==ii].sum() / ii) < SMALL_OBJ_THRESHOLD:
						out_cut[np.nonzero(out_cut==ii)] = 0.

				out_cut[np.nonzero(out_cut!=0)] = 1.
				out_cut = ndimage.binary_fill_holes(out_cut).astype(out_cut.dtype)
				out_cut = ndimage.binary_dilation(out_cut, iterations=2).astype(out_cut.dtype)

				imsave(model_name+'_out_all_folds/'+image_id + '.png', out_cut)

				rle = mask_to_rle(out_cut, 1024, 1024)
				sublist.append([image_id, rle])

			else:
				rle = " -1"
				sublist.append([image_id, rle])

			submission_df = pd.DataFrame(sublist, columns=sample_df.columns.values)
			submission_df.to_csv('outs/submission_'+model_name+'_'+str(IMG_SIZE)+'_allfolds_'+str(threshold)+'_fill_dill_2.csv', index=False)
