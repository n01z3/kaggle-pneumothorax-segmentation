import numpy as np
import pandas as pd
import os
import glob
import sys
import tqdm
import shutil
from tqdm import tqdm_notebook
import datetime
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data as D
from PIL import Image, ImageFile
from scipy.misc import imread, imresize, imsave

import torchvision
from torchvision import transforms
import random
from sklearn.model_selection import KFold

from pretrained_models import UnetSEResNext101, UnetSEResNext50, UnetSENet154
from augs import soft_aug, strong_aug, strong_aug2

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_SIZE = 512

seed = 486

random.seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)
np.random.seed(seed)

if torch.cuda.is_available:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
else:
    print ("ERROR: CUDA is not available. Exit")
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True


def dice_coef_metric(inputs, target):

	intersection = 2. * (target * inputs).sum()
	union = target.sum() + inputs.sum()

	if target.sum() == 0 and inputs.sum() == 0:
		return 1.0
		
	return (intersection / union)

def dice_coef_loss(inputs, target, prints=0):
	smooth = 1.
	if prints:
		print ("inp and tar:", inputs.min(), inputs.max(), target.min(), target.max())

	intersection = 2. * ((target * inputs).sum()) + smooth
	union = target.sum() + inputs.sum() + smooth
	if prints:
		print ("intersection and union:", intersection, union)
	return 1 - (intersection / union)


def bce_dice_loss(inputs, target):
	dicescore = dice_coef_loss(inputs, target)
	bcescore = nn.BCELoss()
	bceloss = bcescore(inputs, target)

	return bceloss + dicescore


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




def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

	def f(x):
		if x >= warmup_iters:
			return 1
		alpha = float(x) / warmup_iters
		return warmup_factor * (1 - alpha) + alpha

	return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class SIIMDataset_Unet(torch.utils.data.Dataset):
	def __init__(self, df_path, img_dir, augmentations):
		self.df = pd.read_csv(df_path)
		self.gb = self.df.groupby('ImageId')
		self.fnames = list(self.gb.groups.keys())

		self.height = IMG_SIZE
		self.width = IMG_SIZE
		self.image_dir = img_dir
		self.augs = augmentations

	def __getitem__(self, idx):
		image_id = self.fnames[idx]
		df = self.gb.get_group(image_id)
		annotations = df[' EncodedPixels'].tolist()
		image_path = os.path.join(self.image_dir, image_id + ".png")
		img = imread(image_path)
		width, height = img.shape[0], img.shape[1]
		if width != self.width:
			img = imresize(img, (self.width, self.height), interp='bilinear')

		mask = np.zeros((self.width,self.height))
		annotations = [item.strip() for item in annotations]
		if annotations[0] != '-1':
			for rle in annotations:
				mask_orig = rle2mask(rle, width, height).T
			if width != self.width:
				mask_orig = imresize(mask_orig, (self.width, self.height), interp='bilinear').astype(float)
			mask += mask_orig

		mask = (mask >= 1).astype('float32')

		if self.augs:
			aug = strong_aug()
			augmented = aug(image=img, mask=mask)

			img = augmented['image']
			mask = augmented['mask']
		
		# mask = mask / 255.
		img = img[np.newaxis,:,:]
		mask = mask[np.newaxis,:,:]

		return torch.FloatTensor(img), torch.FloatTensor(mask)


	def __len__(self):
		return len(self.fnames)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, losstype='bcedice'):
	model.train()

	cntr = 0
	losses = []
	accur = []

	lr_scheduler = None
	if epoch == 0:
		warmup_factor = 1. / 100
		warmup_iters = min(100, len(data_loader) - 1)

		lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

	for i, traindata in enumerate(data_loader):
		if (traindata[1].sum()):
		
			images, targets = traindata[0], traindata[1]

			images_3chan = torch.FloatTensor(np.empty((images.shape[0],3,images.shape[2],images.shape[3])))
			# kostyl
			# print(i, data[0].shape, images_3chan.shape)
			for chan_idx in range(3):
				images_3chan[:,chan_idx:chan_idx+1,:,:]=images
			# print ("train: ", images.shape, targets.shape, images_3chan.shape)


			images = Variable(images_3chan.cuda())
			targets = Variable(targets.cuda())

			outputs = model(images)

			out_cut = np.copy(outputs.data.cpu().numpy())
			out_cut[np.nonzero(out_cut<0.5)]=0.
			out_cut[np.nonzero(out_cut>=0.5)]=1.

			train_dice = dice_coef_metric(out_cut, targets.data.cpu().numpy())
			
			if losstype == 'dice_only':
				loss = dice_coef_loss(outputs, targets)
			else:
				loss = bce_dice_loss(outputs, targets)

			losses.append(loss.item())
			accur.append(train_dice)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (cntr+1)% 200 == 0:
				print('iteration: ', cntr, ' loss:', loss.item())
				losses1 = np.array(losses)
				print("Mean loss on train:", losses1.mean(), "Mean train DICE:", np.array(accur).mean())

			if lr_scheduler is not None:
				lr_scheduler.step()
			cntr += 1
	
	print("Epoch [%d]" % (epoch))
	print("Mean loss on train:", losses1.mean(), "Mean DICE on train:", np.array(accur).mean())


def val_epoch(model, optimizer, data_loader, device, epoch):

	print("START validation")
	model.eval()

	cntr = 0
	valloss = 0

	# thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
	thresholds = [0.5]

	for threshold in thresholds:
		for i, valdata in enumerate(data_loader):
			cntr += 1
			images = valdata[0]
			targets = valdata[1]

			images_3chan = torch.FloatTensor(np.empty((images.shape[0],3,images.shape[2],images.shape[3])))
			# kostyl
			# print(i, data[0].shape, images_3chan.shape)
			for chan_idx in range(3):
				images_3chan[:,chan_idx:chan_idx+1,:,:]=images

			images = Variable(images_3chan.cuda())
			targets = Variable(targets.cuda())

			outputs = model(images)

			out_cut = np.copy(outputs.data.cpu().numpy())
			out_cut[np.nonzero(out_cut<threshold)]=0.
			out_cut[np.nonzero(out_cut>=threshold)]=1.

			picloss = dice_coef_metric(out_cut, targets.data.cpu().numpy())
			# print ("Validation loss:", picloss)
			valloss += picloss
			# print ("Validation loss:", picloss.item(), valloss/cntr)

		print ("Epoch:  "+str(epoch)+"  Threshold:  "+str(threshold)+"  Validation loss:", valloss/cntr)

	return valloss/cntr


if __name__ == "__main__":
	ds_train = SIIMDataset_Unet("/mnt/ssd1/dataset/pneumothorax_data/train-rle.csv", "/mnt/ssd1/dataset/pneumothorax_data/dataset1024/train/", augmentations=1)
	train_length=int(0.9* len(ds_train))
	val_length=len(ds_train)-train_length

	dataset_train, dataset_val = D.random_split(ds_train, (train_length, val_length))

	bestscore = 0.001
	device = torch.device('cuda:0')

	model_name = 'UnetSEResNext101'

	if not os.path.exists('outs/'):
		os.makedirs('outs/')

	train_loader = torch.utils.data.DataLoader(
	dataset_train, batch_size=2, shuffle=True, num_workers=4)

	val_loader = torch.utils.data.DataLoader(
	dataset_val, batch_size=1, shuffle=False, num_workers=4)

	model_ft = UnetSEResNext101()
	model_ft.to(device)

	for param in model_ft.parameters():
		param.requires_grad = True

	params = [p for p in model_ft.parameters() if p.requires_grad]
	optimizer = torch.optim.Adam(params, lr=0.0001)
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4, 1e-6)

	num_epochs = 30
	for epoch in range(num_epochs):

		train_one_epoch(model_ft, optimizer, train_loader, device, epoch, print_freq=100)
		valscore = val_epoch(model_ft, optimizer, val_loader, device, epoch)

		if valscore > bestscore: 
			bestscore=valscore
			print ("SAVE BEST MODEL! Epoch: ", epoch)
			torch.save(model_ft, 'outs/'+model_name+'_'+str(IMG_SIZE)+'_bcedice.pt')
		lr_scheduler.step()

	model_ft = torch.load('outs/'+model_name+'_'+str(IMG_SIZE)+'_bcedice.pt')
	optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9)
	lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=2e-4)

	num_epochs = 10
	for epoch in range(num_epochs):

		train_one_epoch(model_ft, optimizer, train_loader, device, epoch+30, print_freq=100)
		valscore = val_epoch(model_ft, optimizer, val_loader, device, epoch+30)

		if valscore > bestscore: 
			bestscore=valscore
			print ("SAVE BEST MODEL! Epoch: ", epoch+30)
			torch.save(model_ft, 'outs/'+model_name+'_'+str(IMG_SIZE)+'_bcedice.pt')
		lr_scheduler.step()

