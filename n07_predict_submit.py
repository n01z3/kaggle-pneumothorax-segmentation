import numpy as np
import pandas as pd
import os
import glob
import sys
import tqdm
from tqdm import tqdm_notebook
from PIL import Image, ImageFile
from scipy.misc import imread, imresize, imsave
from scipy import ndimage

from n04_dataset import SIIMDataset_Unet
from n03_loss_metric import dice_coef_metric_batch
from n02_utils import rle2mask, mask_to_rle

SMALL_OBJ_THRESHOLD = 2000
BATCH_SIZE = 4

warnings.filterwarnings("ignore", category=DeprecationWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

if torch.cuda.is_available:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
else:
    print("ERROR: CUDA is not available. Exit")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def flip(x, dim):
	xsize = x.size()
	dim = x.dim() + dim if dim < 0 else dim
	x = x.view(-1, *xsize[dim:])
	x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
		-1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
	return x.view(xsize)


def val_epoch(model, data_loader_valid, threshold=0.5):
    print("START validation")
    cntr = 0
    valloss = 0

    

    progress_bar_valid = tqdm(
        enumerate(data_loader_valid),
        total=len(data_loader_valid),
        desc="Predicting",
        ncols=0,
        postfix=["dice:"],
    )

    with torch.set_grad_enabled(False):
        for i, valdata in progress_bar_valid:
            cntr += 1
            images = valdata[0]
            targets = valdata[1]

            images_3chan = torch.FloatTensor(
                np.empty((images.shape[0], 3, images.shape[2], images.shape[3]))
            )

            for chan_idx in range(3):
                images_3chan[:, chan_idx: chan_idx + 1, :, :] = images

            images = Variable(images_3chan.cuda())
            targets = Variable(targets.cuda())

            #predict 
            outputs = model(images)

            #hflip
			aug_images = flip(images,3)
			#predict TTA
            aug_outputs = model(aug_images)
            #hflip back to orig
            aug_outputs = flip(aug_outputs,3)

			mean_outs = torch.mean(torch.stack([outputs, aug_outputs], dim=1),dim=1)

            out_cut = np.copy(mean_outs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            picloss = dice_coef_metric_batch(out_cut, targets.data.cpu().numpy())
            valloss += picloss

        print(
            "  Threshold:  "
            + str(threshold)
            + "  Validation DICE score:",
            valloss / cntr,
        )

    return valloss / cntr

def make_predictions(model, data_loader_test, threshold=0.5, model_name, fold):
    print("START predictions")
    
	sublist = []

    progress_bar_valid = tqdm(
        enumerate(data_loader_test),
        total=len(data_loader_test)
    )

    with torch.set_grad_enabled(False):
        for i, valdata in progress_bar_valid:
            images = valdata[0]
            image_ids = valdata[2]

            images_3chan = torch.FloatTensor(
                np.empty((images.shape[0], 3, images.shape[2], images.shape[3]))
            )

            for chan_idx in range(3):
                images_3chan[:, chan_idx: chan_idx + 1, :, :] = images

            images = Variable(images_3chan.cuda())

            #predict 
            outputs = model(images)

            #hflip
			aug_images = flip(images,3)
			#predict TTA
            aug_outputs = model(aug_images)
            #hflip back to orig
            aug_outputs = flip(aug_outputs,3)

			mean_outs = torch.mean(torch.stack([outputs, aug_outputs], dim=1),dim=1)

            out_cut = np.copy(mean_outs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            for jj in range(BATCH_SIZE):
            	one_cut = out_cut[jj,0,::]
            	image_id = image_ids[jj]

				one_cut, nr_objects = ndimage.label(one_cut)

				for ii in range(1, nr_objects+1):
					if (one_cut[one_cut==ii].sum() / ii) < SMALL_OBJ_THRESHOLD:
						one_cut[np.nonzero(one_cut==ii)] = 0.

				one_cut[np.nonzero(one_cut!=0)] = 1.

				## fill
				one_cut = ndimage.binary_fill_holes(one_cut).astype(one_cut.dtype)
				one_cut = ndimage.binary_dilation(one_cut, iterations=2).astype(one_cut.dtype)

				if one_cut.sum() > 0:
					rle = mask_to_rle(one_cut, 1024, 1024)
				else:
					rle = " -1"
				sublist.append([image_id, rle])

		submission_df = pd.DataFrame(sublist, columns=['ImageId', 'EncodedPixels'])
		submission_df.to_csv(f"submission_{model_name}_fold{fold}_best_threshold_{threshold}.pth", index=False)		

    return 0

def parse_args():
    parser = argparse.ArgumentParser(description="pneumo segmentation")
    parser.add_argument("--fold", help="fold id to train", default=0, type=int)
    parser.add_argument("--model_name", help="model name", default='sx101', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
	args = parse_args()
	device = torch.device("cuda:0")

    dataset_valid = SIIMDataset_Unet(mode="valid", fold=args.fold)
    vloader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=4, shuffle=False, num_workers=8
    )

    dataset_test = SIIMDataset_Unet(mode="test", fold=args.fold)
    testloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=4, shuffle=False, num_workers=8
    )

    model_dict = f"{args.model_name}_fold{args.fold}_best.pth"
    dst = "outs"
    os.makedirs(dst, exist_ok=True)

	model_ft = torch.load(model_dict)
	model_ft.to(device)
	model_ft.eval()    

### TODO : make more precise step after first check around best value
	thresholds = np.linspace(0.05, 0.95, num=19) 

	val_dice_list = {}
	
	for threshold in thresholds:
		val_dice_list[threshold]=val_epoch(model_ft, vloader, threshold)

	best_val_th = max(val_dice_list, key=val_dice_list.get)
	print ("Best threshold: ", best_val_th)
	print (" with val DICE score: ", val_dice_list[best_val_th])

	make_predictions(model_ft, testloader, best_val_th, args.model_name, args.fold)

