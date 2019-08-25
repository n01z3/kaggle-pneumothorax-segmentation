import argparse
import errno
import os
import os.path as osp
import random
import warnings

import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import ImageFile
from torch.autograd import Variable
from tqdm import tqdm

from n02_utils import warmup_lr_scheduler
from n03_loss_metric import dice_coef_loss, bce_dice_loss
from n03_loss_metric import dice_coef_metric_batch as dice_coef_metric
from n03_zoo import UnetSEResNext101
from n04_dataset import SIIMDataset_Unet

warnings.filterwarnings("ignore", category=DeprecationWarning)

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_SIZE = 1024

seed = 486

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


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, losstype="bcedice"):
    model.train()

    cntr = 0
    losses = []
    accur = []

    # if not os.path.exists(model_name+'_trout/'):
    # 	os.mkdir(model_name+'_trout/')

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 100
        warmup_iters = min(100, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    progress_bar = tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Predicting", ncols=0, postfix=["dice:", "loss:"]
    )

    for i, traindata in progress_bar:
        if traindata[1].sum():
            # if 1:  # only for fine-tuning!

            images, targets = traindata[0], traindata[1]

            images_3chan = torch.FloatTensor(np.empty((images.shape[0], 3, images.shape[2], images.shape[3])))
            # print(i, data[0].shape, images_3chan.shape)
            for chan_idx in range(3):
                images_3chan[:, chan_idx: chan_idx + 1, :, :] = images
            # print ("train: ", images.shape, targets.shape, images_3chan.shape)

            images = Variable(images_3chan.cuda())
            targets = Variable(targets.cuda())

            outputs = model(images)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

            train_dice = dice_coef_metric(out_cut, targets.data.cpu().numpy())

            if losstype == "dice_only":
                loss = dice_coef_loss(outputs, targets)
            else:
                loss = bce_dice_loss(outputs, targets)

            losses.append(loss.item())
            accur.append(train_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if cntr % 10 == 0:
                progress_bar.postfix[0] = f"loss: {np.mean(np.array(losses)):0.4f}"
                progress_bar.postfix[1] = f"dice: {np.mean(np.array(accur)):0.4f}"
                progress_bar.update()

            if lr_scheduler is not None:
                lr_scheduler.step()
            cntr += 1

    print("Epoch [%d]" % (epoch))
    print("Mean loss on train:", np.array(losses).mean(), "Mean DICE on train:", np.array(accur).mean())


def val_epoch(model, optimizer, data_loader_valid, device, epoch):
    print("START validation")
    model.eval()
    cntr = 0
    valloss = 0

    threshold = 0.5

    progress_bar_valid = tqdm(
        enumerate(data_loader_valid), total=len(data_loader_valid), desc="Predicting", ncols=0, postfix=["dice:"]
    )

    with torch.set_grad_enabled(False):
        for i, valdata in progress_bar_valid:
            cntr += 1
            images = valdata[0]
            targets = valdata[1]
            image_ids = valdata[2]

            images_3chan = torch.FloatTensor(np.empty((images.shape[0], 3, images.shape[2], images.shape[3])))

            for chan_idx in range(3):
                images_3chan[:, chan_idx: chan_idx + 1, :, :] = images

            images = Variable(images_3chan.cuda())
            targets = Variable(targets.cuda())

            outputs = model(images)

            out_cut = np.copy(outputs.data.cpu().numpy())

            for i in range(outputs.shape[0]):
                image_id = image_ids[i]
                y_pred = out_cut[i, 0]
                image_name = osp.join('/mnt/ssd2/dataset/pneumo/predictions/sx101_fold5_val', f'{image_id}.png')

                cv2.imwrite(image_name, np.uint8(255 * y_pred))

            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            picloss = dice_coef_metric(out_cut, targets.data.cpu().numpy())
            valloss += picloss

        print("Epoch:  " + str(epoch) + "  Threshold:  " + str(threshold) + "  Validation DICE score:", valloss / cntr)

    return valloss / cntr


def mycol(x):
    return tuple(zip(*x))


def parse_args():
    parser = argparse.ArgumentParser(description="pneumo segmentation")
    parser.add_argument("--fold", help="fold id to train", default=5, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    dataset_train = SIIMDataset_Unet(mode="train", fold=args.fold)
    tloader = torch.utils.data.DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=12)

    dataset_valid = SIIMDataset_Unet(mode="valid", fold=args.fold)
    vloader = torch.utils.data.DataLoader(dataset_valid, batch_size=4, shuffle=False, num_workers=8)

    switch_grads = 1
    num_classes = 2
    bestscore = 0.001
    device = torch.device("cuda:0")

    model_name = f"UnetSEResNext101_fold{args.fold}_best.pth"
    dst = "outs"
    os.makedirs(dst, exist_ok=True)

    model_ft = UnetSEResNext101()
    model_ft.to(device)

    for param in model_ft.parameters():
        param.requires_grad = True

    params = [p for p in model_ft.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4, 1e-6)

    model_ft = torch.load(osp.join(dst, model_name))
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=2e-4)

    num_epochs = 20
    for epoch in range(num_epochs):
        valscore = val_epoch(model_ft, optimizer, vloader, device, epoch + 30)
