import argparse
import errno
import os
import os.path as osp
import random
import warnings

import numpy as np
import torch
import torch.utils.data
from PIL import ImageFile
from torch.autograd import Variable
from tqdm import tqdm

from n02_utils import warmup_lr_scheduler, select_best_checkpoint
from n03_loss_metric import dice_coef_loss, bce_dice_loss
from n03_loss_metric import dice_coef_metric_batch as dice_coef_metric
from n03_zoo import UnetSENet154, UnetSEResNext101, UnetSEResNext50, get_hypermodel
from n04_dataset import SIIMDataset_Unet

warnings.filterwarnings("ignore", category=DeprecationWarning)

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODELS = {
    "sx50": UnetSEResNext50(),
    "sx101": UnetSEResNext101(),
    "se154": UnetSENet154(),
    "sxh50": get_hypermodel("UNetResNextHyperSE50"),
    "sxh101": get_hypermodel("UNetResNextHyperSE101"),
}
SEED = 486

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

if torch.cuda.is_available:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
else:
    print("ERROR: CUDA is not available. Exit")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, losstype="bcedice"):
    model.train()

    cntr = 0
    losses = []
    accur = []

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 100
        warmup_iters = min(100, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    progress_bar = tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Trainging", ncols=0, postfix=["dice:", "loss:"]
    )

    for i, traindata in progress_bar:
        if traindata[1].sum():
            # if 1:  # only for fine-tuning!

            images, targets = traindata[0], traindata[1]

            images_3chan = torch.FloatTensor(np.empty((images.shape[0], 3, images.shape[2], images.shape[3])))
            # print(i, data[0].shape, images_3chan.shape)
            for chan_idx in range(3):
                images_3chan[:, chan_idx : chan_idx + 1, :, :] = images
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

            if print_freq % 10 == 0:
                progress_bar.postfix[0] = f"loss: {np.mean(np.array(losses)):0.4f}"
                progress_bar.postfix[1] = f"dice: {np.mean(np.array(accur)):0.4f}"
                progress_bar.update()

            if lr_scheduler is not None:
                lr_scheduler.step()
            cntr += 1

    print("Epoch [%d]" % (epoch))
    print("Mean loss on train:", np.array(losses).mean(), "Mean DICE on train:", np.array(accur).mean())


def val_epoch(model, data_loader_valid, epoch):
    print("START validation")
    model.eval()
    cntr = 0
    valloss = 0

    threshold = 0.5

    progress_bar_valid = tqdm(
        enumerate(data_loader_valid), total=len(data_loader_valid), desc="Validation", ncols=0, postfix=["dice:"]
    )

    with torch.set_grad_enabled(False):
        for i, valdata in progress_bar_valid:
            cntr += 1
            images = valdata[0]
            targets = valdata[1]

            images_3chan = torch.FloatTensor(np.empty((images.shape[0], 3, images.shape[2], images.shape[3])))

            for chan_idx in range(3):
                images_3chan[:, chan_idx : chan_idx + 1, :, :] = images

            images = Variable(images_3chan.cuda())
            targets = Variable(targets.cuda())

            outputs = model(images)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            picloss = dice_coef_metric(out_cut, targets.data.cpu().numpy())
            valloss += picloss

        print("Epoch:  " + str(epoch) + "  Threshold:  " + str(threshold) + "  Validation DICE score:", valloss / cntr)

    return valloss / cntr


def parse_args():
    parser = argparse.ArgumentParser(description="pneumo segmentation")
    parser.add_argument("--fold", help="fold id to train", default=0, type=int)
    parser.add_argument("--net", help="net arch", default="sx101", type=str)
    parser.add_argument("--size", help="image size", default=1024, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    eps = 0.0005
    args = parse_args()

    dataset_train = SIIMDataset_Unet(mode="train", fold=args.fold, image_size=args.size)
    tloader = torch.utils.data.DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=12)

    dataset_valid = SIIMDataset_Unet(mode="valid", fold=args.fold, image_size=args.size)
    vloader = torch.utils.data.DataLoader(dataset_valid, batch_size=3, shuffle=False, num_workers=12)

    bestscore = 0.001
    device = torch.device("cuda:0")

    model_name = f"{args.net}_fold{args.fold}_best.pth"
    dst = "outs"
    os.makedirs(dst, exist_ok=True)

    model_ft = MODELS.get(args.net)
    model_ft.to(device)
    for param in model_ft.parameters():
        param.requires_grad = True

    params = [p for p in model_ft.parameters() if p.requires_grad]

    stage_epoch = [40, 15, 16]
    stage_optimizer = [
        torch.optim.Adam(params, lr=0.0001),
        torch.optim.SGD(params, lr=0.0001, momentum=0.9),
        torch.optim.Adam(params, lr=0.0001),
    ]

    stage_scheduler = [
        torch.optim.lr_scheduler.CosineAnnealingLR(stage_optimizer[0], 4, 1e-6),
        torch.optim.lr_scheduler.CyclicLR(stage_optimizer[1], base_lr=1e-5, max_lr=2e-4),
        torch.optim.lr_scheduler.CosineAnnealingLR(stage_optimizer[2], 4, 1e-6),
    ]

    for z, (num_epochs, optimizer, lr_scheduler) in enumerate(zip(stage_epoch, stage_optimizer, stage_scheduler)):
        for epoch in range(num_epochs):

            train_one_epoch(model_ft, optimizer, tloader, device, epoch, print_freq=10)
            valscore = val_epoch(model_ft, vloader, epoch)

            if valscore > bestscore - eps:
                print(f"IMPROVEMENT with delta:{(valscore - bestscore):0.6f}|on epoch{epoch}|stage{z}")
                bestscore = valscore
                torch.save(model_ft, osp.join(dst, f"{valscore:0.5f}_{model_name}"))
            lr_scheduler.step()

        checkpoint = select_best_checkpoint(dst, args.fold, args.net)
        print(f"fold{args.fold} loaded {checkpoint}")
        model_ft = torch.load(checkpoint)
