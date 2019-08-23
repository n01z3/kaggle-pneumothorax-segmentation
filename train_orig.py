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

from n02_utils import warmup_lr_scheduler
from n03_loss_metric import dice_coef_metric, dice_coef_loss, bce_dice_loss
from n03_zoo import UnetSEResNext50
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


def train_one_epoch(
    model, optimizer, data_loader, device, epoch, print_freq, losstype="bcedice"
):
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

    for i, traindata in enumerate(data_loader):
        if traindata[1].sum():
            # if 1:  # only for fine-tuning!

            images, targets = traindata[0], traindata[1]

            images_3chan = torch.FloatTensor(
                np.empty((images.shape[0], 3, images.shape[2], images.shape[3]))
            )
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
            # print ("BATCH train DICE: ", train_dice)
            # print ('output:', outputs.dtype, targets.dtype)

            # bcescore = nn.BCELoss()
            # if targets.sum() == 0:
            # 	loss = dice_coef_loss(outputs, targets, prints=1)
            # else:

            # loss = dice_coef_loss(outputs, targets)
            # loss = nn.BCELoss()(outputs, targets)

            if losstype == "dice_only":
                loss = dice_coef_loss(outputs, targets)
            else:
                loss = bce_dice_loss(outputs, targets)

            # loss = FocalLoss()(outputs, targets)

            losses.append(loss.item())
            accur.append(train_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (cntr + 1) % 400 == 0:
                print("iteration: ", cntr, " loss:", loss.item())
                losses1 = np.array(losses)
                print(
                    "Mean loss on train:",
                    losses1.mean(),
                    "Mean train DICE:",
                    np.array(accur).mean(),
                )

            if lr_scheduler is not None:
                lr_scheduler.step()
            cntr += 1

    print("Epoch [%d]" % (epoch))
    print(
        "Mean loss on train:",
        losses1.mean(),
        "Mean DICE on train:",
        np.array(accur).mean(),
    )
    # if nmbr > 10:
    # 	break


def val_epoch(model, optimizer, data_loader, device, epoch):
    print("START validation")
    model.eval()
    # cntr = 0
    val_outs = []
    val_labels = []
    # threshold = 0.1
    cntr = 0
    valloss = 0

    # thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    thresholds = [0.5]

    # if not os.path.exists(model_name+'_valout/'):
    # 	os.mkdir(model_name+'_valout/')

    # for images, targets in metric_logger.log_every(data_loader, print_freq, header):
    for threshold in thresholds:
        for i, valdata in enumerate(data_loader):
            cntr += 1
            images = valdata[0]
            targets = valdata[1]

            images_3chan = torch.FloatTensor(
                np.empty((images.shape[0], 3, images.shape[2], images.shape[3]))
            )
            # print(i, data[0].shape, images_3chan.shape)
            for chan_idx in range(3):
                images_3chan[:, chan_idx : chan_idx + 1, :, :] = images

            images = Variable(images_3chan.cuda())
            targets = Variable(targets.cuda())

            outputs = model(images)

            # picloss = dice_coef_metric(outputs, targets)
            # print ("Validation loss:", picloss.item())
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            picloss = dice_coef_metric(out_cut, targets.data.cpu().numpy())
            # print ("Validation loss:", picloss)
            valloss += picloss
            # print ("Validation loss:", picloss.item(), valloss/cntr)

            # images = images.data.cpu().numpy()
            # targets = targets.data.cpu().numpy()
            # outputs = outputs.data.cpu().numpy()

            # # print("shapes:", test.shape, gt.shape, pred.shape)
            # if (epoch+1) % 1 == 0:
            # 	for image, image1, image2 in zip(images[0], outputs[0], targets[0]):
            # 		imsave(model_name+'_valout/'+str(cntr)+'_test.png', image)
            # 		imsave(model_name+'_valout/'+str(cntr)+'_pred.png', image1)
            # 		imsave(model_name+'_valout/'+str(cntr)+'_gt.png', image2)

            # loss, bceloss, diceloss = bce_dice_loss(out_cut, val_labels)
            # print ("Sanity:", outputs.max(), targets.max())
            # print ("Sanity:", outputs.min(), targets.min())
            # print ("Validation loss:", picloss.item())
            # print ('output:', outputs.shape)
            # val_outs.append(outputs.data.cpu().numpy())
            # val_labels.append(targets.data.cpu().numpy())

        # loss on validation data #
        # val_labels = np.squeeze(val_labels)
        # val_outs = np.squeeze(val_outs)

        # out_cut = np.copy(val_outs)
        # out_cut[np.nonzero(out_cut<threshold)]=0.
        # out_cut[np.nonzero(out_cut>=threshold)]=1.

        # print ("VALID:", val_labels.max(), out_cut.max())
        # loss = dice_coef_loss(out_cut, val_labels)
        # loss, bceloss, diceloss = bce_dice_loss(out_cut, val_labels)

        print(
            "Epoch:  "
            + str(epoch)
            + "  Threshold:  "
            + str(threshold)
            + "  Validation loss:",
            valloss / cntr,
        )

    return valloss / cntr


def mycol(x):
    return tuple(zip(*x))


def parse_args():
    parser = argparse.ArgumentParser(description="pneumo segmentation")
    parser.add_argument("--fold", help="fold id to train", default=0, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    dataset_train = SIIMDataset_Unet(mode="train", fold=args.fold)
    tloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=8
    )

    dataset_valid = SIIMDataset_Unet(mode="valid", fold=args.fold)
    vloader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=1, shuffle=False, num_workers=8
    )

    switch_grads = 1
    num_classes = 2
    bestscore = 0.001
    device = torch.device("cuda:0")

    model_name = f"UnetSEResNext101_fold{args.fold}_best.pth"
    dst = "outs"
    os.makedirs(dst, exist_ok=True)

    ################################################################################################
    ################################ FROM SCRATCH ON 1024 ##########################################
    ################################################################################################

    model_ft = UnetSEResNext50()
    model_ft.to(device)

    for param in model_ft.parameters():
        param.requires_grad = True

    params = [p for p in model_ft.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4, 1e-6)

    num_epochs = 30
    for epoch in range(num_epochs):

        train_one_epoch(model_ft, optimizer, tloader, device, epoch, print_freq=100)
        valscore = val_epoch(model_ft, optimizer, vloader, device, epoch)

        if valscore > bestscore:
            bestscore = valscore
            print("SAVE BEST MODEL! Epoch: ", epoch)
            torch.save(model_ft, osp.join(dst, model_name))
        lr_scheduler.step()

    model_ft = torch.load(osp.join(dst, model_name))
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-5, max_lr=2e-4
    )

    num_epochs = 10
    for epoch in range(num_epochs):

        train_one_epoch(
            model_ft, optimizer, tloader, device, epoch + 30, print_freq=100
        )
        valscore = val_epoch(model_ft, optimizer, vloader, device, epoch + 30)

        if valscore > bestscore:
            bestscore = valscore
            print("SAVE BEST MODEL! Epoch: ", epoch + 30)
            torch.save(model_ft, osp.join(dst, model_name))
        lr_scheduler.step()

    model_ft = torch.load(osp.join(dst, model_name))
    optimizer = torch.optim.Adam(params, lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4, 1e-6)

    num_epochs = 10
    for epoch in range(num_epochs):

        train_one_epoch(
            model_ft,
            optimizer,
            tloader,
            device,
            epoch + 45,
            print_freq=100,
            losstype="dice_only",
        )
        valscore = val_epoch(model_ft, optimizer, vloader, device, epoch + 45)

        if valscore > bestscore:
            bestscore = valscore
            print("SAVE BEST MODEL! Epoch: ", epoch + 45)
            torch.save(model_ft, osp.join(dst, model_name))
        lr_scheduler.step()
