__author__ = "n01z3"

import argparse
import gc
import os
import os.path as osp
from multiprocessing import Pool

import cv2
import numpy as np
import torch
from tqdm import tqdm

from n01_config import get_paths
from n03_loss_metric import dice_coef_metric
from n04_dataset import SIIMDataset_Unet

DEVICE = torch.device("cuda:0")
from torch.autograd import Variable

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def save_img(data):
    filename, image = data
    cv2.imwrite(filename, image)


def calc_score(data):
    y_true, y_pred = data
    return dice_coef_metric(y_pred >= 128, y_true > 0.5)


@torch.no_grad()
def predict_fold(model_name, fold=0, mode="valid", out_folder="outs", weights_dir="outs", validate=True):
    assert mode in ("train", "valid", "test"), mode

    model_ft = torch.load(osp.join(weights_dir, f"{model_name}_fold{fold}_best.pth"))
    model_ft.to(DEVICE)
    model_ft.eval()

    dst = osp.join(out_folder, f"{fold}_{model_name}_{mode}")
    os.makedirs(dst, exist_ok=True)

    dataset_valid = SIIMDataset_Unet(mode=mode, fold=fold)
    vloader = torch.utils.data.DataLoader(dataset_valid, batch_size=2, shuffle=False, num_workers=4)

    progress_bar = tqdm(enumerate(vloader), total=len(vloader), desc=f"Predicting {mode} {fold}")

    outputs, gts, filenames, all_ids = [], [], [], []
    for i, batch in progress_bar:
        images, targets, ids = batch

        mirror = torch.flip(images, (3,))
        batch1ch = torch.cat([images, mirror], dim=0)
        batch3ch = torch.FloatTensor(np.empty((batch1ch.shape[0], 3, batch1ch.shape[2], batch1ch.shape[3])))
        for chan_idx in range(3):
            batch3ch[:, chan_idx: chan_idx + 1, :, :] = batch1ch
        images = Variable(batch3ch.cuda())
        targets = targets.data.cpu().numpy()

        preictions = model_ft(images)
        probability = preictions.data.cpu().numpy()

        for j in range(targets.shape[0]):
            probabilityTTA = np.mean(
                np.concatenate([probability[0 + j], probability[targets.shape[0] + j][:, :, ::-1]], axis=0), axis=0
            )
            # outputs.append(np.uint8(255 * probabilityTTA))
            outputs.append(probabilityTTA)
            filenames.append(osp.join(dst, f"{ids[j]}.png"))
            all_ids.append(ids[j])
            gts.append(np.array(targets[j, 0] > 0.5))

        if i % 50:
            gc.collect()

    np.savez(dst + '.npz', outputs=np.array(outputs), ids=np.array(all_ids), gts=np.array(gts))

    # with Pool() as p:
    #     list(
    #         tqdm(
    #             p.imap_unordered(save_img, zip(filenames, outputs)),
    #             total=len(filenames),
    #             desc="saving predictions to image",
    #         )
    #     )
    # p.close()

    with Pool() as p:
        scores = list(tqdm(p.imap_unordered(calc_score, zip(gts, outputs)), total=len(filenames), desc="calc score"))
    p.close()

    score = np.mean(scores)
    print(f"\n{model_name} fold{fold} {score:0.4f}\n")
    return score


def parse_args():
    parser = argparse.ArgumentParser(description="pneumo segmentation")
    parser.add_argument("--fold", help="fold id to predict", default=0, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    paths = get_paths()
    weights_dir = osp.join(paths['dumps']['path'], paths['dumps']['weights'])
    dumps_dir = osp.join(paths['dumps']['path'], paths['dumps']['predictions'])

    scores = []
    for mode in ['test', 'valid']:
        if args.fold >= 0:
            lst = [args.fold]
        else:
            lst = list(range(8))

        for fold in lst:
            score = predict_fold(
                "sx50",
                fold=fold,
                mode=mode,
                out_folder=dumps_dir,
                weights_dir=weights_dir,
            )
            scores.append(score)
    print(scores[:10])
    print(scores[-10:])


if __name__ == "__main__":
    main()
