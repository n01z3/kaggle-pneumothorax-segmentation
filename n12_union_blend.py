__author__ = "bobbqe"

import os

import numpy as np
import pandas as pd
import tqdm
from scipy import ndimage

from n02_utils import mask_to_rle, rle2mask


def main():
    os.makedirs("outs/", exist_ok=True)
    model_name = "sx50_sx101_se154"

    sample_df = pd.read_csv("tables/sample_submission.csv")
    sample_df = sample_df.drop_duplicates("ImageId", keep="last").reset_index(drop=True)

    masks_ = sample_df.groupby("ImageId")["ImageId"].count().reset_index(name="N")
    ###

    total_preds = np.zeros([len(sample_df), 1024, 1024])

    subs = os.listdir("subm/corr1")

    pred_dict = {}

    for jj in range(len(subs)):
        print(subs[jj])
        df = pd.read_csv(f"subm/corr1/{subs[jj]}")

        for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            image_id = row["ImageId"]
            annot = row["EncodedPixels"].strip()

            if annot != "-1":
                mask = rle2mask(annot, 1024, 1024)
            else:
                mask = np.zeros((1024, 1024))

            if image_id in pred_dict:
                pred_dict[image_id] += mask
            else:
                pred_dict[image_id] = mask
        del df

    print(len(pred_dict))

    #################################################################

    threshold_list = [0, 1]  # 0 - if union, 1 - if with certainty of 2 for many models

    for threshold in threshold_list:
        sublist = []
        for index, row in tqdm.tqdm(sample_df.iterrows(), total=len(sample_df)):
            image_id = row["ImageId"]
            pred = pred_dict[image_id]

            if pred.sum() > 0:

                out_cut = np.copy(pred)
                out_cut[np.nonzero(out_cut <= threshold)] = 0.0
                out_cut[np.nonzero(out_cut > threshold)] = 1.0

                out_cut = ndimage.binary_fill_holes(out_cut).astype(out_cut.dtype)

                # imsave(model_name+'/'+image_id + '.png', out_cut)

                rle = mask_to_rle(out_cut, 1024, 1024)
                sublist.append([image_id, rle])
            else:
                rle = " -1"
                sublist.append([image_id, rle])

            submission_df = pd.DataFrame(sublist, columns=sample_df.columns.values)
            submission_df.to_csv(f"subm/union_submission_corr1_{model_name}_{threshold}.csv", index=False)


if __name__ == "__main__":
    main()