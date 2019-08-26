__author__ = "n01z3"

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np


def eda():
    filename = "/mnt/hdd2/learning_dumps/pneumo/predictions/0_sx50_test_index50.npz"
    tfz = np.load(filename)
    outputs, outputs_mirror, ids, gts = tfz["outputs"], tfz["outputs_mirror"], tfz["ids"], tfz["gts"]
    # ids = np.array(all_ids), gts = np.array(gts)

    print(outputs_mirror.shape, outputs.shape, ids.shape, gts.shape)
    all_pixels = []
    all_pixels = np.array([outputs_mirror, outputs]).flatten()

    # plt.hist(all_pixels, bins=300)
    # plt.yscale('log')
    # plt.show()

    values = []
    for i in range(outputs.shape[0]):
        # print(np.array([outputs[i], outputs_mirror[i]]))
        mean_probs = np.mean(np.array([outputs[i], outputs_mirror[i]]), axis=0)

        area = np.sum(mean_probs > 0.5)
        if area > 0:
            print(mean_probs.shape)
            dist1 = np.mean(np.abs(mean_probs - outputs[i]))
            dist2 = np.mean(np.abs(mean_probs - outputs_mirror[i]))

            values.append((dist1 + dist2) / np.mean(mean_probs > 0.5) * 1e8)
            if dist1 != dist2:
                print(dist1, dist2)
        else:
            values.append(0)

    print(sorted(values)[::-1])

    indexs = np.argsort(np.array(values))[::-1]

    index = 4
    plt.figure()
    for i in range(4):
        plt.subplot(4, 2, 1 + 2 * i)
        plt.title(f"disagreement {values[i] / np.mean(values)}")
        plt.imshow(outputs[indexs[i], 0], cmap="gray")
        plt.subplot(4, 2, 2 + 2 * i)
        plt.title(f"disagreement mirror {values[i] / np.mean(values)}")
        plt.imshow(outputs_mirror[indexs[i], 0], cmap="gray")

    plt.show()


def score_select_top(fold=1, dst="folder"):
    eps = 1e-7
    y_preds = []
    for model in ["sx50", "sx101"]:
        tfz = np.load(
            f"/mnt/ssd2/dataset/pneumo/predictions/uint8/{model}/{fold}_{model}_valid/{fold}_{model}_valid_index.npz"
        )
        for key in ["outputs", "outputs_mirror"]:
            y_preds.append(tfz[key])

        ids = tfz["ids"]
        y_trues = tfz["gts"]

    print(y_trues.shape)
    y_preds = np.array(y_preds)
    print(y_preds.shape)

    mean_masks = np.mean(y_preds, axis=0)
    print(mean_masks.shape)

    scores = []
    for i in range(4):
        scores.append(np.abs(mean_masks - y_preds[i]).sum(axis=-1).sum(axis=-1).sum(axis=-1))

    scores = np.sum(np.array(scores), axis=0)
    areas = np.array(mean_masks > 128.0).sum(axis=-1).sum(axis=-1).sum(axis=-1)

    scores /= areas + eps
    scores[areas == 0] = 0

    print(scores.shape)
    indexs = np.argsort(scores)[::-1]

    names = ["sx50", "sx50m", "sx101", "sx101m"]

    top_index = indexs[: int(len(indexs) * 0.2)]
    top_preds = y_preds[:, top_index, 0]
    top_gts = y_trues[top_index]
    top_ids = ids[top_index]

    np.savez_compressed(
        osp.join(dst, f"{fold}_top20"), top_preds=top_preds, top_gts=top_gts, top_ids=top_ids, desc=names
    )

    # amt = 10
    # plt.figure()
    # for i in range(amt):
    #     for j in range(4):
    #         plt.subplot(6, amt, 1 + i + amt * j)
    #         plt.title(f"sample{i} {names[j]}")
    #         plt.imshow(y_preds[j, indexs[i], 0], cmap="gray")
    #
    #     plt.subplot(6, amt, 1 + i + amt * 4)
    #     plt.title(f"sample{i} mean")
    #     plt.imshow(mean_masks[indexs[i], 0], cmap="gray")
    #
    #     plt.subplot(6, amt, 1 + i + amt * 5)
    #     plt.title(f"sample{i} gt")
    #     plt.imshow(y_trues[indexs[i]], cmap="gray")
    #
    # plt.show()


def main():
    dst = "/mnt/ssd2/dataset/pneumo/predictions/top_disagree"
    os.makedirs(dst, exist_ok=True)
    for fold in range(8):
        score_select_top(fold=fold, dst=dst)


def read_top_disagre():
    fold = 0

    tfz = np.load(f"/mnt/ssd2/dataset/pneumo/predictions/top_disagree/{fold}_top20.npz")
    y_preds, y_gts, ids, names = tfz["top_preds"], tfz["top_gts"], tfz["top_ids"], tfz["desc"]
    print(y_preds.shape)  # (n_aug_net, n_samples, wight, height)
    print(y_gts.shape)  # (n_samples, wight, height)
    print(ids[:10])
    print(names)


if __name__ == "__main__":
    # score_select_top()
    read_top_disagre()
