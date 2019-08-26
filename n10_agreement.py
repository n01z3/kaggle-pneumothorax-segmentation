__author__ = "n01z3"

import matplotlib.pyplot as plt
import numpy as np


def main():
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
        plt.title(f"disagreement {values[i]/np.mean(values)}")
        plt.imshow(outputs[indexs[i], 0], cmap="gray")
        plt.subplot(4, 2, 2 + 2 * i)
        plt.title(f"disagreement mirror {values[i]/np.mean(values)}")
        plt.imshow(outputs_mirror[indexs[i], 0], cmap="gray")

    plt.show()


if __name__ == "__main__":
    main()
