import tensorflow as tf
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

def main(FLAGS):
    str_path = FLAGS.path

    list_loss = []
    for summary in tf.train.summary_iterator(str_path):
        for v in summary.summary.value:
            if v.tag == 'loss':
                list_loss.append(v.simple_value)

    print(list_loss)

    str_outpath = os.path.join("/".join(str_path.split("/")[0:-1]), "loss.log")
    with open(str_outpath, "w")as f:
        for elem in list_loss:
            f.write(str(elem) + "\n")

    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    # plt.plot(list_loss, 'o')
    # plt.plot(smooth(list_loss, 3), 'r-')
    plt.plot(smooth(list_loss, 51), 'g-')

    # plt.plot(list_loss)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="", help="path of snapshot folder")
    main(parser.parse_args())
