import argparse
import os

import neural_network.training.losses as losses
from neural_network.train_simple import train
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(FLAGS):
    dpath = FLAGS.dpath
    if dpath == 'cluster':
        img_path = '/data/scratch/einig/SkinCancerData/train/Images/*_resized.jpg'
    elif dpath == 'florence':
        pass
    elif dpath == 'julian':
        img_path = '/Users/spaethju//Desktop/Images/*_resized.jpg'
    elif dpath == 'jonas':
        img_path = 'D:/Data/Documents/AutomaticSaveToDisc/Datasets/ISIC-Archive-Downloader-master/Data/Images/*_resized.jpg'

    lossid = FLAGS.lossid
    if lossid == 'l1':
        loss_func = losses.l1_loss
    elif lossid == 'l2':
        loss_func = losses.l2_loss
    elif lossid == 'cr':
        loss_func = losses.sm_cross_loss

    if dpath == 'cluster':
        save_intervals = [500, 7000]
    else:
        save_intervals = [100, 1000]
    batchsize = FLAGS.bs
    learning_rate = FLAGS.lr

    train(img_path=img_path, loss_func=loss_func, batch_size=batchsize, learning_rate=learning_rate, save_intervals=save_intervals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpath", default="cluster", help="datapath identifier to use")
    parser.add_argument("--lossid", default="l1", help="loss function identifier to use")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--bs", default=6, type=int, help="batch size")

    FLAGS = parser.parse_args()
    main(FLAGS)
