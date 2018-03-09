import argparse
import os
import datetime

import neural_network.losses.losses as losses
from neural_network.train_network import train


def main(FLAGS):
    str_list_log = []

    dpath = FLAGS.dpath
    str_list_log.append("Machine Identifier: {}".format(dpath))
    if dpath == 'cluster':
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device
        str_list_log.append("Cuda visible device name: {}".format(FLAGS.cuda_device))
        data_set = FLAGS.set
        str_list_log.append("Dataset: " + data_set)
        img_path = '/nfs/wsi/MIVC/proj1/einig/SkinCancerData/' + data_set + '/Images/*_resized.jpg'
    elif dpath == 'florence':
        pass
    elif dpath == 'julian':
        img_path = '/Users/spaethju//Desktop/Images/*_resized.jpg'
    elif dpath == 'jonas':
        img_path = 'D:/Data/Documents/AutomaticSaveToDisc/Datasets/ISIC-Archive-Downloader-master/Data/train/' \
                   'Images/*_resized.jpg'

    str_list_log.append("Dataset Image Path: {}".format(img_path))

    lossid = FLAGS.lossid
    if lossid == 'l1':
        loss_func = losses.l1_loss
    elif lossid == 'l2':
        loss_func = losses.l2_loss
    elif lossid == 'cr':
        loss_func = losses.sm_cross_loss

    str_list_log.append("Loss ID: {}".format(lossid))
    str_list_log.append("Loss function name: {}".format(loss_func.__name__))

    if dpath == 'cluster':
        save_intervals = [500, 7000]
    else:
        save_intervals = [100, 1000]

    str_list_log.append("Intervals to log network [console print , snapshot save]: {}".format(save_intervals))
    batchsize = FLAGS.bs
    str_list_log.append("Batchsize: {}".format(batchsize))
    learning_rate = FLAGS.lr
    str_list_log.append("Learning Rate: {}".format(learning_rate))

    snapshot_folder = "./neural_network/snapshots/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "/"
    str_list_log.append("Snapshotfolder: {}".format(snapshot_folder))

    if not os.path.exists(os.path.expanduser(snapshot_folder)):
        os.makedirs(os.path.expanduser(snapshot_folder))

    with open(snapshot_folder + "logfile.log", "w") as f:
        f.write('\n'.join(str_list_log))

    train(img_path=img_path, loss_func=loss_func, batch_size=batchsize, learning_rate=learning_rate,
          snapshot_folder=snapshot_folder, save_intervals=save_intervals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", default="0", help="gpu device name, only used in dpath=cluster mode")
    parser.add_argument("--dpath", default="cluster", help="data path identifier to use")
    parser.add_argument("--lossid", default="l1", help="loss function identifier to use")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--set", default="train", help="dataset to use")
    parser.add_argument("--bs", default=6, type=int, help="batch size")

    FLAGS = parser.parse_args()
    main(FLAGS)
