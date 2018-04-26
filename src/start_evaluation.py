import argparse
import os

from neural_network.evaluation import evaluate


def main(FLAGS):
    str_list_log = []

    dpath = FLAGS.dpath
    str_list_log.append("Machine Identifier: {}".format(dpath))
    if dpath == 'cluster':
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device
        str_list_log.append("Cuda visible device name: {}".format(FLAGS.cuda_device))
        data_set = FLAGS.set
        type = FLAGS.type
        str_list_log.append("Type: {}".format(type))
        img_path = '/nfs/wsi/MIVC/proj1/einig/SkinCancerData/' + type + "/" + data_set + '/Images/*_resized.jpg'
        print(img_path)
    elif dpath == 'florence':
        pass
    elif dpath == 'julian':
        img_path = '/Users/spaethju//Desktop/Images/*_resized.jpg'
    elif dpath == 'jonas':
        data_set = FLAGS.set
        img_path = 'D:/Data/Documents/AutomaticSaveToDisc/Datasets/ISIC-Archive-Downloader-master/Data/' + data_set + '/Images/*_resized.jpg'

    str_list_log.append("Datase Image Path: {}".format(img_path))

    snapshot_path = "./neural_network/snapshots/" + type + "/"
    print(snapshot_path)

    eval_all = FLAGS.all
    if eval_all:
        snapshots = [os.path.join(snapshot_path, o) for o in os.listdir(snapshot_path)
                     if os.path.isdir(os.path.join(snapshot_path, o))]
    else:
        snapshots = [os.path.join(snapshot_path, o) for o in os.listdir(snapshot_path)
                     if os.path.isdir(os.path.join(snapshot_path, o)) and not os.path.exists(
                os.path.join(os.path.join(snapshot_path, o), 'evaluation'))]

    print(snapshots)
    for snap in snapshots:
        # evaluate
        verbose = FLAGS.v
        evaluate(img_path=img_path, snapshot_folder=snap,
                 eval_path=os.path.join(snap, 'evaluation'), verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", default="0", help="gpu device name, only used in dpath=cluster mode")
    parser.add_argument("--dpath", default="cluster", help="datapath identifier to use")
    parser.add_argument("--type", default="50-25-25", help="50-25-25 or 60-20-20 or 80-10-10")
    parser.add_argument("--set", default="val", help="dataset to use: train, test or val")
    parser.add_argument("--all", default=False, type=bool, help="if set all snapshots are evaluated again")
    parser.add_argument("--v", default=False, type=bool, help="verbose output")

    FLAGS = parser.parse_args()
    main(FLAGS)
