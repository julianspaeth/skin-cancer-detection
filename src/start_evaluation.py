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

        img_path = '/data/scratch/einig/SkinCancerData/train/Images/*_resized.jpg'
    elif dpath == 'florence':
        pass
    elif dpath == 'julian':
        img_path = '/Users/spaethju//Desktop/Images/*_resized.jpg'
    elif dpath == 'jonas':
        img_path = 'D:/Data/Documents/AutomaticSaveToDisc/Datasets/ISIC-Archive-Downloader-master/Data/Images/*_resized.jpg'

    str_list_log.append("Datase Image Path: {}".format(img_path))

    snapshot_path = "./neural_network/snapshots/"
    snapshots = [os.path.join(snapshot_path, o) for o in os.listdir(snapshot_path)
                 if os.path.isdir(os.path.join(snapshot_path, o)) and not os.path.exists(
            os.path.join(os.path.join(snapshot_path, o), 'evaluation'))]

    print(snapshots)
    for snap in snapshots:
        # evaluate

        evaluate(img_path=img_path, snapshot_folder=snap,
                 eval_path=os.path.exists(os.path.join(snap, 'evaluation')))

        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
        parser.add_argument("--cuda_device", default="0", help="gpu device name, only used in dpath=cluster mode")
        parser.add_argument("--dpath", default="cluster", help="datapath identifier to use")

        FLAGS = parser.parse_args()
        main(FLAGS)
