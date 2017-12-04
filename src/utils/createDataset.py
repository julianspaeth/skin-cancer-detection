import argparse
import glob
import os


def main(args):
    # # Jonas image path:
    # D:\Data\Documents\AutomaticSaveToDisc\Datasets\ISIC - Archive - Downloader - master\Data\Images

    list_fns = glob.glob(os.path.expanduser(args.datapath))

    out_path = "../datasets"
    if args.o is not None:
        out_path = os.path.expanduser(args.o)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    out_path = out_path + "/" + args.name + ".dataset"
    with open(out_path, 'w') as f:
        for fn in list_fns:
            f.write(fn + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='name of the new data set')
    parser.add_argument('datapath', help='glob regex path of the data for the new data set')
    parser.add_argument('-o', help='output path for the new data set')

    main(parser.parse_args())
