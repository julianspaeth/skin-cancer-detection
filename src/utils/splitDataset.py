import numpy as np
import argparse
import glob
import os

# splits a dataset to training (50%), test (25%) and validation (25%)

def main(args):

    # TODO paths
    out_path_training = '/Users/spaethju/Desktop/training.dataset'
    out_path_test = '/Users/spaethju/Desktop/test.dataset'
    out_path_validation = '/Users/spaethju/Desktop/validation.dataset'

    input_list = glob.glob(os.path.expanduser(args.datapath))
    lists = []
    for l in input_list:
        lists.append(l.split("/")[-1])

    x = np.asarray(lists, dtype=np.string_)
    split_index = int(len(x)/2)

    np.random.shuffle(x)
    training, rest = x[:split_index], x[split_index:]

    rest_split_index = int(len(rest)/2)
    np.random.shuffle(rest)
    test, validation = rest[:rest_split_index], rest[rest_split_index:]

    with open(out_path_training, 'w') as f:
        for fn in training:
            f.write(str(fn).replace("b", "").replace("'", "") + "\n")

    with open(out_path_test, 'w') as f:
        for fn in test:
            f.write(str(fn).replace("b", "").replace("'", "") + "\n")

    with open(out_path_validation, 'w') as f:
        for fn in validation:
            f.write(str(fn).replace("b", "").replace("'", "") + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', help='glob regex path of the data for the new data set')
    main(parser.parse_args())
