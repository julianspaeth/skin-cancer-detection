import os

import shutil

dataset_local_path = "/data/scratch/einig/SkinCancerData"


prefix_test = '/test'
dataset_path_test = '../datasets/test.dataset'

if not os.path.exists(dataset_local_path + prefix_test):
    os.makedirs(dataset_local_path + prefix_test)

with open(os.path.expanduser(dataset_path_test)) as f:
    for line in f:
        src_path = dataset_local_path + "/Images/" + line.strip() + "_resized.jpg"
        dest_path = dataset_local_path + prefix_test + "/Images/" + line.strip() + "_resized.jpg"

        shutil.move(src=src_path, dst=dest_path)

        src_path = dataset_local_path + "/Descriptions/" + line.strip()
        dest_path = dataset_local_path + prefix_test + "/Descriptions/" + line.strip()

        shutil.move(src=src_path, dst=dest_path)

        exit(0)




