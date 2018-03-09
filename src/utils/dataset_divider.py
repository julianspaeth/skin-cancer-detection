import os

import shutil

# dataset_local_path = "/data/scratch/einig/SkinCancerData"
# dataset_local_path = "D:\Data\Documents\AutomaticSaveToDisc\Datasets\ISIC-Archive-Downloader-master\Data"
dataset_local_path = "/nfs/wsi/MIVC/proj1/einig/SkinCancerData/all_images"
dataset_local__output_path = "/nfs/wsi/MIVC/proj1/einig/SkinCancerData/60_20_20"
# dataset_local__output_path = "/nfs/wsi/MIVC/proj1/einig/SkinCancerData/80_10_10"

# prefix = '/train'
# dataset_path = '../datasets/60-20-20/training.dataset'
# dataset_path = '../datasets/80-10-10/training.dataset'

prefix = '/val'
dataset_path = '../datasets/60-20-20/validation.dataset'
# dataset_path = '../datasets/80-10-10/validation.dataset'
# dataset_path = '../datasets/validation.dataset'

# prefix = '/test'
# dataset_path = '../datasets/60-20-20/test.dataset'
# dataset_path = '../datasets/80-10-10/test.dataset'
# dataset_path = '../datasets/test.dataset'

if not os.path.exists(dataset_local_path + prefix):
    os.makedirs(dataset_local__output_path + prefix)

if not os.path.exists(dataset_local_path + prefix + "/Images"):
    os.makedirs(dataset_local__output_path + prefix + "/Images")

if not os.path.exists(dataset_local_path + prefix + "/Descriptions"):
    os.makedirs(dataset_local__output_path + prefix + "/Descriptions")

with open(os.path.expanduser(dataset_path)) as f:
    for line in f:
        src_path = dataset_local_path + "/Images/" + line.strip() + "_resized.jpg"
        dest_path = dataset_local__output_path + prefix + "/Images/" + line.strip() + "_resized.jpg"

        try:
            shutil.copy(src=src_path, dst=dest_path)
        except:
            print("following file was not found: {}".format(src_path) )

        src_path = dataset_local_path + "/Descriptions/" + line.strip()
        dest_path = dataset_local__output_path + prefix + "/Descriptions/" + line.strip()

        try:
            shutil.copy(src=src_path, dst=dest_path)
        except:
            print("following file was not found: {}".format(src_path) )






