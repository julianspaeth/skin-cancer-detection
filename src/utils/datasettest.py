import glob
import os
import numpy as np
import json

from PIL import Image


def dataloader(list_fns_img, batch_size=1):
    print(len(list_fns_img))

    benign = 0
    malign = 0
    other = 0

    i = 0
    for img_fn in list_fns_img:
        res = []
        lesion_classes = np.zeros([batch_size, 2])
        for j in range(batch_size):
            single_img_path =img_fn.replace("\\", "/")

            fn_name = "_".join(single_img_path.split('/')[-1].split("_")[0: 2])
            json_single_img_path = "/".join(single_img_path.split('/')[0: -2]) + "/Descriptions/" + fn_name

            # # IMAGE
            # image = Image.open(single_img_path)
            # np_image = np.asarray(image)
            #
            # if np_image.shape[0] > np_image.shape[1]:
            #     np_image = np.rot90(np_image, axes=(-3, -2))
            #
            # res.append(np_image)

            # JSON
            json_file = json.load(open(json_single_img_path))

            # search for the lesion class
            clinical_class = json_file["meta"]["clinical"]["benign_malignant"]

            if clinical_class == "benign":
                benign = benign + 1
                lesion_classes[j, 0] = 1

            elif clinical_class == "malignant":
                malign = malign + 1
                lesion_classes[j, 1] = 1
            else:
                other = other + 1

            print(i)
            i = i + 1

        print("Malignant: {}".format(malign))
        print("Benign: {}".format(benign))
        print("Other: {}".format(other))
        # yield res, lesion_classes


img_path = 'D:/Data/Documents/AutomaticSaveToDisc/Datasets/ISIC-Archive-Downloader-master/Data/train/Images/*_resized.jpg'

list_fns_img = glob.glob(os.path.expanduser(img_path))
int_image_files = len(list_fns_img)

dataloader(list_fns_img=list_fns_img)
