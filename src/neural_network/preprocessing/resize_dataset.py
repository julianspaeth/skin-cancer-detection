import numpy as np
from PIL import Image

import glob
import os

path = 'D:\Data\Documents\AutomaticSaveToDisc\Datasets\ISIC-Archive-Downloader-master\Data\Images\*jpg'

images = glob.glob(os.path.expanduser(path))
final_height = 542
final_width = 718
def resize_all_dataset_images(images):
    i = 0

    for fn_image in images:
        print(i)
        i = i+1
        image = Image.open(fn_image)
        width, height = image.size
        if width > height:
            image = image.resize([final_width, final_height], resample=Image.BILINEAR)

        else:
            image = image.resize([ final_height, final_width], resample=Image.BILINEAR)

        image.save(fn_image.split('.')[0] + '_resized.jpg')
resize_all_dataset_images(images)