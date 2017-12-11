import numpy as np
from PIL import Image
import glob
import os

path = 'D:\Data\Documents\AutomaticSaveToDisc\Datasets\ISIC-Archive-Downloader-master\Data\Images\*jpg'

images = glob.glob(os.path.expanduser(path))

def get_image_sizes(images):
    widths = []
    heights = []
    for image in images:
        image = Image.open(image)
        width, height = image.size
        widths.append(width)
        heights.append(height)

    widths = np.unique(widths)
    heights = np.unique(heights)

    print("widths:")
    for width in widths:
        print(width)

    print("heights:")
    for height in heights:
        print(height)

get_image_sizes(images)