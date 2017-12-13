from PIL import Image
import glob
import os


def resize_all_dataset_images(path, final_height, final_width):
    """
    Resizes the images to a certain height and width and saves them as a copy in the same directory
    :param path: Path of all images
    :param final_height: Height after resize
    :param final_width: Width after resize
    """
    images = glob.glob(os.path.expanduser(path))
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


image_path = 'D:\Data\Documents\AutomaticSaveToDisc\Datasets\ISIC-Archive-Downloader-master\Data\Images\*jpg'
resize_height = 542
resize_width = 718
resize_all_dataset_images(image_path, resize_height, resize_width)
