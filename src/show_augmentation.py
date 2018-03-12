from PIL import Image
import os
import tensorflow as tf
import numpy as np
from neural_network.image_tools.augmentations import random_brightness
from neural_network.image_tools.preprocess import preprocess
from neural_network.image_tools.augmentations import random_contrast
from neural_network.image_tools.augmentations import random_horizontal_flip
from neural_network.image_tools.augmentations import random_hue
from neural_network.image_tools.augmentations import random_rotation
from neural_network.image_tools.augmentations import random_vertical_flip
from neural_network.image_tools.augmentations import random_saturation

img_path = "./ISIC_0000003_resized.jpg"

log_folder = "disp_augmentations"
if not os.path.exists(os.path.expanduser(log_folder)):
    os.makedirs(os.path.expanduser(log_folder))

img = Image.open(os.path.expanduser(path=img_path))
np_img = np.asarray(img, dtype = np.uint8)
np_img = np.expand_dims(np_img, axis=0)
x_prep = tf.placeholder(dtype=tf.float32, shape=[1, 542, 718, 3], name='input')
x = x_prep
# x_prep = 1
sess = tf.Session()
for i in range(10):
    np_x_prep = sess.run(x_prep, {x: np_img})
    Image.fromarray(np.squeeze(np_x_prep).astype(dtype=np.uint8)).save(os.path.join(log_folder, "preprocessed_"+str(i)+".png"))

    augmentation_brightness = random_brightness(x_prep, 50, random=False, percentage=100)
    np_aug_bright = sess.run(augmentation_brightness, {x: np_img})
    Image.fromarray(np.squeeze(np_aug_bright).astype(dtype=np.uint8)).save(os.path.join(log_folder, "brightness_"+str(i)+".png"))

    augmentation_sat = random_saturation(x_prep, 50, random=False, percentage=100)
    np_aug_sat = sess.run(augmentation_sat, {x: np_img})
    Image.fromarray(np.squeeze(np_aug_sat).astype(dtype=np.uint8)).save(os.path.join(log_folder, "saturation_"+str(i)+".png"))

    augmentation_contrast = random_contrast(x_prep, 50, random=False, percentage=100)
    np_aug_cont = sess.run(augmentation_contrast, {x: np_img})
    Image.fromarray(np.squeeze(np_aug_cont).astype(dtype=np.uint8)).save(os.path.join(log_folder, "contrast_"+str(i)+".png"))

    augmentation_v_flip = random_vertical_flip(x_prep, random=False, percentage=100)
    np_aug_vflip = sess.run(augmentation_v_flip, {x: np_img})
    Image.fromarray(np.squeeze(np_aug_vflip).astype(dtype=np.uint8)).save(os.path.join(log_folder, "vertical_flip_"+str(i)+".png"))

    augmentation_hue = random_hue(x_prep, 50, random=False, percentage=100)
    np_aug_hue = sess.run(augmentation_hue, {x: np_img})
    Image.fromarray(np.squeeze(np_aug_hue).astype(dtype=np.uint8)).save(os.path.join(log_folder, "hue_"+str(i)+".png"))

    augmentation_rotation = random_rotation(x_prep, random=False, percentage=100)
    np_aug_rot = sess.run(augmentation_rotation, {x: np_img})
    Image.fromarray(np.squeeze(np_aug_rot).astype(dtype=np.uint8)).save(os.path.join(log_folder, "rotation_"+str(i)+".png"))

    augmentation_h_flip= random_horizontal_flip(x_prep, random=False, percentage=100)
    np_aug_hflip= sess.run(augmentation_h_flip, {x: np_img})
    Image.fromarray(np.squeeze(np_aug_hflip).astype(dtype=np.uint8)).save(os.path.join(log_folder, "horizontal_flip_"+str(i)+".png"))
