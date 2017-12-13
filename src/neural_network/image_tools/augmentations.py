"""
Image augmentations applied that should be only applied to the training data
"""

import tensorflow as tf
from random import *


def random_rotate(images):
    """
    Rotates the images randomly 0 to 3 times by 90 degrees
    :param images: TODO
    :return: Rotated images
    """
    number_of_images = images.shape[0]
    rotated_images = []
    for i in range(number_of_images):
        image = images[i, :, :, :]
        number_of_rotations = randint(0,3)
        rotated_image = tf.image.rot90(image, number_of_rotations)
        rotated_images.append(rotated_image)
    images = tf.stack(rotated_images)
    return images


def random_flip_vertically(images):
    """
    Flips an image vertically (upside down) with a 50:50 chance
    :param images: TODO
    :return: Flipped images
    """
    number_of_images = images.shape[0]
    flipped_vertically_images = []
    for i in range(number_of_images):
        image = images[i, :, :, :]
        flipped_vertically_image = tf.image.random_flip_up_down(image)
        flipped_vertically_images.append(flipped_vertically_image)
    images = tf.stack(flipped_vertically_images)
    return images


def random_flip_horizontally(images):
    """
    Flips an image horizontally (right to left) with a 50:50 chance
    :param images: TODO
    :return: Flipped images
    """
    number_of_images = images.shape[0]
    flipped_horizontally_images = []
    for i in range(number_of_images):
        image = images[i, :, :, :]
        flipped_horizontally_image = tf.image.random_flip_left_right(image)
        flipped_horizontally_images.append(flipped_horizontally_image)
    images = tf.stack(flipped_horizontally_images)
    return images

# More Ideas: https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9
