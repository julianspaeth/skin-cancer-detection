"""
Image preprocessing that should be applied to both training and test data
"""

import tensorflow as tf


def normalize(images):
    """
    Normalizes the data by ImageNet Normalization
    :param images:
    :return: normalized images
    """
    number_of_images = images.shape[0]
    normalized_images = []
    for i in range(number_of_images):
        image = images[i, :, :, :]
        image = tf.cast(image, dtype=tf.float32)
        mean = tf.convert_to_tensor([104.00698793, 116.66876762, 122.67891434], dtype=tf.float32)
        normalized_image = tf.subtract(image, mean)
        normalized_images.append(normalized_image)
    images = tf.stack(normalized_images)
    return images


"""
    def image_net_normalization(images):
    # gets tuple of rgb image and groundtruth
    data = images
    img_data = data[0]

    # switch to BGR necessary?
    img_data = img_data[:, :, ::-1]
    img_data = tf.cast(img_data, dtype=tf.float32)
    mean = tf.convert_to_tensor([104.00698793, 116.66876762, 122.67891434], dtype=tf.float32)
    img_data = tf.subtract(img_data, mean)
    return img_data  # , data[1]
"""


def crop(images):
    """
    Crops the images to its maximum central square
    :param images: TODO
    :return: cropped images
    """
    height = tf.constant(images.shape[1])
    width = images.shape[2]
    number_of_images = images.shape[0]
    cropped_images = []
    crop_size = min(height, width)
    for i in range(number_of_images):
        image = images[i, :, :, :]
        offset = (width - height) // 2
        cropped_image = tf.image.crop_to_bounding_box(image, 0, offset, crop_size, crop_size)
        cropped_images.append(cropped_image)
    images = tf.stack(cropped_images)
    return images


def resize(images):
    """
    Resizes images to a size of 299x299
    :param images:
    :return: resized images
    """
    resized_images = tf.image.resize_bilinear(images=images, size=[299, 299], name='resize')
    return resized_images
