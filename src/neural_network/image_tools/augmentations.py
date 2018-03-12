"""
Image augmentations applied that should be only applied to the training data
"""

import tensorflow as tf


def random_rotation(images, random, percentage):
    """
    Rotates the images randomly 0 to 3 times by 90 degrees
    :param images: tensor of images NHWC
    :return: Rotated images
    """

    def augment(images):
        number_of_images = images.shape[0]
        rotated_images = []
        for i in range(number_of_images):
            image = images[i, :, :, :]
            # number_of_rotations = randint(0, 3)
            rotated_image = tf.image.rot90(image, tf.random_uniform([], dtype=tf.int32, minval=0, maxval=3))
            rotated_images.append(rotated_image)
        return tf.stack(rotated_images)

    def no_augment(images):
        return images

    if random:
        return augment(images)

    tf_flag = tf.less(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1),
                      tf.constant((percentage / 100), dtype=tf.float32))
    images = tf.cond(pred=tf_flag, true_fn=lambda: augment(images), false_fn=lambda: no_augment(images))
    return images


def random_vertical_flip(images, random, percentage):
    """
    Flips the images vertically (upside down) with a 50:50 chance
    :param images: tensor of images NHWC
    :return: Flipped images
    """

    def augment(images):
        number_of_images = images.shape[0]
        flipped_vertically_images = []
        for i in range(number_of_images):
            image = images[i, :, :, :]
            flipped_vertically_image = tf.image.random_flip_up_down(image)
            flipped_vertically_images.append(flipped_vertically_image)
        return tf.stack(flipped_vertically_images)

    def no_augment(images):
        return images

    if random:
        return augment(images)

    tf_flag = tf.less(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1),
                      tf.constant((percentage / 100), dtype=tf.float32))
    images = tf.cond(pred=tf_flag, true_fn=lambda: augment(images), false_fn=lambda: no_augment(images))
    return images


def random_horizontal_flip(images, random, percentage):
    """
    Flips the images horizontally (right to left) with a 50:50 chance
    :param images: tensor of images NHWC
    :return: Flipped images
    """

    def augment(images):
        number_of_images = images.shape[0]
        flipped_horizontally_images = []
        for i in range(number_of_images):
            image = images[i, :, :, :]
            flipped_horizontally_image = tf.image.random_flip_left_right(image)
            flipped_horizontally_images.append(flipped_horizontally_image)
        return tf.stack(flipped_horizontally_images)

    def no_augment(images):
        return images

    if random:
        return augment(images)

    tf_flag = tf.less(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1),
                      tf.constant((percentage / 100), dtype=tf.float32))
    images = tf.cond(pred=tf_flag, true_fn=lambda: augment(images), false_fn=lambda: no_augment(images))
    return images


def random_brightness(images, max_delta, random, percentage):
    """ 
    Sets a random brightness to the images with a delta of +-max_delta
    :param images: tensor of images NHWC
    :param max_delta: Delta randomly picked in the interval [-max_delta, max_delta]
    :return: Images with new brightness
    """

    def augment(images):
        number_of_images = images.shape[0]
        brightness_images = []
        for i in range(number_of_images):
            image = images[i, :, :, :]
            brightness_image = tf.image.random_brightness(image, max_delta / 100)
            brightness_images.append(brightness_image)
        return tf.stack(brightness_images)

    def no_augment(images):
        return images

    if random:
        return augment(images)

    tf_flag = tf.less(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1),
                      tf.constant((percentage / 100), dtype=tf.float32))
    images = tf.cond(pred=tf_flag, true_fn=lambda: augment(images), false_fn=lambda: no_augment(images))
    return images


def random_contrast(images, max_delta, random, percentage):
    """
    Sets a random contrast to the images with a delta of +-max_delta
    :param images: tensor of images NHWC
    :param max_delta: Delta randomly picked in the interval [-max_delta, max_delta]
    :return: Images with new contrast
    """

    def augment(images):
        number_of_images = images.shape[0]
        contrast_images = []
        for i in range(number_of_images):
            image = images[i, :, :, :]
            contrast_image = tf.image.random_contrast(image, (1 - (max_delta / 100)), (1 + (max_delta / 100)))
            contrast_images.append(contrast_image)
        return tf.stack(contrast_images)

    def no_augment(images):
        return images

    if random:
        return augment(images)

    tf_flag = tf.less(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1),
                      tf.constant((percentage / 100), dtype=tf.float32))
    images = tf.cond(pred=tf_flag, true_fn=lambda: augment(images), false_fn=lambda: no_augment(images))
    return images


def random_hue(images, max_delta, random, percentage):
    """
    Sets a random hue to the images with a delta of +-max_delta
    :param images: tensor of images NHWC
    :param max_delta: Delta randomly picked in the interval [-max_delta, max_delta]
    :return: Images with new hue
    """

    def augment(images):
        number_of_images = images.shape[0]
        hue_images = []
        for i in range(number_of_images):
            image = images[i, :, :, :]
            hue_image = tf.image.random_hue(image, max_delta / 100)
            hue_images.append(hue_image)
        return tf.stack(hue_images)

    def no_augment(images):
        return images

    if random:
        return augment(images)

    tf_flag = tf.less(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1),
                      tf.constant((percentage / 100), dtype=tf.float32))
    images = tf.cond(pred=tf_flag, true_fn=lambda: augment(images), false_fn=lambda: no_augment(images))
    return images


def random_saturation(images, max_delta, random, percentage):
    """
    Sets a random contrast to the images with a delta of +-max_delta
    :param images: tensor of images NHWC
    :param max_delta: Delta randomly picked in the interval [-max_delta, max_delta]
    :return: Images with new saturation
    """

    def augment(images):
        number_of_images = images.shape[0]
        saturation_images = []
        for i in range(number_of_images):
            image = images[i, :, :, :]
            saturation_image = tf.image.random_saturation(image, (1 - (max_delta / 100)), (1 + (max_delta / 100)))
            saturation_images.append(saturation_image)
        return tf.stack(saturation_images)

    def no_augment(images):
        return images

    if random:
        return augment(images)

    tf_flag = tf.less(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1),
                      tf.constant((percentage / 100), dtype=tf.float32))
    images = tf.cond(pred=tf_flag, true_fn=lambda: augment(images), false_fn=lambda: no_augment(images))
    return images


def augment(images,
            random=True,
            percentage=30,
            rotation=None,
            vertical_flip=None,
            horizontal_flip=None,
            brightness=None,
            contrast=None,
            hue=None,
            saturation=None):
    """
    Performs an augmentation to the images
    :param images:  tensor of images NHWC
    :param random: bool use random or not
    :param percentage:
    :param rotation:
    :param vertical_flip:
    :param horizontal_flip:
    :param brightness: Double between 0 and 1 for the +- delta range of the random brightness
    :param contrast: Double between 0 and 1 for the +- delta range of the random contrast
    :param hue: Double between 0 and 1 for the +- delta range of the random hue
    :param saturation: Double between 0 and 1 for the +- delta range of the random saturation
    :return: Augmented images
    """
    if rotation:
        images = random_rotation(images, random, percentage)

    if vertical_flip:
        images = random_vertical_flip(images, random, percentage)

    if horizontal_flip:
        images = random_horizontal_flip(images, random, percentage)

    if brightness:
        images = random_brightness(images, brightness, random, percentage)

    if contrast:
        images = random_contrast(images, contrast, random, percentage)

    if hue:
        images = random_hue(images, hue, random, percentage)

    if saturation:
        images = random_saturation(images, saturation, random, percentage)

    return images

# More Ideas: https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9
