import tensorflow as tf


def normalize(images):
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


def crop(images):
    # crops an image to the maximum central square
    height = images.shape[1]
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
    # resizes an image to 299x299
    resized_images = tf.image.resize_bilinear(images=images, size=[299, 299], name='resize')
    return resized_images
