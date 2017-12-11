import tensorflow as tf


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
