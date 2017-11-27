import tensorflow as tf
from nets.inception_v3 import inception_v3_base
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import inception_v3

def data_gen():
    while (True):
        data = np.random.random([299, 299, 3])
        yield tf.constant(value=data)


def main():
    input = tf.placeholder(tf.uint8, [299, 299, 1], name='input')

    logits, end_points = inception_v3_base(inputs=input, scope='inception')

    result = end_points["Mixed_7c"]

    with tf.Session() as sess:
        feed_dict = {input: data_gen()}
        prediction = sess.run(result, feed_dict=feed_dict)

        print(prediction.shape)


if __name__ == '__main__':
    main()
