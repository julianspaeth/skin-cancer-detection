import tensorflow as tf
from PIL import Image
import os
import numpy as np

import glob
from tensorflow.contrib.slim.python.slim.nets import inception_v3
# from .nets.inception_v3 import inception_v3


def dataloader_gen():
    path = 'D:\Data\Documents\AutomaticSaveToDisc\Datasets\ISIC-Archive-Downloader-master\Data\Images\*jpg'
    os_path = os.path.expanduser(path)

    list_fns = glob.glob(os_path)
    print(len(list_fns))
    i = 0
    while (True):
        image = Image.open(list_fns[i % len(list_fns)])
        np_image = np.asarray(image)
        res = np.expand_dims(np_image, 0)
        i = i + 1
        yield res
    # img_input.show()


def load_net(x=None):
    return tf.image.resize_bilinear(images=x, size=[299, 299], name='resize')


x = tf.placeholder(dtype=tf.float32, shape=[1, 767, 1022, 3], name='input')
x_resized = load_net(x)
net, endpoints = inception_v3.inception_v3(x_resized, 2, is_training=True, dropout_keep_prob=0.8)

gen = dataloader_gen()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # saver = tf.train.Saver()
    # saver.restore(sess=sess, save_path='../neural_network/nets/weights/inception_v3.ckpt')


    for i in range(2):
        feed_dict = {x: gen.__next__()}
        result = sess.run(net, feed_dict=feed_dict)

        # print(result[1].keys())
        res_test = result.squeeze()

        print(type(res_test))
        print(res_test)

        input()
