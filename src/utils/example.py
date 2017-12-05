import tensorflow as tf
from PIL import Image
import os
import numpy as np

import glob
from tensorflow.contrib.slim.python.slim.nets import inception_v3

def dataloader_gen():
    path = '/Users/spaethju/Desktop/*jpg'
    os_path = os.path.expanduser(path)

    list_fns = glob.glob(os_path)
    print(len(list_fns))
    while(True):
        image = Image.open(list_fns[0])
        np_image = np.asarray(image)
        res = np.expand_dims(np_image, 0)
        yield res
       # img_input.show()


def load_net(x=None):
    return tf.image.resize_bilinear(images=x, size=[ 500, 500], name='resize')

x = tf.placeholder(dtype=tf.float32, shape=[1, 767, 1022, 3])
net = inception_v3.inception_v3_base([2,767,1022,3], )

gen = dataloader_gen()

with tf.Session() as sess:

    for i in range(2):
        feed_dict = {x: gen.__next__()}
        result = sess.run(net, feed_dict=feed_dict)

        print(result.shape)
        res_test = result.squeeze()
        print(type(res_test))
        img_res = Image.fromarray(res_test.astype(np.uint8))
        img_res.show()
        input()




