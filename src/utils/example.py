import tensorflow as tf
from PIL import Image
import os
import numpy as np
import json

import glob
from tensorflow.contrib.slim.python.slim.nets import inception_v3

def dataloader_gen():
    #img_path = '/Users/spaethju/Desktop/*jpg'
    #json_path = '/Users/spaethju/Desktop/*'

    img_path = '/Users/florencelopez/Desktop/Archiv/images/*jpg'
    json_path = '/Users/florencelopez/Desktop/Archiv/descriptions/*'

    os_path_img = os.path.expanduser(img_path)
    os_path_json = os.path.expanduser(json_path)

    list_fns_img = glob.glob(os_path_img)
    list_fns_json = glob.glob(os_path_json)

    print(len(list_fns_img))
    print(len(list_fns_json))

    i = 0
    lesion_classes = np.zeros([len(list_fns_json), 2])
    print(lesion_classes)

    while(i < len(list_fns_img)):
        image = Image.open(list_fns_img[i % len(list_fns_img)])
        np_image = np.asarray(image)
        res = np.expand_dims(np_image, 0)
        yield res

        json_file = json.load(open(list_fns_json[i % len(list_fns_json)]))

        #search for the lesion class
        clinical_class = json_file["meta"]["clinical"]["benign_malignant"]

        # benign = [1, 0]
        if clinical_class == "benign":
            lesion_classes[i] = [1, 0]

        # maligne = [0, 1]
        elif clinical_class == "malignant":
            lesion_classes[i] = [0, 1]

        i = i + 1

        yield lesion_classes


       #img_input.show()


def load_net(x=None):
    return tf.image.resize_bilinear(images=x, size=[ 500, 500], name='resize')

x = tf.placeholder(dtype=tf.float32, shape=[1, 767, 1022, 3])
net = inception_v3.inception_v3_base(x)

gen = dataloader_gen()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2):
        feed_dict = {x: gen.__next__()}
        result = sess.run(net, feed_dict=feed_dict)

        #print(result.shape)
        res_test = result.squeeze()
        print(type(res_test))
        img_res = Image.fromarray(res_test.astype(np.uint8))
        img_res.show()
        input()




