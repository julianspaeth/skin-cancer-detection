import datetime
import json

import tensorflow as tf
from PIL import Image
import os
import numpy as np

import glob

from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3


# from .nets.inception_v3 import inception_v3


def dataloader_gen(batch_size=2):
    # img_path = '/Users/spaethju/Desktop/*jpg'
    # json_path = '/Users/spaethju/Desktop/*'

    img_path = 'D:\Data\Documents\AutomaticSaveToDisc\Datasets\ISIC-Archive-Downloader-master\Data\Images\*_resized.jpg'
    json_path = 'D:\Data\Documents\AutomaticSaveToDisc\Datasets\ISIC-Archive-Downloader-master\Data\Descriptions\*'

    os_path_img = os.path.expanduser(img_path)
    os_path_json = os.path.expanduser(json_path)

    list_fns_img = glob.glob(os_path_img)
    list_fns_json = glob.glob(os_path_json)

    print(len(list_fns_img))
    print(len(list_fns_json))

    i = 0
    # lesion_classes = np.zeros([len(list_fns_json), 2])
    # print(lesion_classes)

    while (i < len(list_fns_img)):
        # IMAGE
        image = Image.open(list_fns_img[i % len(list_fns_img)])
        np_image = np.asarray(image)

        if np_image.shape[0] > np_image.shape[1]:
            np_image = np.rot90(np_image, axes=(-3, -2))

        res = np.expand_dims(np_image, 0)



        # JSON
        json_file = json.load(open(list_fns_json[i % len(list_fns_json)]))

        # search for the lesion class
        clinical_class = json_file["meta"]["clinical"]["benign_malignant"]

        lesion_classes = np.zeros([1, 2])
        if clinical_class == "benign":
            lesion_classes[0, 0] = 1

        # maligne = [0, 1]
        elif clinical_class == "malignant":
            lesion_classes[0, 1] = 1

        i = i + 1

        yield res, lesion_classes


def make_square(x=None):
    # TODO use correct preprocessing!!
    return tf.image.resize_bilinear(images=x, size=[299, 299], name='resize')


# x = tf.placeholder(dtype=tf.float32, shape=[-1, 767, 1022, 3], name='input')
# y = tf.placeholder(dtype=tf.float32, shape=[-1, 1, 2, 1], name='label')

x = tf.placeholder(dtype=tf.float32, shape=[1, 542, 718, 3], name='input')
y = tf.placeholder(dtype=tf.float32, shape=[1, 2], name='label')

x_resized = make_square(x)
net, endpoints = inception_v3.inception_v3(inputs=x_resized, num_classes=2, is_training=True, dropout_keep_prob=0.8)

gen = dataloader_gen()
exclude_set_restore = ['.*biases',
               'InceptionV3/AuxLogits/Conv2d_2b_1x1/weights',
               'InceptionV3/Logits/Conv2d_1c_1x1/weights']
variables_to_restore = slim.get_variables_to_restore(exclude=exclude_set_restore)

# Define loss and optimizer
# Todo: find loss function
learning_rate = 1e-3
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

restorer = tf.train.Saver(variables_to_restore)
saver = tf.train.Saver()
snapshot_folder = "./snapshots/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "/"

if not os.path.exists(os.path.expanduser(snapshot_folder)):
    os.makedirs(os.path.expanduser(snapshot_folder))

max_timesteps = 100000

# TODO make tensorboard history stuff!


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    restorer.restore(sess=sess, save_path='../neural_network/nets/weights/inception_v3.ckpt')

    for i in range(max_timesteps):
        img_input, label_input = gen.__next__()

        feed_dict = {x: img_input, y: label_input}
        # pred = sess.run(net, feed_dict=feed_dict)
        # current_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        if i % 100 == 0:
            current_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
            print("iteration: " + str(i) + " current loss (on single image): " + str(current_loss))
        else:
            sess.run([optimizer], feed_dict=feed_dict)
        if i % 1000 == 0:
            saver.save(sess=sess, save_path=snapshot_folder + "model")


        # input()
