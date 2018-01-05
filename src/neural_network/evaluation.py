import datetime
import json

import tensorflow as tf
from PIL import Image
import os
import numpy as np

import glob

from image_tools.preprocess import preprocess
from load_data import dataset_loader
from tensorflow.contrib.slim.python.slim.nets import inception_v3


def dataloader_gen(batch_size=2):
    # img_path = '/Users/spaethju/Desktop/*jpg'
    # json_path = '/Users/spaethju/Desktop/*'

    # img_path = 'D:\Data\Documents\AutomaticSaveToDisc\Datasets\ISIC-Archive-Downloader-master\Data\Images\*_resized.jpg'
    # json_path = 'D:\Data\Documents\AutomaticSaveToDisc\Datasets\ISIC-Archive-Downloader-master\Data\Descriptions\*'
    img_path = '/Users/spaethju//Desktop/Images/*_resized.jpg'
    json_path = '/Users/spaethju//Desktop/Labels/*'

    os_path_img = os.path.expanduser(img_path)
    os_path_json = os.path.expanduser(json_path)

    list_fns_img = glob.glob(os_path_img)
    list_fns_json = glob.glob(os_path_json)

    print(len(list_fns_img))
    print(len(list_fns_json))

    i = 0
    # lesion_classes = np.zeros([len(list_fns_json), 2])
    # print(lesion_classes)

    while (True):
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


# x = tf.placeholder(dtype=tf.float32, shape=[-1, 767, 1022, 3], name='input')
# y = tf.placeholder(dtype=tf.float32, shape=[-1, 1, 2, 1], name='label')

x = tf.placeholder(dtype=tf.float32, shape=[1, 542, 718, 3], name='input')
y = tf.placeholder(dtype=tf.float32, shape=[1, 2], name='label')

x_preprocessed = preprocess(x)

net, endpoints = inception_v3.inception_v3(inputs=x_preprocessed, num_classes=2, is_training=True,
                                           dropout_keep_prob=0.8)

gen = dataloader_gen()



# Define loss and optimizer
# Todo: find loss function
learning_rate = 1e-3
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

restorer = tf.train.Saver() # load correct weights
saver = tf.train.Saver()
snapshot_folder = "./snapshots/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "/"

if not os.path.exists(os.path.expanduser(snapshot_folder)):
    os.makedirs(os.path.expanduser(snapshot_folder))

train_dataset, test_dataset, validation_dataset = dataset_loader()
max_timesteps = len(test_dataset)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    restorer.restore(sess=sess, save_path='./snapshots/2018-01-05_11-43-33/model')

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i in range(max_timesteps):
        img_input, label_input = gen.__next__()
        feed_dict = {x: img_input, y: label_input}
        result, label = sess.run([net,y], feed_dict=feed_dict)
        if (result[0][0] >= result[0][1]):
            result[0][0] = 1
            result[0][1] = 0
        else:
            result[0][0] = 0
            result[0][1] = 1

        result_set = set(result[0])
        label_set = set(label[0])

        if ((result_set == label_set) and (result_set == set([1,0]))):
            true_negatives+=1
        elif ((result_set == label_set) and (result_set == set([0,1]))):
            true_positives+=1
        elif ((result_set != label_set) and (result_set == set([1, 0]))):
            false_negatives+=1
        elif ((result_set != label_set) and (result_set == set([0, 1]))):
            false_positives+=1

    acc = (true_positives + true_negatives) / max_timesteps

    print("TP: " + str(true_positives) + " TN: " + str(true_negatives) + " FP: " + str(false_positives) + " FN: " + str(false_negatives)
          + " Acc: " + str(acc))







