import glob
import json
import os
from random import shuffle

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3

from neural_network.image_tools.augmentations import augment
from neural_network.image_tools.preprocess import preprocess


def dataloader_gen(img_path=None, batch_size=2):
    os_path_img = os.path.expanduser(img_path)
    list_fns_img = glob.glob(os_path_img)
    shuffle(list_fns_img)

    print(len(list_fns_img))

    i = 0

    while (True):
        res = []
        lesion_classes = np.zeros([batch_size, 2])
        for j in range(batch_size):
            single_img_path = list_fns_img[i % len(list_fns_img)].replace("\\", "/")

            fn_name = "_".join(single_img_path.split('/')[-1].split("_")[0: 2])
            json_single_img_path = "/".join(single_img_path.split('/')[0: -2]) + "/Descriptions/" + fn_name

            # IMAGE
            image = Image.open(single_img_path)
            np_image = np.asarray(image)

            if np_image.shape[0] > np_image.shape[1]:
                np_image = np.rot90(np_image, axes=(-3, -2))

            res.append(np_image)

            # JSON
            json_file = json.load(open(json_single_img_path))

            # search for the lesion class
            clinical_class = json_file["meta"]["clinical"]["benign_malignant"]

            if clinical_class == "benign":
                lesion_classes[j, 0] = 1

            elif clinical_class == "malignant":
                lesion_classes[j, 1] = 1

            i = i + 1

        yield res, lesion_classes


def train(img_path, loss_func, learning_rate, batch_size, snapshot_folder, save_intervals):
    x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 542, 718, 3], name='input')
    y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2], name='label')

    x_preprocessed = preprocess(x)
    x_augmented  = augment(x_preprocessed, random=True, rotation=True, vertical_flip=True)
    net, endpoints = inception_v3.inception_v3(inputs=x_augmented, num_classes=2, is_training=True,
                                               dropout_keep_prob=0.8)

    gen = dataloader_gen(img_path=img_path, batch_size=batch_size)
    exclude_set_restore = ['.*biases',
                           'InceptionV3/AuxLogits/Conv2d_2b_1x1/weights',
                           'InceptionV3/Logits/Conv2d_1c_1x1/weights']
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude_set_restore)

    loss = tf.reduce_mean(loss_func(logits=net, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

    restorer = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()

    max_timesteps = 1000000

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        restorer.restore(sess=sess, save_path='neural_network/nets/weights/inception_v3.ckpt')

        summary_loss = tf.summary.scalar("loss", loss)
        summary_pred_hist = tf.summary.histogram("label histogram", net)
        summary_pred_fst = tf.summary.scalar("first value", net[0][0])
        summary_pred_snd = tf.summary.scalar("second value", net[0][1])
        summary_image = tf.summary.image("preprocessed image", x_preprocessed)
        summary_image = tf.summary.image("augmented image", x_augmented)
        summaries = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(snapshot_folder, sess.graph)
        # summary_writer.add_meta_graph(tf.get_default_graph())

        for i in range(max_timesteps):
            img_input, label_input = gen.__next__()

            feed_dict = {x: img_input, y: label_input}
            # pred = sess.run(net, feed_dict=feed_dict)
            # current_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
            if i % save_intervals[0] == 0:
                evaluated_summaries, current_loss, _ = sess.run([summaries, loss, optimizer], feed_dict=feed_dict)
                summary_writer.add_summary(evaluated_summaries, i)
                summary_writer.flush()
                print("iteration: " + str(i) + " current loss (on single image): " + str(current_loss))
            else:
                sess.run([optimizer], feed_dict=feed_dict)
            if i % save_intervals[1] == 0:
                saver.save(sess=sess, save_path=snapshot_folder + "model")
                print("Saved snapshot for iteration: " + str(i))
