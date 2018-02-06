import datetime
import json

import tensorflow as tf
from PIL import Image
import os
import numpy as np

import glob

from image_tools.preprocess import preprocess
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def dataloader_gen(batch_size=2):
    # img_path = '/Users/spaethju/Desktop/*jpg'
    # json_path = '/Users/spaethju/Desktop/*'

    ## Cluster paths
    # img_path = '/data/scratch/einig/SkinCancerData/train/Images/*_resized.jpg'
    # json_path = '/data/scratch/einig/SkinCancerData/train/Descriptions/*'

    img_path = 'D:/Data/Documents/AutomaticSaveToDisc/Datasets/ISIC-Archive-Downloader-master/Data/Images/*_resized.jpg'
    json_path = 'D:/Data/Documents/AutomaticSaveToDisc/Datasets/ISIC-Archive-Downloader-master/Data/Descriptions/*'
    #img_path = '/Users/spaethju//Desktop/Images/*_resized.jpg'
    #json_path = '/Users/spaethju//Desktop/Labels/*'

    # img_path = '../datasets/Minimalbeispiel/images/*_resized.jpg'
    # json_path = '../datasets/Minimalbeispiel/descriptions/*'

    os_path_img = os.path.expanduser(img_path)
    os_path_json = os.path.expanduser(json_path)

    list_fns_img = glob.glob(os_path_img)
    # list_fns_json = glob.glob(os_path_json)

    print(len(list_fns_img))
    # print(len(list_fns_json))

    i = 0

    while (True):
        # batch-size
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

            # res = np.expand_dims(np_image, 0)
            res.append(np_image)

            # JSON
            # json_file = json.load(open(list_fns_json[i % len(list_fns_json)]))
            json_file = json.load(open(json_single_img_path))

            # search for the lesion class
            clinical_class = json_file["meta"]["clinical"]["benign_malignant"]

            #lesion_classes = np.zeros([1, 2])
            if clinical_class == "benign":
                lesion_classes[j, 0] = 1

            # maligne = [0, 1]
            elif clinical_class == "malignant":
                lesion_classes[j, 1] = 1

            i = i + 1

        yield res, lesion_classes


def l1_loss(logits=None, labels=None):
    subs = tf.subtract(labels, logits)
    return tf.reduce_sum(tf.abs(subs))

def l2_loss(logits=None, labels=None):
    subs = tf.subtract(labels, logits)
    return tf.reduce_sum(tf.pow(subs, 2))

def sm_cross_loss(logits=None, labels=None):
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)


batch_size = 2

#x = tf.placeholder(dtype=tf.float32, shape=[-1, 767, 1022, 3], name='input')
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2], name='label')

x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 542, 718, 3], name='input')
#y = tf.placeholder(dtype=tf.float32, shape=[1, 2], name='label')

x_preprocessed = preprocess(x)

net, endpoints = inception_v3.inception_v3(inputs=x_preprocessed, num_classes=2, is_training=True,
                                           dropout_keep_prob=0.8)

gen = dataloader_gen(batch_size)
exclude_set_restore = ['.*biases',
                       'InceptionV3/AuxLogits/Conv2d_2b_1x1/weights',
                       'InceptionV3/Logits/Conv2d_1c_1x1/weights']
variables_to_restore = slim.get_variables_to_restore(exclude=exclude_set_restore)

# Define loss and optimizer
learning_rate = 1e-3
# Todo: find loss function
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))
loss = tf.reduce_mean(l1_loss(logits=net, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

restorer = tf.train.Saver(variables_to_restore)
saver = tf.train.Saver()
snapshot_folder = "./snapshots/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "/"

if not os.path.exists(os.path.expanduser(snapshot_folder)):
    os.makedirs(os.path.expanduser(snapshot_folder))

max_timesteps = 1000000

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    restorer.restore(sess=sess, save_path='../neural_network/nets/weights/inception_v3.ckpt')

    summary_loss = tf.summary.scalar("loss", loss)
    summary_pred_hist = tf.summary.histogram("label histogram", net)
    summary_pred_fst = tf.summary.scalar("first value", net[0][0])
    summary_pred_snd = tf.summary.scalar("second value", net[0][1])
    # summary_label_fst = tf.summary.scalar("first value", y[0])
    # summary_label_snd = tf.summary.scalar("second value", y[1])
    summary_image = tf.summary.image("image", x_preprocessed)
    summaries = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(snapshot_folder, sess.graph)
    # summary_writer.add_meta_graph(tf.get_default_graph())

    for i in range(max_timesteps):
        img_input, label_input = gen.__next__()

        feed_dict = {x: img_input, y: label_input}
        # pred = sess.run(net, feed_dict=feed_dict)
        # current_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        if i % 100 == 0:
            evaluated_summaries, current_loss, _ = sess.run([summaries, loss, optimizer], feed_dict=feed_dict)
            summary_writer.add_summary(evaluated_summaries, i)
            summary_writer.flush()
            print("iteration: " + str(i) + " current loss (on single image): " + str(current_loss) )
        else:
            sess.run([optimizer], feed_dict=feed_dict)
        if i % 1000 == 0:
            saver.save(sess=sess, save_path=snapshot_folder + "model")
            print("Saved snapshot for iteration: " + str(i))

