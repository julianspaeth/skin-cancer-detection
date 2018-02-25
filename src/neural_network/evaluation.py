import glob
import json
import os
import sys
import logging

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.contrib.slim.python.slim.nets import inception_v3
from roc import roc_functions
from neural_network.image_tools.preprocess import preprocess


def dataloader_gen(list_fns_img, batch_size=1):
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


def evaluate(img_path=None, snapshot_folder=None, eval_path=None, verbose=False):
    tf.reset_default_graph()

    x = tf.placeholder(dtype=tf.float32, shape=[1, 542, 718, 3], name='input')
    y = tf.placeholder(dtype=tf.float32, shape=[1, 2], name='label')

    x_preprocessed = preprocess(x)

    net, endpoints = inception_v3.inception_v3(inputs=x_preprocessed, num_classes=2, is_training=True,
                                               dropout_keep_prob=0.8)

    list_fns_img = glob.glob(os.path.expanduser(img_path))
    int_image_files = len(list_fns_img)

    gen = dataloader_gen(list_fns_img=list_fns_img)

    restorer = tf.train.Saver()  # load correct weights

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Evaluating: " + snapshot_folder + '/model')

        restorer.restore(sess=sess, save_path=snapshot_folder + '/model')

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        eval_list_test = []
        eval_list_pred = []

        for i in range(int_image_files):
            img_input, label_input = gen.__next__()
            feed_dict = {x: img_input, y: label_input}
            result, label = sess.run([net, y], feed_dict=feed_dict)

            result_set = np.zeros([2])

            if result[0][0] > result[0][1]:
                if verbose:
                    print("first larger")
                result_set[0] = 1
                result_set[1] = 0

            elif result[0][0] < result[0][1]:
                if verbose:
                    print("second larger")

                result_set[0] = 0
                result_set[1] = 1

            else:
                if verbose:
                    print("equal")
                result_set[0] = 0
                result_set[1] = 0

            label_set = label[0]
            if verbose:
                print(label_set)
                print(result_set)
            other = 0

            str_debug = ""

            if result_set[0] == 1 and result_set[1] == 0:
                str_debug += "benign_"
                if result_set[0] == label_set[0] and result_set[1] == label_set[1]:
                    str_debug += "true"

                    eval_list_test.append(0)
                    eval_list_pred.append(0)
                    true_negatives += 1
                elif result_set[0] != label_set[0] or result_set[1] != label_set[1]:
                    str_debug += "false"

                    eval_list_test.append(0)
                    eval_list_pred.append(1)

                    false_negatives += 1
            elif result_set[0] == 0 and result_set[1] == 1:
                str_debug += "malignant_"

                if result_set[0] == label_set[0] and result_set[1] == label_set[1]:
                    str_debug += "true"
                    eval_list_test.append(1)
                    eval_list_pred.append(1)
                    true_positives += 1
                elif result_set[0] != label_set[0] or result_set[1] != label_set[1]:
                    str_debug += "false"

                    eval_list_test.append(1)
                    eval_list_pred.append(0)

                    false_positives += 1
            else:
                str_debug = "other"
                other = other + 1
            if verbose:
                print(str_debug)

            # if ((result_set == label_set) and (result_set == set([1, 0]))):
            #     true_negatives += 1
            # elif ((result_set == label_set) and (result_set == set([0, 1]))):
            #     true_positives += 1
            # elif ((result_set != label_set) and (result_set == set([1, 0]))):
            #     false_negatives += 1
            # elif ((result_set != label_set) and (result_set == set([0, 1]))):
            #     false_positives += 1
            # else:
            #     other = other + 1
            if i % 100 == 0:
                print("Progress: \t{}%\t{}/{}".format(round(i / int_image_files * 100, 2), i, int_image_files))

            # if i % 1000 == 0:
            #     try:
            #         print("1000")
            #         print(eval_list_pred)
            #         print(eval_list_test)
            #         print(eval_path + "/roc_curve_" + str(i) + ".png")
            #         roc_functions.plotROC(pred_labels=eval_list_pred, test_labels=eval_list_test,
            #                               save_path=eval_path + "/roc_curve_" + str(i) + ".png")
            #     except:
            #         print("Exception 1")

        acc = (true_positives + true_negatives) / int_image_files

        if not os.path.exists(eval_path):
            os.makedirs(eval_path)

        print(eval_path)

        try:
            print(len(eval_list_test))
            print(len(eval_list_pred))
            print(eval_list_test)
            print(eval_list_pred)
            print(eval_path + "/roc_curve_" + str(i) + ".png")
            roc_functions.plotROC(pred_labels=eval_list_pred, test_labels=eval_list_test,
                                  save_path=eval_path + "/roc_curve_" + str(i) + ".png")
        except:
            logging.exception("Something awful happened!")
            print(sys.exc_info()[0])
            raise

        with open(eval_path + '/eval.log', 'w') as f:
            eval_string = "TP: " + str(true_positives) + "\nTN: " + str(true_negatives) + "\nFP: " + \
                          str(false_positives) + "\nFN: " + str(false_negatives) \
                          + "\nAcc: " + str(acc) + "\nother: " + str(other)
            f.writelines(eval_string)
            print(eval_string)
