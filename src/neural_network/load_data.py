import datetime
import json

import tensorflow as tf
from PIL import Image
import os
import csv
import numpy as np
import glob



def dataset_loader():
    #train_path = '/Users/florencelopez/Desktop/STUDIUM /Master/Fächer/Praktikum Machine Learning/PrakikumML2017/src/datasets/training.dataset'
    #test_path = '/Users/florencelopez/Desktop/STUDIUM /Master/Fächer/Praktikum Machine Learning/PrakikumML2017/src/datasets/test.dataset'
    #validation_path = '/Users/florencelopez/Desktop/STUDIUM /Master/Fächer/Praktikum Machine Learning/PrakikumML2017/src/datasets/validation.dataset'

    train_path = '/Users/florencelopez/Desktop/Minimalbeispiel/sets/train.dataset'
    test_path = '/Users/florencelopez/Desktop/Minimalbeispiel/sets/test.dataset'
    validation_path = '/Users/florencelopez/Desktop/Minimalbeispiel/sets/validation.dataset'
    overall_description = '/Users/florencelopez/Desktop/Minimalbeispiel/descriptions/'

    os_path_train = os.path.expanduser(train_path)
    os_path_test = os.path.expanduser(test_path)
    os_path_validation = os.path.expanduser(validation_path)

    # load training data as array
    raw_data_train = open(os_path_train, 'r')
    reader_train = csv.reader(raw_data_train, delimiter = '\t')
    x_train = list(reader_train)
    data_train = np.array(x_train).astype('str')
    #print(data_train[:10])
    data_train = np.squeeze(data_train)

    # load test data as array
    raw_data_test = open(os_path_test, 'r')
    reader_test = csv.reader(raw_data_test, delimiter='\t')
    x_test = list(reader_test)
    data_test = np.array(x_test).astype('str')
    #print(data_test[:10])
    data_test = np.squeeze(data_test)

    # load validation data as array
    raw_data_validation = open(os_path_validation, 'r')
    reader_validation = csv.reader(raw_data_validation, delimiter='\t')
    x_validation = list(reader_validation)
    data_validation = np.array(x_validation).astype('str')
    #print(data_validation[:10])
    data_validation = np.squeeze(data_validation)


    # load descriptions as array




    lesion_classes_train = np.zeros([len(data_train), 2])

    for i in range(len(data_train)):
        description_path = overall_description + data_train[i]
        print(description_path)
        os_path_description = os.path.expanduser(description_path)
        list_fns_description = glob.glob(os_path_description)
        json_file = json.load(open(list_fns_description[0]))
        # search for the lesion class
        clinical_class = json_file["meta"]["clinical"]["benign_malignant"]
        print(clinical_class)

        if clinical_class == "benign":
            lesion_classes_train[i, 0] = 1

        # maligne = [0, 1]
        elif clinical_class == "malignant":
            lesion_classes_train[i, 1] = 1

    print(lesion_classes_train)

    lesion_classes_test = np.zeros([len(data_test), 2])

    for i in range(len(data_test)):
        description_path = overall_description + data_test[i]
        print(description_path)
        os_path_description = os.path.expanduser(description_path)
        list_fns_description = glob.glob(os_path_description)
        json_file = json.load(open(list_fns_description[0]))
        # search for the lesion class
        clinical_class = json_file["meta"]["clinical"]["benign_malignant"]
        print(clinical_class)

        if clinical_class == "benign":
            lesion_classes_test[i, 0] = 1

        # maligne = [0, 1]
        elif clinical_class == "malignant":
            lesion_classes_test[i, 1] = 1

    print(lesion_classes_test)

    lesion_classes_validation = np.zeros([len(data_test), 2])

    for i in range(len(data_validation)):
        description_path = overall_description + data_validation[i]
        print(description_path)
        os_path_description = os.path.expanduser(description_path)
        list_fns_description = glob.glob(os_path_description)
        json_file = json.load(open(list_fns_description[0]))
        # search for the lesion class
        clinical_class = json_file["meta"]["clinical"]["benign_malignant"]
        print(clinical_class)

        if clinical_class == "benign":
            lesion_classes_validation[i, 0] = 1

        # maligne = [0, 1]
        elif clinical_class == "malignant":
            lesion_classes_validation[i, 1] = 1

    print(lesion_classes_validation)


    dataset_train = list(zip(data_train, lesion_classes_train))
    dataset_test = list(zip(data_test, lesion_classes_test))
    dataset_validation = list(zip(data_validation, lesion_classes_validation))
    print(dataset_train)
    print(dataset_test)
    print(dataset_validation)


dataset_loader()