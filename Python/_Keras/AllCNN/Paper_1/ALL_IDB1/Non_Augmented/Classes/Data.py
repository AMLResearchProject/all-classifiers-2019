############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    AML/ALL Classifiers
# Project:       Keras AllCNN
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Data Class
# Description:   Helper data functions used with Keras AllCNN.
# License:       MIT License
# Credit:        Based on Amita Kapoor & Taru Jain's QuantizedCode notebook.
#                https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/Python/_Keras/QuantisedCode/QuantisedCode.ipynb
# Last Modified: 2019-10-26
#
############################################################################################

import os
import cv2
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.utils import np_utils
from pathlib import Path
from PIL import Image
from scipy import ndimage
from skimage import transform as tm


class Data():
    """ Data Class

    Helper data functions used with Keras AllCNN.
    """

    def __init__(self, logger, confs):
        """ Initializes the Data class. """

        self.confs = confs
        self.files_made = 0
        self.expected_file = ".jpg"
        self.logger = logger

        self.negativeTrainAmnt = 40
        self.positiveTrainAmnt = 40

        self.negativeTestAmnt = 19
        self.positiveTestAmnt = 9

        self.train_data_0 = []
        self.train_data_1 = []

    def prepare_data(self, data_dir):
        """ Prepares the data. """

        images_dir = Path(data_dir)
        images = images_dir.glob("*" + self.confs["expected_file"])

        temp_data_0 = []
        temp_data_1 = []
        train_data = []
        train_data_0 = []
        train_data_1 = []
        val_data = []
        val_data_0 = []
        val_data_1 = []

        counter = 0
        for img in images:
            if "_0" in os.path.basename(img):
                train_data_0.append((img, 0))
            else:
                train_data_1.append((img, 1))
            counter += 1

        self.logger.info("Positive training data length: " +
                         str(len(train_data_1)))
        self.logger.info("Negative training data length: " +
                         str(len(train_data_0)))
        self.logger.info("Total training data length: " + str(counter))
        print("")

        random.Random(3).shuffle(train_data_0)
        random.Random(3).shuffle(train_data_1)

        temp_data_0 = train_data_0
        temp_data_1 = train_data_1

        train_data_0 = temp_data_0[0:40]
        train_data_1 = temp_data_1[0:40]

        self.logger.info(
            "Paper positive training dataset recreated, size: " + str(len(train_data_1)))
        self.logger.info(
            "Paper negative training dataset recreated, size: " + str(len(train_data_0)))

        for i in range(0, len(train_data_0)):
            train_data.append(train_data_0[i])
        for i in range(0, len(train_data_1)):
            train_data.append(train_data_1[i])

        self.logger.info("Total training dataset recreated: " +
                         str(len(train_data)))
        print("")

        val_data_0 = temp_data_0[40:]
        val_data_1 = temp_data_1[40:]

        self.logger.info(
            "Paper positive validation data created, size: " + str(len(val_data_1)))
        self.logger.info(
            "Paper negative validation data created, size: " + str(len(val_data_0)))

        for i in range(0, len(val_data_0)):
            val_data.append(val_data_0[i])
        for i in range(0, len(val_data_1)):
            val_data.append(val_data_1[i])

        self.logger.info(
            "Total validation dataset recreated: " + str(len(val_data)))
        print("")

        return train_data, val_data

    def data_and_labels(self, train_data, val_data):

        vdata, vlabels = self.get_data("Validation", val_data)
        data, labels = self.get_data("Training", train_data)

        return data, labels, vdata, vlabels

    def get_data(self, dclass, tdata):

        training_data = pd.DataFrame(tdata, columns=[
            'image', 'label'], index=None)
        tdata = training_data.sample(
            frac=1.).reset_index(drop=True)

        count = 0
        n = len(tdata)
        data = np.zeros(
            (n, self.confs["dims"], self.confs["dims"], 3), dtype=np.float32)
        labels = np.zeros((n, 2), dtype=np.float32)

        for j in range(0, n):

            img_name = tdata.iloc[j]['image']
            img_sname = os.path.basename(img_name)
            label = tdata.iloc[j]['label']

            encoded_label = np_utils.to_categorical(label, num_classes=2)

            img = self.resize(cv2.imread(str(img_name)),
                              self.confs["dims"], show=False)

            if img.shape[2] == 1:
                img = np.dstack([img, img, img])

            data[count] = img.astype(np.float32)/255.
            labels[count] = encoded_label

            count += 1

        self.logger.info(dclass + " data shape: " + str(data.shape))
        self.logger.info(dclass + " labels shape: " + str(labels.shape))
        print("")

        return data, labels

    def resize(self, img, dim, show=False):
        """ Writes a resized image to provided file path. """

        img = cv2.resize(img, (dim, dim))

        if show is True:
            plt.imshow(img)
            plt.show()

        return img
