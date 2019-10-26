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
        self.logger = logger

        self.train_data = []

    def prepare_data(self, data_dir):
        """ Prepares the data. """

        images_dir = Path(data_dir)
        images = images_dir.glob("*" + self.confs["expected_file"])

        counter = 0

        for img in images:
            if "_0" in os.path.basename(img):
                self.train_data.append((img, 0))
            else:
                self.train_data.append((img, 1))
            counter += 1

        train_dataf = pd.DataFrame(
            self.train_data, columns=['image', 'label'], index=None)
        self.train_data = train_dataf.sample(frac=1.).reset_index(drop=True)

        count = 0
        n = len(self.train_data)
        data = np.zeros(
            (n, self.confs["dims"], self.confs["dims"], 3), dtype=np.float32)
        labels = np.zeros((n, 2), dtype=np.float32)

        for j in range(0, n):

            img_name = self.train_data.iloc[j]['image']
            img_sname = os.path.basename(img_name)
            label = self.train_data.iloc[j]['label']

            encoded_label = np_utils.to_categorical(label, num_classes=2)

            img = self.resize(cv2.imread(str(img_name)),
                              self.confs["dims"], show=False)

            if img.shape[2] == 1:
                img = np.dstack([img, img, img])

            data[count] = img.astype(np.float32)/255.
            labels[count] = encoded_label

            count += 1

        self.logger.info("Sorted data shape: " + str(data.shape))
        self.logger.info("Sorted labels shape: " + str(labels.shape))

        return data, labels

    def resize(self, img, dim, show=False):
        """ Writes a resized image to provided file path. """

        img = cv2.resize(img, (dim, dim))

        if show is True:
            plt.imshow(img)
            plt.show()

        return img
