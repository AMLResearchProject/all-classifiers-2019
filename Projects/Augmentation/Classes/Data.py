############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    ALL Detection System 2019
# Project:       Data Augmentation
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Data Class
# Description:   Data augmentation class for the ALL Detection System 2019.
# License:       MIT License
# Credit:        Based on methods outline in the Leukemia Blood Cell Image
#                Classification Using Convolutional Neural Network paper:
#                http://www.ijcte.org/vol10/1198-H0012.pdf
# Last Modified: 2020-07-14
#
############################################################################################

import os
import cv2
import random
import time

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import seed
from scipy import ndimage
from PIL import Image

from Classes.Helpers import Helpers


class Data():
    """ ALL Detection System 2019 Data Class

    Data augmentation class for the ALL Detection System 2019 Data Augmentation project.
    """

    def __init__(self):
        """ Initializes the Data class. """

        self.Helpers = Helpers()
        self.confs = self.Helpers.loadConfs()
        self.fixed = tuple(
            (self.confs["Settings"]["ImgDims"], self.confs["Settings"]["ImgDims"]))

        self.filesMade = 0
        self.trainingDir = self.confs["Settings"]["TrainDir"]

        self.seed = self.confs["Settings"]["Seed"]
        seed(self.seed)

    def writeImage(self, filename, image):
        """ Writes an image to provided file path. """

        if filename is None:
            print("Filename does not exist, file cannot be written.")
            return

        if image is None:
            print("Image does not exist, file cannot be written.")
            return

        try:
            cv2.imwrite(filename, image)
        except:
            print("File was not written! " + filename)

    def resize(self, filePath, savePath, show=False):
        """ Writes a resized image to provided file path. """

        image = cv2.resize(cv2.imread(filePath), self.fixed)
        self.writeImage(savePath, image)
        self.filesMade += 1
        print("Resized image written to: " + savePath)

        if show is True:
            plt.imshow(image)
            plt.show()

        return image

    def grayScale(self, image, grayPath, show=False):
        """ Writes a grayscaled image to provided file path. """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.writeImage(grayPath, gray)
        self.filesMade += 1
        print("Grayscaled image written to: " + grayPath)

        if show is True:
            plt.imshow(gray)
            plt.show()

        return image, gray

    def equalizeHist(self, gray, histPath, show=False):
        """ Writes a histogram equalized image to provided file path. """

        hist = cv2.equalizeHist(gray)
        self.writeImage(histPath, cv2.equalizeHist(gray))
        self.filesMade += 1
        print("Histogram equalized image written to: " + histPath)

        if show is True:
            plt.imshow(hist)
            plt.show()

        return hist

    def reflection(self, image, horPath, verPath, show=False):
        """ Writes a reflected image to provided file path. """

        horImg = cv2.flip(image, 0)
        self.writeImage(horPath, horImg)
        self.filesMade += 1
        print("Horizontally reflected image written to: " + horPath)

        if show is True:
            plt.imshow(horImg)
            plt.show()

        verImg = cv2.flip(image, 1)
        self.writeImage(verPath, verImg)
        self.filesMade += 1
        print("Vertical reflected image written to: " + verPath)

        if show is True:
            plt.imshow(verImg)
            plt.show()

        return horImg, verImg

    def gaussian(self, filePath, gaussianPath, show=False):
        """ Writes a gaussian blurred image to provided file path. """

        gaussianBlur = ndimage.gaussian_filter(
            plt.imread(filePath), sigma=5.11)
        self.writeImage(gaussianPath, gaussianBlur)
        self.filesMade += 1
        print("Gaussian image written to: " + gaussianPath)

        if show is True:
            plt.imshow(gaussianBlur)
            plt.show()
        return gaussianBlur

    def translate(self, image, filePath, show=False):
        """
        Writes transformed copy of the image to the filepath provided. 
        """

        cols, rows, chs = image.shape

        translated = cv2.warpAffine(image, np.float32([[1, 0, 84], [0, 1, 56]]), (rows, cols),
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(144, 159, 162))

        self.writeImage(filePath, translated)
        self.filesMade += 1
        print("Translated image written to: " + filePath)

        if show is True:
            plt.imshow(translated)
            plt.show()

        return translated

    def rotation(self, path, filePath, filename, show=False):
        """
        Writes rotated copies of the image to the filepath provided. 
        """

        img = Image.open(filePath)

        image = cv2.imread(filePath)
        cols, rows, chs = image.shape

        for i in range(0, 20):
            # Seed needs to be set each time randint is called
            random.seed(self.seed)
            randDeg = random.randint(-180, 180)
            matrix = cv2.getRotationMatrix2D((cols/2, rows/2), randDeg, 0.70)
            rotated = cv2.warpAffine(image, matrix, (rows, cols), borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(144, 159, 162))
            fullPath = os.path.join(
                path, str(randDeg) + '-' + str(i) + '-' + filename)

            self.writeImage(fullPath, rotated)
            self.filesMade += 1
            print("Rotated image written to: " + fullPath)

            if show is True:
                plt.imshow(rotated)
                plt.show()

    def processDataset(self):
        """ Processes the dataset. """

        for directory in os.listdir(self.trainingDir):

            # Skip none data directories
            if(directory == ".ipynb_checkpoints" or directory == "__pycache__"):
                continue

            self.filesMade = 0

            path = os.path.join(self.confs["Settings"]["TrainDir"], directory)
            sortedPath = os.path.join(
                self.confs["Settings"]["AugDir"], directory)

            # Stops program from crashing if augmented folders do not exist
            if not os.path.exists(sortedPath):
                os.makedirs(sortedPath)

            if os.path.isdir(path):
                fCount = 0
                for filename in os.listdir(path):
                    if filename.endswith('.jpg'):

                        filePath = os.path.join(path, filename)
                        fullPath = sortedPath + "/" + filename

                        image = self.resize(filePath, fullPath, False)
                        image, gray = self.grayScale(image, os.path.join(
                                                     sortedPath, "Gray-" + filename), False)

                        hist = self.equalizeHist(gray, os.path.join(
                                                 sortedPath, "Hist-" + filename), False)

                        horImg, verImg = self.reflection(image, os.path.join(sortedPath, "Hor-" + filename),
                                                         os.path.join(sortedPath, "Ver-" + filename), False)

                        gaussianBlur = self.gaussian(filePath, os.path.join(
                                                     sortedPath, "Gaus-" + filename), False)

                        translated = self.translate(image, os.path.join(
                            sortedPath, "Trans-"+filename), False)

                        self.rotation(sortedPath, fullPath, filename)
                        fCount += 1
                        print("Total augmented files created so far " +
                              str(self.filesMade))
                        print("")
                    else:
                        continue

                print(" AML/ALL Augmentation: " + self.Helpers.currentDateTime() + "  - Added filters to " + str(fCount) +
                      " files in the " + str(directory) + " directory, with a total of " + str(self.filesMade) + " augmented files created.")
                print("")
