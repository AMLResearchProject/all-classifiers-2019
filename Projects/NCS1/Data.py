############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    ALL Detection System 2019
# Project:       Facial Authentication Server
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Training Data Class
# Description:   Training Data class for the ALL Detection System 2019 NCS1 Classifier.
# License:       MIT License
# Last Modified: 2020-07-16
#
############################################################################################

import os, random, sys

from Classes.Helpers import Helpers
from Classes.Data import Data as DataProcess


class Data():
    """ Trainer Data Class

    Sorts the ALL Detection System 2019 NCS1 Classifier training data.
    """

    def __init__(self):
        """ Initializes the Data Class """

        self.Helpers = Helpers("Data")
        self.confs = self.Helpers.confs

        self.DataProcess = DataProcess()
        self.labelsToName = {}

        self.Helpers.logger.info("Data class initialization complete.")

    def sortData(self):
        """ Sorts the training data """

        humanStart, clockStart = self.Helpers.timerStart()

        self.Helpers.logger.info("Loading & preparing training data.")

        dataPaths, classes = self.DataProcess.processFilesAndClasses()

        classId = [int(i) for i in classes]
        classNamesToIds = dict(zip(classes, classId))

        # Divide the training datasets into train and test
        numValidation = int(
            self.confs["Classifier"]["ValidationSize"] * len(dataPaths))
        self.Helpers.logger.info("Number of classes: " + str(classes))
        self.Helpers.logger.info("Validation data size: " + str(numValidation))
        random.seed(self.confs["Classifier"]["RandomSeed"])
        random.shuffle(dataPaths)
        trainingFiles = dataPaths[numValidation:]
        validationFiles = dataPaths[:numValidation]

        # Convert the training and validation sets
        self.DataProcess.convertToTFRecord(
            'train', trainingFiles, classNamesToIds)
        self.DataProcess.convertToTFRecord(
            'validation', validationFiles, classNamesToIds)

        # Write the labels to file
        labelsToClassNames = dict(zip(classId, classes))
        self.DataProcess.writeLabels(labelsToClassNames)

        self.Helpers.logger.info(
            "Loading & preparing training data completed.")

    def cropTestData(self):
        """ Crops the testing data """
        
        self.DataProcess.cropTestDataset()

        self.Helpers.logger.info(
            "Testing data resized.")


if __name__ == "__main__":

    ProcessData = Data()
    ProcessData.sortData()
