##################################################################################################
#
# The MIT License (MIT)
# 
# Peter Moss Acute Myeloid Leukemia Research Project
# Copyright (C) 2018 Adam Milton-Barker (AdamMiltonBarker.com)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Title:         Acute Myeloid Leukemia Movidius Classifier Data Class
# Description:   Class for sorting the data for the Acute Myeloid Leukemia Movidius Classifier.
# Configuration: required/confs.json
# Last Modified: 2018-12-23
#
##################################################################################################

import os, sys, random

from Classes.Helpers import Helpers
from Classes.Data import Data as DataProcess

class Data():
        
    ###############################################################
    #
    # Core Data class wrapper.
    #
    ###############################################################

    def __init__(self):
        
        ###############################################################
        #
        # Sets up all default requirements and placeholders 
        # needed for this class. 
        #
        ###############################################################
        
        self.Helpers = Helpers()
        self.confs = self.Helpers.loadConfs()
        self.logFile = self.Helpers.setLogFile(self.confs["Settings"]["Logs"]["DataLogDir"])
        
        self.DataProcess = DataProcess()
        self.labelsToName = {}
        
        self.Helpers.logMessage(self.logFile, "init", "INFO", "Init complete")

    def sortData(self):
        
        ###############################################################
        #
        # Sorts the data 
        #
        ###############################################################

        humanStart, clockStart = self.Helpers.timerStart()
        self.Helpers.logMessage(self.logFile, "sortData", "INFO", "Loading & preparing training data")
        
        dataPaths, classes = self.DataProcess.processFilesAndClasses()

        classId = [ int(i) for i in classes]
        classNamesToIds = dict(zip(classes, classId))

        # Divide the training datasets into train and test
        numValidation = int(self.confs["Classifier"]["ValidationSize"] * len(dataPaths))
        self.Helpers.logMessage(self.logFile, "sortData", "Validation Size", str(numValidation))
        self.Helpers.logMessage(self.logFile, "sortData", "Class Size", str(len(classes)))
        random.seed(self.confs["Classifier"]["RandomSeed"])
        random.shuffle(dataPaths)
        trainingFiles = dataPaths[numValidation:]
        validationFiles = dataPaths[:numValidation]

        # Convert the training and validation sets
        self.DataProcess.convertToTFRecord('train', trainingFiles, classNamesToIds)
        self.DataProcess.convertToTFRecord('validation', validationFiles, classNamesToIds)

        # Write the labels to file
        labelsToClassNames = dict(zip(classId, classes))
        self.DataProcess.writeLabels(labelsToClassNames)
        self.Helpers.logMessage(self.logFile, "sortData", "COMPLETE", "Completed sorting data!")


if __name__ == "__main__":
        
    ###############################################################
    #
    # Sorts the data ready for training
    #
    ###############################################################

    ProcessData = Data()
    ProcessData.sortData()