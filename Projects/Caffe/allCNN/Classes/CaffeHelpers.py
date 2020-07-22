############################################################################################
#
# The MIT License (MIT)
# 
# Peter Moss Acute Myeloid/Lymphoblastic Leukemia AI Research Project
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
# Title:         Caffe Acute Lymphoblastic Leukemia CNN Caffe Helpers
# Description:   Common Caffe tools used by the Caffe Acute Lymphoblastic Leukemia CNN
# Configuration: Required/Confs.json
# Last Modified: 2019-03-10
# References:    Based on: ACUTE LEUKEMIA CLASSIFICATION USING CONVOLUTION NEURAL NETWORK 
#                IN CLINICAL DECISION SUPPORT SYSTEM
#                https://airccj.org/CSCP/vol7/csit77505.pdf
#
############################################################################################

import os, sys, random, cv2, lmdb
sys.path.append('/home/upsquared/caffe/python')

from caffe.proto import caffe_pb2

class CaffeHelpers():
    
    def __init__(self, confs, helpers, logFile):

        """
        Sets up all default requirements and placeholders needed for the 
        Caffe Acute Lymphoblastic Leukemia CNN Helpers.
        """

        self.Helpers = helpers
        self.confs = confs
        self.logFile = logFile

        self.labels = None
        self.trainLMDB = None

        self.trainData = []
        self.trainData0 = []
        self.trainData1 = []
        self.valData = []
        self.valData0 = []
        self.valData1 = []
        self.classNames = []

        self.imSize = (self.confs["Settings"]["Classifier"]["Input"]["imageWidth"], 
                       self.confs["Settings"]["Classifier"]["Input"]["imageHeight"])

        self.negativeTrainAmnt = self.confs["Settings"]["Classifier"]["Data"]["negativeTrainAmnt"]
        self.positiveTrainAmnt = self.confs["Settings"]["Classifier"]["Data"]["positiveTrainAmnt"]

        self.negativeTestAmnt = self.confs["Settings"]["Classifier"]["Data"]["negativeTestAmnt"]
        self.positiveTestAmnt = self.confs["Settings"]["Classifier"]["Data"]["positiveTestAmnt"]
        
        self.Helpers.logMessage(self.logFile, "allCNN", "Status", "CaffeHelpers initiated")

    def deleteLMDB(self):

        """
        Deletes existing LMDB files.
        """
        
        os.system('rm -rf  ' + self.confs["Settings"]["Classifier"]["LMDB"]["train"])
        os.system('rm -rf  ' + self.confs["Settings"]["Classifier"]["LMDB"]["validation"])
        
        self.Helpers.logMessage(self.logFile, "allCNN", "Status", "Existing LMDB deleted")

    def sortLabels(self):

        """
        Sorts the training / validation data and labels.
        """

        self.labels = open(self.confs["Settings"]["Classifier"]["Data"]["labels"], "w")

        for dirName in os.listdir(self.confs["Settings"]["Classifier"]["Model"]["dirData"]):
            if dirName == ".ipynb_checkpoints":
                continue
            path = os.path.join(self.confs["Settings"]["Classifier"]["Model"]["dirData"], dirName)
            if os.path.isdir(path):
                self.classNames.append(path)
                self.labels.write(dirName+"\n")
        
        self.labels.close()
        
        self.Helpers.logMessage(self.logFile, "allCNN",  "Status", str(len(self.classNames)) + " label(s) created")

    def appendDataPaths(self, directory, filename):

        """
        Appends training data paths.
        """
        
        filePath = os.path.join(directory, filename)
        cDirectory = os.path.basename(os.path.normpath(directory))

        image = cv2.imread(filePath)
        image = cv2.resize(image, self.imSize)
        cv2.imwrite(filePath, image)

        if cDirectory is self.confs["Settings"]["Classifier"]["Data"]["negativeDir"]:
            self.trainData0.append(filePath)
        elif cDirectory is self.confs["Settings"]["Classifier"]["Data"]["positiveDir"]:
            self.trainData1.append(filePath)

    def isValidFile(self, filename):

        """
        Checks that input file type is allowed.
        """

        return filename.endswith(tuple(self.confs["Settings"]["Classifier"]["Data"]["validFiles"]))

    def sortTrainingData(self):

        """
        Sorts the training / validation data
        """

        for directory in self.classNames:
            for filename in os.listdir(directory):
                if self.isValidFile(filename):
                    self.appendDataPaths(directory, filename)
                else:
                    continue

    def recreatePaperData(self):

        """
        Recreates the dataset sizes specified in the paper.
        """
        
        self.Helpers.logMessage(self.logFile, "allCNN", "Status", "Total data size: " + str(len(self.trainData0) + len(self.trainData1)) + " (" + str(len(self.trainData0)) + " + " + str(len(self.trainData1)) + ")")

        msg = "Recreating negative training size of " + str(self.negativeTrainAmnt) 
        msg += " with negative testing size of " + str(self.negativeTestAmnt) 
        msg += " and positive training size of " + str(self.positiveTrainAmnt) 
        msg += " with positive testing size of " + str(self.positiveTestAmnt)

        self.Helpers.logMessage(self.logFile, "allCNN", "Status", msg)
        
        random.shuffle(self.trainData0)
        random.shuffle(self.trainData1)

        trainingData0 = self.trainData0[0:self.negativeTrainAmnt]
        trainingData1 = self.trainData1[0:self.positiveTrainAmnt]

        self.Helpers.logMessage(self.logFile, "allCNN", "Data", "Negative training data created, size: " + str(len(trainingData0)))
        self.Helpers.logMessage(self.logFile, "allCNN", "Data", "Positive training data created, size: " + str(len(trainingData1)))
        
        for i in range(0, len(trainingData0)): 
            self.trainData.append(trainingData0[i])
        for i in range(0, len(trainingData1)): 
            self.trainData.append(trainingData1[i])
        
        self.Helpers.logMessage(self.logFile, "allCNN", "Status", "Paper training data created. " + str(len(trainingData0)) + " x Negative & " + str(len(trainingData1)) + " x Positive ")

        valData0 = self.trainData0[self.negativeTrainAmnt:]
        valData1 = self.trainData1[self.positiveTrainAmnt:]

        self.Helpers.logMessage(self.logFile, "allCNN", "Data", "Negative validation data created, size: " + str(len(valData0)))
        self.Helpers.logMessage(self.logFile, "allCNN", "Data", "Positive validation data created, size: " + str(len(valData1)))
        
        for i in range(0, len(valData0)): 
            self.valData.append(valData0[i])
        for i in range(0, len(valData1)): 
            self.valData.append(valData1[i])
        
        self.Helpers.logMessage(self.logFile, "allCNN", "Status", "Paper validation data created. " + str(len(valData0)) + " x Negative & " + str(len(valData1)) + " x Positive ")

        msg = "Recreated negative training size of " + str(len(trainingData0)) 
        msg += " with negative testing size of " + str(len(trainingData1)) 
        msg += " and positive training size of " + str(len(valData0)) 
        msg += " with positive testing size of " + str(len(valData1))

        self.Helpers.logMessage(self.logFile, "allCNN", "Status", msg)

    def createDatum(self, imageData, label):

        """
        Generates a Datum object including label.
        """
    
        datum = caffe_pb2.Datum()
        datum.channels = imageData.shape[2]
        datum.height = imageData.shape[0]
        datum.width = imageData.shape[1]
        datum.data = imageData.tobytes()
        datum.label = int(label)

        return datum
        
    def transform(self, img):

        """
        Transforms image using histogram equalization and resizing.
        """
        
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
        
        return cv2.resize(img, 
                          self.imSize, 
                          interpolation = cv2.INTER_CUBIC)

    def createAllDatum(self, rlmdb, data, dType):

        """
        Generates all Datum objects.
        """
        
        if dType is "Training":
            dataPath = self.trainData
        elif dType is "Validation":
            dataPath = self.valData

        with rlmdb.begin(write=True) as i:
            count = 0
            for data in dataPath:
                i.put(
                    '{:08}'.format(count).encode('ascii'), 
                    self.createDatum(
                        cv2.resize(
                            self.transform(
                                cv2.imread(data, cv2.IMREAD_COLOR)
                            ), 
                            (self.confs["Settings"]["Classifier"]["Input"]["imageHeight"], self.confs["Settings"]["Classifier"]["Input"]["imageWidth"])), 
                        os.path.basename(os.path.dirname(data))
                    ).SerializeToString())
                count = count + 1
        rlmdb.close()
        
        self.Helpers.logMessage(self.logFile,  "allCNN", "Status", dType + " data count: " + str(count))

    def createTrainingLMDB(self):

        """
        Creates training LMDB database.
        """

        random.shuffle(self.trainData)
        self.createAllDatum(lmdb.open(self.confs["Settings"]["Classifier"]["LMDB"]["train"], map_size=int(1e12)), self.trainData, "Training")
        
        self.Helpers.logMessage(self.logFile, "allCNN", "Status", "Training LDBM created")

    def createValidationLMDB(self):

        """
        Creates validation LMDB database.
        """

        random.shuffle(self.valData)
        self.createAllDatum(lmdb.open(self.confs["Settings"]["Classifier"]["LMDB"]["validation"], map_size=int(1e12)), self.trainData, "Validation")
        
        self.Helpers.logMessage(self.logFile, "allCNN", "Status", "Validation LDBM created")

    def computeMean(self):

        """
        Computes the mean.
        """

        os.system('/home/upsquared/caffe/build/tools/compute_image_mean -backend=lmdb  ' + self.confs["Settings"]["Classifier"]["LMDB"]["train"] + ' ' + self.confs["Settings"]["Classifier"]["Caffe"]["proto"])
        
        self.Helpers.logMessage(self.logFile, "allCNN", "Status", "Mean computed")