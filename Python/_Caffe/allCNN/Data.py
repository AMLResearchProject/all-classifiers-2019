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
# Title:         Caffe Acute Lymphoblastic Leukemia CNN
# Description:   Used to create the dataset for the Caffe Acute Lymphoblastic Leukemia CNN
# Configuration: Required/Confs.json
# Last Modified: 2019-03-16
# References:    Based on: ACUTE LEUKEMIA CLASSIFICATION USING CONVOLUTION NEURAL NETWORK 
#                IN CLINICAL DECISION SUPPORT SYSTEM
#                https://airccj.org/CSCP/vol7/csit77505.pdf
#
############################################################################################

from Classes.Helpers import Helpers
from Classes.CaffeHelpers import CaffeHelpers

class Data():
    
    def __init__(self):

        """
        Sets up all default requirements and placeholders 
        needed for the Caffe Acute Lymphoblastic Leukemia CNN data script.
        """
        
        self.Helpers = Helpers()
        self.confs = self.Helpers.loadConfs()
        self.logFile = self.Helpers.setLogFile(self.confs["Settings"]["Logs"]["allCNN"])

        self.CaffeHelpers = CaffeHelpers(self.confs, self.Helpers, self.logFile)
        
        self.Helpers.logMessage(self.logFile, "allCNN", "Status", "Data init complete")

    def sortData(self):

        """
        Prepares the data ready for training.
        """
        
        self.CaffeHelpers.deleteLMDB()
        self.CaffeHelpers.sortLabels()
        self.CaffeHelpers.sortTrainingData()
        self.CaffeHelpers.recreatePaperData()
        self.CaffeHelpers.createTrainingLMDB()
        self.CaffeHelpers.createValidationLMDB()
        self.CaffeHelpers.computeMean()
        
        self.Helpers.logMessage(self.logFile, "allCNN", "Status", "Data sorting complete")

Data = Data()
Data.sortData()