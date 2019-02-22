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
# Title:         Augmentation Program
# Description:   Augmentation program for the Acute Myeloid/Lymphoblastic Leukemia Classifier.
# Configuration: required/confs.json
#
# Last Modified: 2019-02-22
#
############################################################################################

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from Classes.Data import Data

class ManualAugmentation():
    
    def __init__(self):
        
        ###############################################################
        #
        # Sets up all default requirements and placeholders 
        # needed for the Acute Myeloid Leukemia Classifier. 
        #
        ###############################################################

        self.Data = Data()
        
    def processDataset(self):
        
        ###############################################################
        #
        # Make sure you have your equal amounts of positive and negative
        # samples in the Model/Data directories.
        # 
        # Only run this function once! it will continually make copies 
        # of all images in the Settings->TrainDir directory specified 
        # in Required/confs.json        
        #
        ###############################################################
        
        self.Data.processDataset() 
        
ManualAugmentation = ManualAugmentation()
print("!! Data Augmentation Program Starting !!")
print("")
ManualAugmentation.processDataset() 
print(" Data Augmentation Program Complete")
print("")