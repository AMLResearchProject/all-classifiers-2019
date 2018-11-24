############################################################################################
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
# Title:         Acute Myeloid Leukemia Classifier Data Tools
# Description:   Tools for data augmentation used with the Acute Myeloid Leukemia Classifier.
# Configuration: required/confs.json
# Last Modified: 2018-11-24
#
############################################################################################

import os, cv2

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from Classes.Tools import Tools

class Data():
    
    def __init__(self):
        
        ###############################################################
        #
        # Sets up all default requirements and placeholders 
        # needed for the Acute Myeloid Leukemia Classifier. 
        #
        ###############################################################
        
        self.Tools = Tools()
        self.confs = self.Tools.loadConfs()
        self.fixed = tuple((257, 257))
        
    def writeImage(self, filename, image):
        
        ###############################################################
        #
        # Writes an image based on the filepath and the image provided. 
        #
        ###############################################################
        
        cv2.imwrite(filename, image)

    def grayScale(self, image, filePath, filename, show = False):
        
        ###############################################################
        #
        # Writes a grayscale copy of the image to the filepath provided. 
        #
        ###############################################################
        
        print(filePath)
        image = cv2.resize(image, self.fixed)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.writeImage(filePath, gray)
        if show is True:
            plt.imshow(gray)
            plt.show()
        return image, gray

    def equalizeHist(self, gray, histPath, filename, show = False):
        
        ###############################################################
        #
        # Writes histogram equalized copy of the image to the filepath 
        # provided. 
        #
        ###############################################################
        
        print(histPath)
        hist = cv2.equalizeHist(gray)
        self.writeImage(histPath, cv2.equalizeHist(gray))
        if show is True:
            plt.imshow(hist)
            plt.show()
        return hist

    def reflection(self, image, horPath, verPath, show = False):
        
        ###############################################################
        #
        # Writes histogram equalized copy of the image to the filepath 
        # provided. 
        #
        ###############################################################
        
        print(horPath)
        horImg = cv2.flip(image, 0)
        self.writeImage(horPath, horImg)
        if show is True:
            plt.imshow(horImg)
            plt.show()
        print(verPath)
        verImg = cv2.flip( image, 1 )
        self.writeImage(verPath, verImg)
        if show is True:
            plt.imshow(verImg)
            plt.show()
        return horImg, verImg

    def gaussian(self, filePath, gaussianPath, show = False):
        
        ###############################################################
        #
        # Writes gaussian blurred copy of the image to the filepath 
        # provided. 
        #
        ###############################################################
        
        print(gaussianPath)
        gaussianBlur = ndimage.gaussian_filter(plt.imread(filePath), sigma=5.11)
        self.writeImage(gaussianPath, gaussianBlur)
        if show is True:
            plt.imshow(gaussianBlur)
            plt.show()
        return gaussianBlur	

    def processDataset(self):
        
        ###############################################################
        #
        # Runs all of the above functions saving the new dataset to the
        # default training directory. 
        #
        ###############################################################
        
        for directory in os.listdir(self.confs["Settings"]["TrainDir"]):
            if os.path.isdir(os.path.join(self.confs["Settings"]["TrainDir"], directory)):
                path = os.path.join(self.confs["Settings"]["TrainDir"], directory)
                print(path)
                fCount = 0
                for filename in os.listdir(path):
                    if filename.endswith('.jpg'):
                        filePath = os.path.join(path, filename)
                        grayPath = os.path.join(path, "Gray-"+filename)
                        print(grayPath)
                        image, gray = self.grayScale(cv2.resize(cv2.imread(filePath), self.fixed), grayPath, filename, show = True)
                        histPath = os.path.join(path, "hist-"+filename)
                        hist = self.equalizeHist(gray, histPath, filename, show = True)
                        horPath = os.path.join(path, "hor-"+filename)
                        verPath = os.path.join(path, "ver-"+filename)
                        horImg, verImg = self.reflection(image, horPath, verPath, True)
                        gaussianPath = os.path.join(path, "gaus-"+filename)
                        gaussianBlur = self.gaussian(filePath, gaussianPath, True)
                        fCount += 1
                    else:
                        continue
                print(" AML-DNN: " + self.Tools.currentDateTime() + "  - Gray scaled " + str(fCount) + " files in " + str(directory))