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

import os, cv2, random

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image

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
        self.fixed = tuple((self.confs["Settings"]["ImgDims"], self.confs["Settings"]["ImgDims"]))
        
    def writeImage(self, filename, image):
        
        ###############################################################
        #
        # Writes an image based on the filepath and the image provided. 
        #
        ###############################################################
        
        cv2.imwrite(filename, image)
        
    def resize(self, filePath, savePath, show = False):
        
        ###############################################################
        #
        # Writes an image based on the filepath and the image provided. 
        #
        ###############################################################

        image = cv2.resize(cv2.imread(filePath), self.fixed)
        self.writeImage(savePath, image)
        if show is True:
            print(savePath)
            plt.imshow(image)
            plt.show()
        return image

    def grayScale(self, image, grayPath, show = False):
        
        ###############################################################
        #
        # Writes a grayscale copy of the image to the filepath provided. 
        #
        ###############################################################
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.writeImage(grayPath, gray)
        if show is True:
            print(grayPath)
            plt.imshow(gray)
            plt.show()
        return image, gray

    def equalizeHist(self, gray, histPath, show = False):
        
        ###############################################################
        #
        # Writes histogram equalized copy of the image to the filepath 
        # provided. 
        #
        ###############################################################
        
        hist = cv2.equalizeHist(gray)
        self.writeImage(histPath, cv2.equalizeHist(gray))
        if show is True:
            print(histPath)
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
        
        horImg = cv2.flip(image, 0)
        self.writeImage(horPath, horImg)
        if show is True:
            print(horPath)
            plt.imshow(horImg)
            plt.show()
        verImg = cv2.flip( image, 1 )
        self.writeImage(verPath, verImg)
        if show is True:
            print(verPath)
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
        
        gaussianBlur = ndimage.gaussian_filter(plt.imread(filePath), sigma=5.11)
        self.writeImage(gaussianPath, gaussianBlur)
        if show is True:
            print(gaussianPath)
            plt.imshow(gaussianBlur)
            plt.show()
        return gaussianBlur
        
    def rotation(self, path, filePath, filename, show = False): 
        
        ###############################################################
        #
        # Writes rotated copies of the image to the filepath 
        # provided. 
        #
        ###############################################################
        
        img = Image.open(filePath)

        for i in range(1, 10):
            randDeg = random.randint(-180, 180)
            fullPath = os.path.join(path, str(randDeg) + '-' + filename)
            if show is True:
                print(fullPath)
                img.rotate(randDeg, expand=True).resize((self.confs["Settings"]["ImgDims"], self.confs["Settings"]["ImgDims"])).save(fullPath).show()
            else:
                print(fullPath)
                img.rotate(randDeg, expand=True).resize((self.confs["Settings"]["ImgDims"], self.confs["Settings"]["ImgDims"])).save(fullPath)

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
                sortedPath = os.path.join(self.confs["Settings"]["TrainDir"]+"augmented/", directory)
                fCount = 0
                for filename in os.listdir(path):
                    if filename.endswith('.jpg'):
                        filePath = os.path.join(path, filename)
                        image = self.resize(filePath, sortedPath+"/"+filename, True)
                        image, gray = self.grayScale(image, os.path.join(sortedPath, "Gray-"+filename), True)
                        hist = self.equalizeHist(gray, os.path.join(sortedPath, "Hist-"+filename), True)
                        horImg, verImg = self.reflection(image, os.path.join(sortedPath, "Hor-"+filename), os.path.join(sortedPath, "Ver-"+filename), True)
                        gaussianBlur = self.gaussian(filePath, os.path.join(sortedPath, "Gaus-"+filename), True)
                        self.rotation(sortedPath, filePath, filename)
                        fCount += 1
                    else:
                        continue
                print(" AML-DNN: " + self.Tools.currentDateTime() + "  - Added filters to " + str(fCount) + " files in " + str(directory))