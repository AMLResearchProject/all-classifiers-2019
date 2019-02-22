############################################################################################
#
# The MIT License (MIT)
# 
# Peter Moss Acute Myeloid/Lymphoblastic Leukemia Research Project
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
# Title:         Augmentation Data Helpers
# Description:   Helpers for data augmentation used with the Acute Myeloid/Lymphoblastic Leukemia Classifier.
# Configuration: required/confs.json
#
# Last Modified: 2019-02-22
#
############################################################################################

import os, cv2, random, time

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image

from Classes.Helpers import Helpers

class Data():
    
    def __init__(self):
        
        ###############################################################
        #
        # Sets up all default requirements and placeholders 
        # needed for the Acute Myeloid Leukemia Classifier. 
        #
        ###############################################################
        
        self.Helpers = Helpers()
        self.confs = self.Helpers.loadConfs()
        self.fixed = tuple((self.confs["Settings"]["ImgDims"], self.confs["Settings"]["ImgDims"]))
        
        self.filesMade = 0
        self.trainingDir = self.confs["Settings"]["TrainDir"]
        
    def writeImage(self, filename, image):
        
        ###############################################################
        #
        # Writes an image based on the filepath and the image provided. 
        #
        ###############################################################

        if filename is None:
            print("Filename does not exist, file cannot be written.")
            return
            
        if image is None:
            print("Image does not exist, file cannot be written.")
            return
            
        try:
           cv2.imwrite(filename, image)
        except:
            print("File was not written! "+filename)
        
    def resize(self, filePath, savePath, show = False):
        
        ###############################################################
        #
        # Writes an image based on the filepath and the image provided. 
        #
        ###############################################################

        image = cv2.resize(cv2.imread(filePath), self.fixed)
        self.writeImage(savePath, image)
        self.filesMade += 1
        print("Resized image written to: " + savePath)
        
        if show is True:
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
        self.filesMade += 1
        print("Grayscaled image written to: " + grayPath)
        
        if show is True:
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
        self.filesMade += 1
        print("Histogram equalized image written to: " + histPath)
        
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

    def gaussian(self, filePath, gaussianPath, show = False):
        
        ###############################################################
        #
        # Writes gaussian blurred copy of the image to the filepath 
        # provided. 
        #
        ###############################################################
        
        gaussianBlur = ndimage.gaussian_filter(plt.imread(filePath), sigma=5.11)
        self.writeImage(gaussianPath, gaussianBlur)
        self.filesMade += 1
        print("Gaussian image written to: " + gaussianPath)

        if show is True:
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

        for i in range(0, 10):
            randDeg = random.randint(-180, 180)
            fullPath = os.path.join(path, str(randDeg) + '-' + str(i) + '-' + filename)

            try:
                if show is True:
                    img.rotate(randDeg, expand=True).resize((self.confs["Settings"]["ImgDims"], self.confs["Settings"]["ImgDims"])).save(fullPath).show()
                    self.filesMade += 1
                else:
                    img.rotate(randDeg, expand=True).resize((self.confs["Settings"]["ImgDims"], self.confs["Settings"]["ImgDims"])).save(fullPath)
                    self.filesMade += 1
                print("Rotated image written to: " + fullPath)
            except:
                print("File was not written! "+filename)

            time.sleep(1)

    def processDataset(self):
        
        ###############################################################
        #
        # Runs all of the above functions saving the new dataset to the
        # default training directory. 
        #
        ###############################################################
        
        for directory in os.listdir(self.trainingDir):
            
            # Skip none data directories
            if(directory==".ipynb_checkpoints" or directory=="__pycache__"):
                continue
                
            self.filesMade = 0
            
            path = os.path.join(self.confs["Settings"]["TrainDir"], directory)
            sortedPath = os.path.join(self.confs["Settings"]["AugDir"], directory)
            
            # Stops program from crashing if augmented folders do not exist
            if not os.path.exists(sortedPath):
                os.makedirs(sortedPath)
            
            if os.path.isdir(path):
                fCount = 0
                for filename in os.listdir(path):
                    if filename.endswith('.jpg'):
                        
                        filePath = os.path.join(path, filename)
                        
                        image = self.resize(filePath, sortedPath+"/"+filename, False)
                        image, gray = self.grayScale(image, os.path.join(sortedPath, "Gray-"+filename), False)
                        
                        hist = self.equalizeHist(gray, os.path.join(sortedPath, "Hist-"+filename), False)
                        
                        horImg, verImg = self.reflection(image, os.path.join(sortedPath, "Hor-"+filename), 
                                                         os.path.join(sortedPath, "Ver-"+filename), False)\
                        
                        gaussianBlur = self.gaussian(filePath, os.path.join(sortedPath, "Gaus-"+filename), False)
                        
                        self.rotation(sortedPath, filePath, filename)
                        fCount += 1
                        print("Total augmented files created so far " + str(self.filesMade))
                        print("")
                    else:
                        print("File was not jpg! "+filename)
                        continue
                        
                print(" AML/ALL Augmentation: " + self.Helpers.currentDateTime() + "  - Added filters to " + str(fCount) + " files in the " + str(directory) + " directory, with a total of " + str(self.filesMade) + " augmented files created.")
                print("")