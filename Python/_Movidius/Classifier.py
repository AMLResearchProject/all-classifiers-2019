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
# Title:         Acute Myeloid Leukemia Movidius Classifier
# Description:   The Acute Myeloid Leukemia Movidius Classifier.
# Configuration: required/confs.json
# Last Modified: 2018-12-23
#
############################################################################################
import os, sys, time, csv, getopt, json, time, cv2

import numpy as np

from mvnc import mvncapi as mvnc
from datetime import datetime
from skimage.transform import resize

from Classes.Helpers import Helpers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

print("-- Setup Environment Settings")

class Classifier():

    def __init__(self):

        self.Helpers = Helpers()
        self.confs = self.Helpers.loadConfs()
        self.logFile = self.Helpers.setLogFile(self.confs["Settings"]["Logs"]["DataLogDir"])
        self.Helpers.logMessage(self.logFile, "init", "INFO", "Init complete")

        self.movidius = None

        self.mean = 128
        self.std = 1/128

        self.categories = []
        self.graphfile = None
        self.graph = None
        self.reqsize = None

        self.extensions = [
            ".jpg",
            ".png"
        ]

        self.CheckDevices()
        

    def CheckDevices(self):

        #mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0:
            self.Helpers.logMessage(self.logFile, "CheckDevices", "WARNING", "No Movidius Devices Found")
            quit()
        self.movidius = mvnc.Device(devices[0])
        self.movidius.OpenDevice()
        self.Helpers.logMessage(self.logFile, "CheckDevices", "STATUS", "Movidius Connected")

    def AllocateGraph(self,graphfile):

        self.graph = self.movidius.AllocateGraph(graphfile)

    def LoadInception(self):

        self.reqsize = self.confs["Classifier"]["ImageSize"]
        with open(self.confs["Classifier"]["NetworkPath"] + self.confs["Classifier"]["InceptionGraph"], mode='rb') as f:
            self.graphfile = f.read()
        self.AllocateGraph(self.graphfile)
        self.Helpers.logMessage(self.logFile, "LoadInception", "STATUS", "Graph Allocated")

        with open(self.confs["Classifier"]["NetworkPath"] + 'Model/classes.txt', 'r') as f:
            for line in f:
                cat = line.split('\n')[0]
                if cat != 'classes':
                    self.categories.append(cat)
            f.close()
        self.Helpers.logMessage(self.logFile, "LoadInception", "STATUS", "Categories Loaded")

Classifier = Classifier()

def main(argv):

    if argv[0] == "InceptionTest":

        humanStart, clockStart = Classifier.Helpers.timerStart()
        Classifier.Helpers.logMessage(Classifier.logFile, "InceptionTest", "STATUS", "INCEPTION V3 TEST MODE STARTING " + humanStart)

        Classifier.LoadInception()

        files = 0
        identified = 0
        incorrect = 0
        correct = 0
        rootdir = Classifier.confs["Classifier"]["NetworkPath"] + Classifier.confs["Classifier"]["TestImagePath"]

        for testFile in os.listdir(rootdir):
            if os.path.splitext(testFile)[1] in Classifier.confs["Classifier"]["ValidIType"]:

                files = files + 1
                fileName = rootdir + "/" + testFile

                Classifier.Helpers.logMessage(Classifier.logFile, "InceptionTest", "STATUS", "Loaded Test Image " + fileName)
                img = cv2.imread(fileName).astype(np.float32)

                dx,dy,dz= img.shape
                delta=float(abs(dy-dx))

                if dx > dy:
                    img=img[int(0.5*delta):dx-int(0.5*delta),0:dy]
                else:
                    img=img[0:dx,int(0.5*delta):dy-int(0.5*delta)]

                img = cv2.resize(img, (Classifier.reqsize, Classifier.reqsize))
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

                for i in range(3):
                    img[:,:,i] = (img[:,:,i] - Classifier.mean) * Classifier.std

                detectionStart, detectionStart = Classifier.Helpers.timerStart()
                Classifier.Helpers.logMessage(Classifier.logFile, "InceptionTest", "STATUS", "Detection started " + str(detectionStart))

                Classifier.graph.LoadTensor(img.astype(np.float16), 'user object')
                output, userobj = Classifier.graph.GetResult()

                detectionClockEnd, difference, detectionEnd = Classifier.Helpers.timerEnd(detectionStart)
                Classifier.Helpers.logMessage(Classifier.logFile, "InceptionTest", "STATUS", "Detection ended " + str(detectionEnd) + " taking " + str(difference))

                top_inds = output.argsort()[::-1][:5]

                if output[top_inds[0]] > Classifier.confs["Classifier"]["InceptionThreshold"] and Classifier.categories[top_inds[0]] == "1":
                    identified += 1
                    if "_1." in fileName:
                        correct += 1
                        Classifier.Helpers.logMessage(Classifier.logFile, "InceptionTest", "INFERENCE", "TASS Identified Correctly AML with a confidence of " + str(output[top_inds[0]]) + " in " + str(difference))
                    else:
                        incorrect += 1
                        Classifier.Helpers.logMessage(Classifier.logFile, "InceptionTest", "INFERENCE", "TASS Identified Incorrectly AML with a confidence of " + str(output[top_inds[0]]) + " in " + str(difference))
                else:
                    if "_0." in fileName:
                        correct += 1
                        Classifier.Helpers.logMessage(Classifier.logFile, "InceptionTest", "INFERENCE", "TASS Identified Correctly NO AML with a confidence of " + str(output[top_inds[0]]) + " in " + str(difference))
                    else:
                        incorrect += 1
                        Classifier.Helpers.logMessage(Classifier.logFile, "InceptionTest", "INFERENCE", "TASS Identified Incorrectly NO AML with a confidence of " + str(output[top_inds[0]]))
        
        clockEnd, difference, humanEnd = Classifier.Helpers.timerEnd(clockStart)
        Classifier.Helpers.logMessage(Classifier.logFile, "InceptionTest", "STATUS", "Testing ended. " + str(correct) + " correct identifications. " + str(incorrect) + " incorrect identifications. In " + str(clockEnd - clockStart) + " seconds including logging and printouts.")

        Classifier.movidius.DeAllocateGraph()
        Classifier.movidius.CloseDevice()

    else:
        print("**ERROR** Check Your Commandline Arguments")
        print("")

if __name__ == "__main__":
	main(sys.argv[1:])