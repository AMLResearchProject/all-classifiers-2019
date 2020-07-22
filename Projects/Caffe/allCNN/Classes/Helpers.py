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
# Title:         Caffe Acute Lymphoblastic Leukemia CNN Helpers
# Description:   Common tools used by the Caffe Acute Lymphoblastic Leukemia CNN
# Configuration: Required/Confs.json
# Last Modified: 2019-03-10
#
############################################################################################

import json, time

from datetime import datetime

class Helpers():
    
    def __init__(self):

        """
        Sets up all default requirements and placeholders 
        needed for the Caffe Acute Lymphoblastic Leukemia CNN Helpers.
        """
        
        pass
    
    def loadConfs(self):

        """
        Load the allCNN Classifier configuration. 
        """

        confs = {}
        with open('Required/Confs.json') as confs:
            confs = json.loads(confs.read())
        return confs
    
    def currentDateTime(self):

        """
        Gets the current date and time in words.
        """
        
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def timerStart(self):

        """
        Starts a timer.
        """

        return str(datetime.now()), time.time()

    def timerEnd(self, start):

        """
        Starts the timer.
        """

        return time.time(), (time.time() - start), str(datetime.now())

    def setLogFile(self, path):

        """
        Sets a log file path.
        """
        
        return path + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S') + ".txt"
        
    def logMessage(self, logfile, process, messageType, message, hide = False):

        """
        Logs a message to a log file.
        """

        logString = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "|" + process + "|" + messageType + ": " + message
        with open(logfile,"a") as logLine:
            logLine.write(logString+'\r\n')
        if hide == False:
            print(logString)