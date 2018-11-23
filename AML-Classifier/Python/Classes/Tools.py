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
# Title:         Acute Myeloid Leukemia Classifier Tools
# Description:   Common tools used by the Acute Myeloid Leukemia Classifier.
# Configuration: required/confs.json
# Last Modified: 2018-11-17
#
############################################################################################

import datetime, json

import numpy as np
import matplotlib.pyplot as plt

class Tools():
    
    def __init__(self):
        
        ###############################################################
        #
        # Sets up all default requirements and placeholders 
        # needed for the Acute Myeloid Leukemia Classifier. 
        #
        ###############################################################
        
        pass
    
    def loadConfs(self):
        
        ###############################################################
        #
        # Load the AML DNN Classifier configuration. 
        #
        ###############################################################

        confs = {}
        with open('Required/confs.json') as confs:
            confs = json.loads(confs.read())
        return confs
    
    def currentDateTime(self):
        
        ###############################################################
        #
        # Gets the current date and time in words. 
        #
        ###############################################################
        
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")