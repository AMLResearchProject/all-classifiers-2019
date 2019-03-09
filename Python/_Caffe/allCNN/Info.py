import os, sys, cv2
import numpy as np

sys.path.append('/home/upsquared/caffe/python')
import caffe

class allCNN():

    def __init__(self):

        self.net = caffe.Net('allCNN.prototxt', caffe.TEST)

allCNN = allCNN()

print("")
print("Net Inputs")
print(allCNN.net.inputs)
print("")

print("Net Blobs")
print(allCNN.net.blobs)
print("")

print("Net Params")
print(allCNN.net.params)
print("")