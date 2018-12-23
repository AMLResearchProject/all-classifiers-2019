##########################################################################################################
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
# Title:         Acute Myeloid Leukemia Movidius Classifier Data Helpers
# Description:   Helpers for data augmentation used with the Acute Myeloid Leukemia Movidius Classifier.
# Configuration: required/confs.json
# Last Modified: 2018-12-23
#
##########################################################################################################

import os, sys, time, math, random, json, glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

from sys import argv
from datetime import datetime

from Classes.Helpers import Helpers

class ImageReader(object):
        
    ###############################################################
    #
    # Helper class that provides TensorFlow image coding utilities. 
    #
    ###############################################################

    def __init__(self):
        
        ###############################################################
        #
        # Initializes function that decodes RGB JPEG data. 
        #
        ###############################################################
        
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_image(self._decode_jpeg_data, channels=3)
        #self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        
        ###############################################################
        #
        # Reads the dimensions of the image passed here. 
        #
        ###############################################################
        
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        
        ###############################################################
        #
        # Decodes the jpeg image passed here. 
        #
        ###############################################################
        
        image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

class Data():
        
    ###############################################################
    #
    # Core Data class.
    #
    ###############################################################

    def __init__(self):
        
        ###############################################################
        #
        # Sets up all default requirements and placeholders 
        # needed for this class. 
        #
        ###############################################################
        
        self.Helpers = Helpers()
        self.confs = self.Helpers.loadConfs()
        self.logFile = self.Helpers.setLogFile(self.confs["Settings"]["Logs"]["DataLogDir"])

    def getLabelsAndDirectories(self):
        
        ###############################################################
        #
        # Returns a list of classes/labels and directories. 
        #
        ###############################################################

        labels = [name for name in os.listdir(self.confs["Classifier"]["DatasetDir"]) if os.path.isdir(os.path.join(self.confs["Classifier"]["DatasetDir"], name)) and name != '.ipynb_checkpoints']

        directories = []
        for dirName in os.listdir(self.confs["Classifier"]["DatasetDir"]):
            if dirName != '.ipynb_checkpoints':
                path = os.path.join(self.confs["Classifier"]["DatasetDir"], dirName)
                if os.path.isdir(path):
                    directories.append(path)
        return labels, directories

    def processFilesAndClasses(self):
        
        ###############################################################
        #
        # Returns a list of filenames and classes/labels. 
        #
        ###############################################################

        labels, directories = self.getLabelsAndDirectories()
        
        data = []
        for directory in directories:
            for filename in os.listdir(directory):
                if os.path.splitext(filename)[1] in self.confs["Classifier"]["ValidIType"]:
                    data.append(os.path.join(directory, filename))
                else:
                    continue
        return data, sorted(labels)

    def convertToTFRecord(self, split_name, filenames, labels_to_ids):
        
        ###############################################################
        #
        # Converts the given filenames to a TFRecord dataset. 
        #
        ###############################################################
        
        assert split_name in ['train', 'validation']

        num_per_shard = int(math.ceil(len(filenames) / float(self.confs["Classifier"]["Shards"])))
        self.Helpers.logMessage(self.logFile, "convertToTFRecord", "INFO", "Number of files: " + str(len(filenames)))
        self.Helpers.logMessage(self.logFile, "convertToTFRecord", "INFO", "Number per shard: " + str(num_per_shard))

        with tf.Graph().as_default():
            image_reader = ImageReader()
            with tf.Session('') as sess:
                for shard_id in range(self.confs["Classifier"]["Shards"]):
                    output_filename = self.getDatasetFilename(split_name, shard_id)
                    self.Helpers.logMessage(self.logFile, "convertToTFRecord", "STATUS", "Saving: " + str(output_filename))
                    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                        start_ndx = shard_id * num_per_shard
                        end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                        for i in range(start_ndx, end_ndx):
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                i+1, len(filenames), shard_id))
                            sys.stdout.flush()
                            print("")
                            # Read the filename:
                            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                            height, width = image_reader.read_image_dims(sess, image_data)
                            class_name = os.path.basename(os.path.dirname(filenames[i]))
                            class_id = labels_to_ids[class_name]
                            self.Helpers.logMessage(self.logFile, "convertToTFRecord", "INFO", "class_name: " + str(class_name))
                            self.Helpers.logMessage(self.logFile, "convertToTFRecord", "INFO", "class_id: " + str(class_id))
                            example = self.imageToTFExample(
                                image_data, b'jpg', height, width, class_id)
                            tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()

    def getDatasetFilename(self, split_name, shard_id):
        
        ###############################################################
        #
        # Gets the dataset filename
        #
        ###############################################################

        output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
            self.confs["Classifier"]["TFRecordFile"], split_name, shard_id, self.confs["Classifier"]["Shards"])
        return os.path.join(self.confs["Classifier"]["DatasetDir"], output_filename)

    def int64Feature(self, values):
        
        ###############################################################
        #
        # Returns a TF-Feature of int64s
        #
        ###############################################################
        
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def bytesFeature(self, values):
        
        ###############################################################
        #
        # Returns a TF-Feature of bytes
        #
        ###############################################################
        
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

    def imageToTFExample(self, image_data, image_format, height, width, class_id):
        
        ###############################################################
        #
        # Converts an image to a TF Example
        #
        ###############################################################

        return tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': self.bytesFeature(image_data),
            'image/format': self.bytesFeature(image_format),
            'image/class/label': self.int64Feature(class_id),
            'image/height': self.int64Feature(height),
            'image/width': self.int64Feature(width)
        }))

    def writeLabels(self, labels_to_labels):
        
        ###############################################################
        #
        # Writes a file with the list of class names
        #
        ###############################################################

        labels_filename = os.path.join(self.confs["Classifier"]["DatasetDir"], self.confs["Classifier"]["Labels"])

        with tf.gfile.Open(self.confs["Classifier"]["Classes"], 'w') as f:
            for label in labels_to_labels:
                f.write('%s\n' % (label))
                
        with tf.gfile.Open(labels_filename, 'w') as f:
            for label in labels_to_labels:
                class_name = labels_to_labels[label]
                f.write('%d:%s\n' % (label, class_name))