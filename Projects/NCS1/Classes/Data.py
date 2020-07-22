############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    ALL Detection System 2019
# Project:       Facial Authentication Server
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Data Class
# Description:   Data class for the ALL Detection System 2019 NCS1 Classifier.
# License:       MIT License
# Last Modified: 2020-07-16
#
############################################################################################

import cv2, glob, json, math, os, pathlib, random, sys, time 

import numpy as np
import tensorflow as tf

from datetime import datetime
from PIL import Image
from sys import argv

from Classes.Helpers import Helpers


class Data():
    """ Data Helper Class

    Core data management class for the ALL Detection System 2019 NCS1 Classifier
    """

    def __init__(self):
        """ Initializes the Data Class. """

        self.Helpers = Helpers("DataProcessor")
        self.confs = self.Helpers.confs

        self.Helpers.logger.info("Data helper class initialization complete.")

    def getLabelsAndDirectories(self):
        """ Returns a list of classes/labels and directories. """

        labels = [name for name in os.listdir(self.confs["Classifier"]["DatasetDir"]) if os.path.isdir(
            os.path.join(self.confs["Classifier"]["DatasetDir"], name)) and name != '.ipynb_checkpoints']

        directories = []
        for dirName in os.listdir(self.confs["Classifier"]["DatasetDir"]):
            if dirName != '.ipynb_checkpoints':
                path = os.path.join(
                    self.confs["Classifier"]["DatasetDir"], dirName)
                if os.path.isdir(path):
                    directories.append(path)

        return labels, directories

    def processFilesAndClasses(self):
        """ Returns a list of filenames and classes/labels. """

        labels, directories = self.getLabelsAndDirectories()

        data = []
        for directory in directories:
            for filename in os.listdir(directory):
                if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.gif'):
                    data.append(os.path.join(directory, filename))
                else:
                    continue

        return data, sorted(labels)

    def writeLabels(self, labels_to_labels):
        """

        Writes a file with the list of class names.

        Args:
            labels_to_labels: A map of (integer) labels to class names.
            filename: The filename where the class names are written.

        """

        labelsFile = os.path.join(
            self.confs["Classifier"]["DatasetDir"], self.confs["Classifier"]["Labels"])

        classesFile = os.path.join(
            self.confs["Classifier"]["DatasetDir"], self.confs["Classifier"]["Classes"])

        with tf.gfile.Open(classesFile, 'w') as f:
            for label in labels_to_labels:
                f.write('%s\n' % (label))

        with tf.gfile.Open(labelsFile, 'w') as f:
            for label in labels_to_labels:
                class_name = labels_to_labels[label]
                f.write('%d:%s\n' % (label, class_name))

    def convertToTFRecord(self, split_name, filenames, labels_to_ids):
        """ Converts the given filenames to a TFRecord dataset. """

        assert split_name in ['train', 'validation']

        num_per_shard = int(
            math.ceil(len(filenames) / float(self.confs["Classifier"]["Shards"])))

        self.Helpers.logger.info("Files: " + str(len(filenames)))
        self.Helpers.logger.info("Files per shard: " + str(num_per_shard))

        with tf.Graph().as_default():
            image_reader = ImageReader()
            with tf.Session('') as sess:
                for shard_id in range(self.confs["Classifier"]["Shards"]):
                    output_filename = self.getDatasetFilename(
                        split_name, shard_id)
                    self.Helpers.logger.info(
                        "Saving shard: " + output_filename)
                    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                        start_ndx = shard_id * num_per_shard
                        end_ndx = min(
                            (shard_id+1) * num_per_shard, len(filenames))
                        for i in range(start_ndx, end_ndx):
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                i+1, len(filenames), shard_id))
                            sys.stdout.flush()

                            image_data = tf.gfile.FastGFile(
                                filenames[i], 'rb').read()
                            height, width = image_reader.read_image_dims(
                                sess, image_data)
                            class_name = os.path.basename(
                                os.path.dirname(filenames[i]))
                            class_id = labels_to_ids[class_name]
                            example = self.imageToTFExample(
                                image_data, b'jpg', height, width, class_id)
                            tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()

    def getDatasetFilename(self, split_name, shard_id):
        """ Gets the model TFRecordFile. """

        output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
            self.confs["Classifier"]["TFRecordFile"], split_name, shard_id, self.confs["Classifier"]["Shards"])
        return os.path.join(self.confs["Classifier"]["DatasetDir"], output_filename)

    def int64Feature(self, values):
        """

        Returns a TF-Feature of int64s.

        Args:
            values: A scalar or list of values.

        Returns:
            a TF-Feature.

        """
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def bytesFeature(self, values):
        """

        Returns a TF-Feature of bytes.

        Args:
            values: A string.

        Returns:
            a TF-Feature.

        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

    def imageToTFExample(self, image_data, image_format, height, width, class_id):

        return tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': self.bytesFeature(image_data),
            'image/format': self.bytesFeature(image_format),
            'image/class/label': self.int64Feature(class_id),
            'image/height': self.int64Feature(height),
            'image/width': self.int64Feature(width)
        }))

    def cropTestDataset(self):
        """ Crops the testing dataset. """
        
        data_dir = pathlib.Path(
            self.confs["Classifier"]["TestImagePath"])
        data = list(data_dir.glob('*.jpg'))
        
        for ipath in data:
            fpath = str(ipath)
            
            image = Image.open(fpath)
            
            image = image.resize((600, 600))
            image.save(fpath)

        self.Helpers.logger.info("Test data resized.")


class ImageReader(object):
    """ ImageReader Helper Class

    Provides TensorFlow image coding utilities
    """

    def __init__(self):
        """ Initializes ImageReader Class """

        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_image(
            self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        """ Gets the dimensions of image_data """

        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        """ Decodes image_data (jpeg)"""

        image = sess.run(self._decode_jpeg, feed_dict={
                         self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image
