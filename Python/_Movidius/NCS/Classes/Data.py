##################################################################################################
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
# Title:         Acute Myeloid Leukemia Movidius Classifier Data Class
# Description:   Class for sorting the data for the Acute Myeloid Leukemia Movidius Classifier.
# Configuration: required/confs.json
# Last Modified: 2019-02-16
#
# Based on code based on https://github.com/tensorflow/
#
##################################################################################################

import os, sys, time, json

from Classes.Helpers import Helpers
from Classes.inception_v3 import inception_v3, inception_v3_arg_scope
import Classes.inception_preprocessing

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pylab as pl

from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.framework import graph_util

import tkinter as tk

plt.style.use('ggplot')
slim = tf.contrib.slim

config = tf.ConfigProto(intra_op_parallelism_threads=12, inter_op_parallelism_threads=2, allow_soft_placement=True,  device_count = {'CPU': 12})

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

class Eval():

    def __init__(self):
        
        self.Helpers = Helpers()
        self.confs = self.Helpers.loadConfs()
        self.logFile = self.Helpers.setLogFile(self.confs["Settings"]["Logs"]["DataLogDir"])
        
        self.labelsToName = {}
        
        self.Helpers.logMessage(self.logFile, "init", "INFO", "Init complete")

        self.checkpoint_file = tf.train.latest_checkpoint(self.confs["Classifier"]["LogDir"])

        #Open the labels file
        self.labels = open(self.confs["Classifier"]["DatasetDir"] + "/" + self.confs["Classifier"]["Labels"], 'r')

        #Create a dictionary to refer each label to their string name
        for line in self.labels:
            label, string_name = line.split(':')
            string_name = string_name[:-1] #Remove newline
            self.labelsToName[int(label)] = string_name

        #Create a dictionary that will help people understand your dataset better. This is required by the Dataset class later.
        self.items_to_descriptions = {
            'image': 'A 3-channel RGB coloured  image that is ex: office, people',
            'label': 'A label that ,start from zero'
        }

    #============== DATASET LOADING ======================
    #We now create a function that creates a Dataset class which will give us many TFRecord files to feed in the examples into a queue in parallel.
    def getSplit(self, split_name):

        '''
            Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
            set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
            Your FilePattern is very important in locating the files later.
            INPUTS:
                - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
            OUTPUTS:
                - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
        '''

        #First check whether the split_name is train or validation
        if split_name not in ['train', 'validation']:
            raise ValueError('The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

        #Create the full path for a general FilePattern to locate the tfrecord_files
        FilePattern_path = os.path.join(self.confs["Classifier"]["DatasetDir"], self.confs["Classifier"]["FilePattern"] % (split_name))

        #Count the total number of examples in all of these shard
        num_samples = 0
        FilePattern_for_counting = '200label_' + split_name
        tfrecords_to_count = [os.path.join(self.confs["Classifier"]["DatasetDir"], file) for file in os.listdir(self.confs["Classifier"]["DatasetDir"]) if file.startswith(FilePattern_for_counting)]

        #print(tfrecords_to_count)
        for tfrecord_file in tfrecords_to_count:
            for record in tf.python_io.tf_record_iterator(tfrecord_file):
                num_samples += 1

        #Create a reader, which must be a TFRecord reader in this case
        reader = tf.TFRecordReader

        #Create the keys_to_features dictionary for the decoder
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
            'image/class/label': tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        }

        #Create the items_to_handlers dictionary for the decoder.
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
        }

        #Start to create the decoder
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

        #Create the labels_to_name file
        labels_to_name_dict = self.labelsToName

        #Actually create the dataset
        dataset = slim.dataset.Dataset(
            data_sources = FilePattern_path,
            decoder = decoder,
            reader = reader,
            num_readers = 4,
            num_samples = num_samples,
            NumClasses = self.confs["Classifier"]["NumClasses"],
            labels_to_name = labels_to_name_dict,
            items_to_descriptions = self.items_to_descriptions)

        return dataset

    def loadBatch(self, dataset, is_training=True):

        '''
            Loads a batch for training.
            INPUTS:
                - dataset(Dataset): a Dataset class object that is created from the get_split function
                - batch_size(int): determines how big of a batch to train
                - height(int): the height of the image to resize to during preprocessing
                - width(int): the width of the image to resize to during preprocessing
                - is_training(bool): to determine whether to perform a training or evaluation preprocessing
            OUTPUTS:
                - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
                - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).
        '''

        #First create the data_provider object
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            common_queue_capacity = 24 + 3 * self.confs["Classifier"]["BatchTestSize"],
            common_queue_min = 24)

        #Obtain the raw image using the get method
        raw_image, label = data_provider.get(['image', 'label'])

        #Perform the correct preprocessing for this image depending if it is training or evaluating
        image = Classes.inception_preprocessing.preprocess_image(raw_image, self.confs["Classifier"]["ImageSize"], self.confs["Classifier"]["ImageSize"], is_training)

        #As for the raw images, we just do a simple reshape to batch it up
        raw_image = tf.image.resize_image_with_crop_or_pad(raw_image, self.confs["Classifier"]["ImageSize"], self.confs["Classifier"]["ImageSize"])

        #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
        images, raw_images, labels = tf.train.batch(
            [image, raw_image, label],
            batch_size = self.confs["Classifier"]["BatchTestSize"],
            num_threads = 4,
            capacity = 4 * self.confs["Classifier"]["BatchTestSize"],
            allow_smaller_final_batch = True)

        return images, raw_images, labels

Eval = Eval()

def run():

    #Create LogDir for evaluation information
    if not os.path.exists(Eval.confs["Classifier"]["LogDirEval"]):
        os.mkdir(Eval.confs["Classifier"]["LogDirEval"])

    #Just construct the graph from scratch again
    with tf.Graph().as_default() as graph:

        tf.logging.set_verbosity(tf.logging.INFO)

        #Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
        dataset = Eval.getSplit('validation')
        images, raw_images, labels = Eval.loadBatch(dataset, is_training = False)

        #Create some information about the training steps
        num_batches_per_epoch = dataset.num_samples / Eval.confs["Classifier"]["BatchTestSize"]
        num_steps_per_epoch = num_batches_per_epoch

        #Now create the inference model but set is_training=False
        with slim.arg_scope(inception_v3_arg_scope()):
            logits, end_points = inception_v3(images, NumClasses = dataset.NumClasses, is_training = False)

        #Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
        one_hot_labels = slim.one_hot_encoding(labels, dataset.NumClasses)

        #Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
        total_loss = tf.losses.get_total_loss()    #obtain the regularization losses as well

        # #get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, Eval.checkpoint_file)

        #Just define the metrics to track without the loss or whatsoever
        probabilities = end_points['Predictions']
        predictions = tf.argmax(probabilities, 1)

        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update)

        #Create the global step and an increment op for monitoring
        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step

        #Create a evaluation step function
        def eval_step(sess, metrics_op, global_step):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, global_step_count, accuracy_value = sess.run([metrics_op, global_step_op, accuracy])
            time_elapsed = time.time() - start_time

            #Log some information
            logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)', global_step_count, accuracy_value, time_elapsed)

            return accuracy_value

        #Define some scalar quantities to monitor
        tf.summary.scalar('Validation Accuracy', accuracy)
        tf.summary.scalar('Validation losses/Total_Loss', total_loss)
        my_summary_op = tf.summary.merge_all()

        #Get your supervisor
        sv = tf.train.Supervisor(logdir = Eval.confs["Classifier"]["LogDirEval"], summary_op = None, init_fn = restore_fn)

        #Now we are ready to run in one session
        with sv.managed_session() as sess:
            for step in range(int(num_batches_per_epoch * Eval.confs["Classifier"]["TestNumEpochs"])):
                #print vital information every start of the epoch as always
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, Eval.confs["Classifier"]["TestNumEpochs"])
                    logging.info('Current Streaming Accuracy: %.4f', sess.run(accuracy))

                #Compute summaries every 10 steps and continue evaluating
                if step % 10 == 0:
                    eval_step(sess, metrics_op = metrics_op, global_step = sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)

                #Otherwise just run as per normal
                else:
                    eval_step(sess, metrics_op = metrics_op, global_step = sv.global_step)

            #At the end of all the evaluation, show the final accuracy
            logging.info('Final Streaming Accuracy: %.4f', sess.run(accuracy))

            #Now we want to visualize the last batch's images just to see what our model has predicted
            raw_images, labels, predictions, probabilities = sess.run([raw_images, labels, predictions, probabilities])
            for i in range(10):
                image, label, prediction, probability = raw_images[i], labels[i], predictions[i], probabilities[i]
                prediction_name, label_name = dataset.labels_to_name[prediction], dataset.labels_to_name[label]
                text = 'Prediction: %s \n Ground Truth: %s \n Probability: %s' %(prediction_name, label_name, probability[prediction])
                img_plot = plt.imshow(image)

                #Set up the plot and hide axes
                plt.title(text)
                img_plot.axes.get_yaxis().set_ticks([])
                img_plot.axes.get_xaxis().set_ticks([])
                plt.show()

            logging.info('Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)

if __name__ == '__main__':
    run()