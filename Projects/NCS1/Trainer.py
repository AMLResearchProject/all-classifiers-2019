from __future__ import print_function
############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    ALL Detection System 2019
# Project:       Facial Authentication Server
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Trainer Class
# Description:   Trainer class for the ALL Detection System 2019 NCS1 Classifier.
# License:       MIT License
# Last Modified: 2020-07-16
#
############################################################################################

import glob, json, math, os, random, sys, time

import numpy as np
import tensorflow as tf

import Classes.inception_preprocessing

from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import graph_util

from sys import argv
from datetime import datetime
from builtins import range

from Classes.Helpers import Helpers
from Classes.Data import Data
from Classes.inception_v3 import inception_v3, inception_v3_arg_scope

slim = tf.contrib.slim


class Trainer():
    """ Trainer Class

    Trains the ALL Detection System 2019 NCS1 Trainer.
    """

    def __init__(self):
        """ Initializes Trainer Class """

        self.Helpers = Helpers("Trainer")
        self.confs = self.Helpers.confs

        self.Helpers.logger.info(
            "Trainer class initialization complete.")

        self.labelsToName = {}

    def getSplit(self, split_name):
        """ Gets the training/validation split """

        # Check whether the split_name is train or validation
        if split_name not in ['train', 'validation']:
            raise ValueError(
                'The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

        # Create the full path for a general FilePattern to locate the tfrecord_files
        FilePattern_path = os.path.join(
            self.confs["Classifier"]["DatasetDir"], self.confs["Classifier"]["FilePattern"] % (split_name))

        # Count the total number of examples in all of these shard
        num_samples = 0
        FilePattern_for_counting = 'ALL_' + split_name
        tfrecords_to_count = [os.path.join(self.confs["Classifier"]["DatasetDir"], file) for file in os.listdir(
            self.confs["Classifier"]["DatasetDir"]) if file.startswith(FilePattern_for_counting)]

        # print(tfrecords_to_count)
        for tfrecord_file in tfrecords_to_count:
            for record in tf.python_io.tf_record_iterator(tfrecord_file):
                num_samples += 1

        # Create a reader, which must be a TFRecord reader in this case
        reader = tf.TFRecordReader

        # Create the keys_to_features dictionary for the decoder
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
            'image/class/label': tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        }

        # Create the items_to_handlers dictionary for the decoder.
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
        }

        # Start to create the decoder
        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)

        # Create the labels_to_name file
        labels_to_name_dict = self.labelsToName

        # Actually create the dataset
        dataset = slim.dataset.Dataset(
            data_sources=FilePattern_path,
            decoder=decoder,
            reader=reader,
            num_readers=4,
            num_samples=num_samples,
            num_classes=self.confs["Classifier"]["NumClasses"],
            labels_to_name=labels_to_name_dict,
            items_to_descriptions=self.items_to_descriptions)

        return dataset

    def loadBatch(self, dataset, is_training=True):
        """ Loads a batch for training """

        # First create the data_provider object
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            common_queue_capacity=24 + 3 *
            self.confs["Classifier"]["BatchSize"],
            common_queue_min=24)

        # Obtain the raw image using the get method
        raw_image, label = data_provider.get(['image', 'label'])

        # Perform the correct preprocessing for this image depending if it is training or evaluating
        image = Classes.inception_preprocessing.preprocess_image(
            raw_image, self.confs["Classifier"]["ImageSize"], self.confs["Classifier"]["ImageSize"], is_training)

        # As for the raw images, we just do a simple reshape to batch it up
        raw_image = tf.image.resize_image_with_crop_or_pad(
            raw_image, self.confs["Classifier"]["ImageSize"], self.confs["Classifier"]["ImageSize"])

        # Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
        images, raw_images, labels = tf.train.batch(
            [image, raw_image, label],
            batch_size=self.confs["Classifier"]["BatchSize"],
            num_threads=4,
            capacity=4 * self.confs["Classifier"]["BatchSize"],
            allow_smaller_final_batch=True)

        return images, raw_images, labels


Trainer = Trainer()


def run():
    """ Trainer Runner

    Runs the ALL Detection System 2019 NCS1 Classifier Trainer.
    """

    humanStart, clockStart = Trainer.Helpers.timerStart()

    Trainer.Helpers.logger.info(
        "ALL Detection System 2019 NCS1 Trainer started.")

    # Open the labels file
    Trainer.labels = open(
        Trainer.confs["Classifier"]["DatasetDir"] + "/" + Trainer.confs["Classifier"]["Labels"], 'r')

    # Create a dictionary to refer each label to their string name
    for line in Trainer.labels:
        label, string_name = line.split(':')
        string_name = string_name[:-1]  # Remove newline
        Trainer.labelsToName[int(label)] = string_name

    # Create a dictionary that will help people understand your dataset better. This is required by the Dataset class later.
    Trainer.items_to_descriptions = {
        'image': 'A 3-channel RGB coloured  image that is ex: office, people',
        'label': 'A label that ,start from zero'
    }

    # Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(Trainer.confs["Classifier"]["LogDir"]):
        os.mkdir(Trainer.confs["Classifier"]["LogDir"])

    # Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        # Set the verbosity to INFO level
        tf.logging.set_verbosity(tf.logging.INFO)

        # First create the dataset and load one batch
        dataset = Trainer.getSplit('train')
        images, _, labels = Trainer.loadBatch(dataset)

        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = dataset.num_samples // Trainer.confs["Classifier"]["BatchSize"]
        # Because one step is one batch processed
        num_steps_per_epoch = num_batches_per_epoch
        decay_steps = int(
            Trainer.confs["Classifier"]["EpochsBeforeDecay"] * num_steps_per_epoch)

        # Create the model inference
        with slim.arg_scope(inception_v3_arg_scope()):
            logits, end_points = inception_v3(
                images, num_classes=dataset.num_classes, is_training=True)

        # Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

        # Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=one_hot_labels, logits=logits)
        # obtain the regularization losses as well
        total_loss = tf.losses.get_total_loss()

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate=Trainer.confs["Classifier"]["LearningRate"],
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=Trainer.confs["Classifier"]["LearningRateDecay"],
            staircase=True)

        # Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate = lr, momentum=0.9)

        # Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(
            predictions, labels)
        metrics_op = tf.group(accuracy_update, probabilities)

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()

        # Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step, epochCount):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            # Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, _ = sess.run(
                [train_op, global_step, metrics_op])
            time_elapsed = time.time() - start_time

            # Run the logging to print some results
            logging.info(' Epch %.2f Glb Stp %s: Loss: %.4f (%.2f sec/step)',
                         epochCount, global_step_count, total_loss, time_elapsed)

            return total_loss, global_step_count

        # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(
            logdir=Trainer.confs["Classifier"]["LogDir"], summary_op=None)

        # Run the managed session
        with sv.managed_session() as sess:
            for step in range(num_steps_per_epoch * Trainer.confs["Classifier"]["Epochs"]):
                # At the start of every epoch, show the vital information:
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1,
                                 Trainer.confs["Classifier"]["Epochs"])
                    learning_rate_value, accuracy_value = sess.run(
                        [lr, accuracy])
                    logging.info('Current Learning Rate: %s',
                                 learning_rate_value)
                    logging.info('Current Streaming Accuracy: %s',
                                 accuracy_value)

                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run(
                        [logits, probabilities, predictions, labels])
                    print('logits: \n', logits_value[:5])
                    print('Probabilities: \n', probabilities_value[:5])
                    print('predictions: \n', predictions_value[:100])
                    print('Labels:\n:', labels_value[:100])

                # Log the summaries every 10 step.
                if step % 10 == 0:
                    loss, _ = train_step(
                        sess, train_op, sv.global_step, step/num_batches_per_epoch + 1)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                # If not, simply run the training step
                else:
                    loss, _ = train_step(
                        sess, train_op, sv.global_step, step/num_batches_per_epoch + 1)

            # We log the final training loss and accuracy
            logging.info('Final Loss: %s', loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))

            # Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')

    checkpoint_file = tf.train.latest_checkpoint(
        Trainer.confs["Classifier"]["LogDir"])

    with tf.Graph().as_default() as graph:

        # images = tf.placeholder(shape=[None, ImageSize, ImageSize, 3], dtype=tf.float32, name = 'Placeholder_only')
        images = tf.placeholder("float", [1, Trainer.confs["Classifier"]["ImageSize"],
                                          Trainer.confs["Classifier"]["ImageSize"], 3], name="input")

        with slim.arg_scope(inception_v3_arg_scope()):
            logits, end_points = inception_v3(
                images, num_classes=Trainer.confs["Classifier"]["NumClasses"], is_training=False)
        probabilities = tf.nn.softmax(logits)
        saver = tf.train.Saver(slim.get_variables_to_restore())

        # Setup graph def
        input_graph_def = graph.as_graph_def()
        output_node_names = Trainer.confs["Classifier"]["OutputNode"]
        output_graph_name = Trainer.confs["Classifier"]["ALLGraph"]

        with tf.Session() as sess:
            saver.restore(sess, checkpoint_file)

            # Exporting the graph
            print("Exporting graph...")
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.split(","))

            with tf.gfile.GFile(output_graph_name, "wb") as f:
                f.write(output_graph_def.SerializeToString())

        clockEnd, difference, humanEnd = Trainer.Helpers.timerEnd(clockStart)

    Trainer.Helpers.logger.info(
        "ALL Detection System 2019 NCS1 Trainer ended in " + str(difference))

if __name__ == "__main__":
    run()
