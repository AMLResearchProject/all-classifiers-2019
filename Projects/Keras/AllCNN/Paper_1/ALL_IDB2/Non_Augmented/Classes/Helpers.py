############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    AML/ALL Classifiers
# Project:       Keras AllCNN
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Helpers Class
# Description:   Helper functions used with Keras AllCNN.
# License:       MIT License
# Last Modified: 2019-10-26
#
############################################################################################

import sys
import time
import logging
import logging.handlers as handlers
import json

from datetime import datetime


class Helpers():
    """ Helpers Class

    Helper functions used with Keras AllCNN.
    """

    def __init__(self, logger_type, path):
        """ Initializes the Helpers Class. """

        self.confs = {}
        self.path = path

        self.loadConfs()

        self.logger = logging.getLogger(logger_type)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        allLogHandler = handlers.TimedRotatingFileHandler(
            self.path + 'Logs/all.log', when='H', interval=1, backupCount=0)
        allLogHandler.setLevel(logging.INFO)
        allLogHandler.setFormatter(formatter)

        errorLogHandler = handlers.TimedRotatingFileHandler(
            self.path + 'Logs/error.log', when='H', interval=1, backupCount=0)
        errorLogHandler.setLevel(logging.ERROR)
        errorLogHandler.setFormatter(formatter)

        warningLogHandler = handlers.TimedRotatingFileHandler(
            self.path + 'Logs/warning.log', when='H', interval=1, backupCount=0)
        warningLogHandler.setLevel(logging.WARNING)
        warningLogHandler.setFormatter(formatter)

        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(formatter)

        self.logger.addHandler(allLogHandler)
        self.logger.addHandler(errorLogHandler)
        self.logger.addHandler(warningLogHandler)
        self.logger.addHandler(consoleHandler)

        self.logger.info("Helpers class initialization complete.")

    def loadConfs(self):
        """ Load the program configuration. """

        with open(self.path + 'config.json') as confs:
            self.confs = json.loads(confs.read())
