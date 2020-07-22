############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    ALL Detection System 2019
# Project:       Facial Authentication Server
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Helpers Class
# Description:   Helpers class for the ALL Detection System 2019 NCS1 Classifier.
# License:       MIT License
# Last Modified: 2020-07-16
#
############################################################################################

import json, logging, sys, time
import logging.handlers as handlers

from datetime import datetime


class Helpers():
    """ Helper Class

    Common helper functions for the ALL Detection System 2019 NCS1 Classifier.
    """

    def __init__(self, loggerType):
        """ Initializes the Helpers Class. """

        self.confs = {}
        self.loadConfs()

        self.logger = logging.getLogger(loggerType)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        allLogHandler = handlers.TimedRotatingFileHandler(
            self.confs["Settings"]["Logs"] + 'all.log', when='H', interval=1, backupCount=0)
        allLogHandler.setLevel(logging.INFO)
        allLogHandler.setFormatter(formatter)

        errorLogHandler = handlers.TimedRotatingFileHandler(
            self.confs["Settings"]["Logs"] + 'error.log', when='H', interval=1, backupCount=0)
        errorLogHandler.setLevel(logging.ERROR)
        errorLogHandler.setFormatter(formatter)

        warningLogHandler = handlers.TimedRotatingFileHandler(
            self.confs["Settings"]["Logs"] + 'warning.log', when='H', interval=1, backupCount=0)
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
        """ Load the  ALL Detection System 2019 NCS1 Classifier configuration. """

        with open('Required/confs.json') as confs:
            self.confs = json.loads(confs.read())

    def currentDateTime(self):
        """ Gets the current date and time in words. """

        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def timerStart(self):
        """ Starts the timer. """

        return str(datetime.now()), time.time()

    def timerEnd(self, start):
        """ Ends the timer. """

        return time.time(), (time.time() - start), str(datetime.now())
