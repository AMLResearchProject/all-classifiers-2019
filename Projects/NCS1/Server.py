############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    ALL Detection System 2019
# Project:       Facial Authentication Server
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Server Class
# Description:   Server for the ALL Detection System 2019 Classifier.
# License:       MIT License
# Last Modified: 2020-07-21
#
############################################################################################

import cv2, json, jsonpickle, os, sys, time

import numpy as np

from mvnc import mvncapi as mvnc
from datetime import datetime
from skimage.transform import resize

from Classes.Helpers import Helpers
from Classes.Movidius import Movidius

from flask import Flask, request, Response


class Server():
    """ ALL Detection System 2019 Server Class

    Server for the ALL Detection System 2019 Classifier.
    """

    def __init__(self):
        """ Initializes the Server Class. """

        self.Helpers = Helpers("Server")
        self.confs = self.Helpers.confs

        self.Movidius = Movidius()
        self.Movidius.checkNCS()
        self.Movidius.loadInception()

        self.Helpers.logger.info(
            "Server class initialization complete.")


app = Flask(__name__)
Server = Server()


@app.route('/Inference', methods=['POST'])
def Inference():
    
    if len(request.files) != 0:
        img = np.fromstring(request.files['file'].read(), np.uint8)
    else:
        img = np.fromstring(request.data, np.uint8)

    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED).astype(np.float32)

    dx, dy, dz = img.shape
    delta = float(abs(dy-dx))

    if dx > dy:
        img = img[int(0.5*delta):dx-int(0.5*delta), 0:dy]
    else:
        img = img[0:dx, int(0.5*delta):dy-int(0.5*delta)]

    img = cv2.resize(img, (Server.Movidius.reqsize,
                           Server.Movidius.reqsize))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(3):
        img[:, :, i] = (img[:, :, i] - Server.Movidius.mean) * \
            Server.Movidius.std

    detectionStart, detectionStart = Server.Helpers.timerStart()
    Server.Helpers.logger.info("API detection started.")

    Server.Movidius.ncsGraph.LoadTensor(img.astype(np.float16), 'user object')
    output, userobj = Server.Movidius.ncsGraph.GetResult()

    detectionClockEnd, difference, detectionEnd = Server.Helpers.timerEnd(
        detectionStart)
    Server.Helpers.logger.info("API detection ended taking " + str(difference))

    top_inds = output.argsort()[::-1][:5]

    if output[top_inds[0]] >= Server.confs["Classifier"]["InceptionThreshold"] and Server.Movidius.classes[top_inds[0]] == "1":
        classification = "AML Positive"
        message = "ALL detected with a confidence of " + \
            str(output[top_inds[0]]) + " in " + str(difference)
    elif output[top_inds[0]] >= Server.confs["Classifier"]["InceptionThreshold"] and Server.Movidius.classes[top_inds[0]] == "0":
        classification = "AML Negative"
        message = "ALL not detected with a confidence of " + \
            str(output[top_inds[0]]) + " in " + str(difference)
    elif output[top_inds[0]] <= Server.confs["Classifier"]["InceptionThreshold"] and Server.Movidius.classes[top_inds[0]] == "1":
        classification = "AML Positive"
        message = "ALL detected with a LOW confidence of " + \
            str(output[top_inds[0]]) + " in " + str(difference)
    elif output[top_inds[0]] <= Server.confs["Classifier"]["InceptionThreshold"] and Server.Movidius.classes[top_inds[0]] == "0":
        classification = "AML Negative"
        message = "ALL not detected with a LOW confidence of " + \
            str(output[top_inds[0]]) + " in " + str(difference)

    Server.Helpers.logger.info(message)

    ServerResponse = jsonpickle.encode({
        'Response': 'OK',
        'Classification': classification,
        'aClassification': Server.Movidius.classes[top_inds[0]],
        'Confidence': str(output[top_inds[0]]),
        'Message': message
    })

    return Response(response=ServerResponse, status=200, mimetype="application/json")


if __name__ == "__main__":
    app.run(host=Server.confs["Classifier"]["IP"],
            port=Server.confs["Classifier"]["Port"])
