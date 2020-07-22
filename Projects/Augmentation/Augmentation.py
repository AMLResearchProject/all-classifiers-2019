############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    ALL Detection System 2019
# Project:       Data Augmentation
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Manual Data Augmentation Class
# Description:   Manual data augmentation class for the ALL Detection System 2019.
# License:       MIT License
# Last Modified: 2020-07-14
#
############################################################################################

import matplotlib.pyplot as plt

from Classes.Data import Data

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


class Augmentation():
    """ ALL Detection System 2019 Manual Data Augmentation Class

    Manual data augmentation wrapper class for the ALL Detection System 2019 Data Augmentation project.
    """

    def __init__(self):
        """ Initializes the Augmentation class. """

        self.Data = Data()

    def processDataset(self):
        """ Processes the AML/ALL Detection System Dataset. 
        Make sure you have your equal amounts of positive and negative
        samples in the Model/Data directories.

        Only run this function once! it will continually make copies
        of all images in the Settings->TrainDir directory specified
        in Required/confs.json
        """

        self.Data.processDataset()


print("!! Data Augmentation Program Starting !!")
print("")
Augmentation = Augmentation()
Augmentation.processDataset()
print(" Data Augmentation Program Complete")
print("")
