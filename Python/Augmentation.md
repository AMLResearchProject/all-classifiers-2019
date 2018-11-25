# AML Classifier Data Augmentation
<img style="width: 100%;" src="../Media/Images/Banner-Social.jpg" title="Peter Moss Acute Myeloid Leukemia Research Project">

The AML Classifier Data Augmentation program applies filters to the original dataset and increases the amount of training / test data. The AML Classifier Data Augmentation program is part of the computer vision research and development for the Peter Moss Acute Myeloid Leukemia Research Project. 

I decided to use some augmentation proposals outlined in Leukemia Blood Cell Image Classification Using Convolutional Neural Network by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon. The augmentations I chose were grayscaling, histogram equalization, horizontal and vertical reflection and gaussian blur to start with. Using these techniques so far I have been able to increase a dataset from 49 positive and 49 negative images to 294 positive and 294 negative, with more augmentations to experiment with.

# Research papers followed
The papers that this part of the project is based on were provided by project team member, Ho Leung, Associate Professor of Biochemistry & Molecular Biophysics at Kansas State University. 

## Leukemia Blood Cell Image Classification Using Convolutional Neural Network
T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon 
http://www.ijcte.org/vol10/1198-H0012.pdf

# Dataset
The [Acute Lymphoblastic Leukemia Image Database for Image Processing](https://homes.di.unimi.it/scotti/all/) dataset is used for this project. The dataset was created by [Fabio Scotti, Associate Professor Dipartimento di Informatica, Università degli Studi di Milano](https://homes.di.unimi.it/scotti/). Big thanks to Fabio for his research and time put in to creating the dataset and documentation, it is one of his personal projects. You will need to follow the steps outlined [here](https://homes.di.unimi.it/scotti/all/#download) to gain access to the dataset.

## Data augmentation
<img style="width: 100%;" src="Media/Images/slides.png" title="Data augmentation example">
<div style="clear:both;"> 

The full Python class that holds the functions mentioned below can be found in [Classes/Data.py](Classes/Data.py), The Data class is a wrapper class around releated functions provided in popular computer vision libraries including as OpenCV and Scipy.

# Clone the code from the Github repo
You will need to clone the project from our Github to which ever device you are going to run it on. 

```
 $ git clone https://github.com/AMLResearchProject/AML-Classifiers.git
```

# Dataset Access
The [Acute Lymphoblastic Leukemia Image Database for Image Processing](https://homes.di.unimi.it/scotti/all/) by [Fabio Scotti, Associate Professor Dipartimento di Informatica, Università degli Studi di Milano](https://homes.di.unimi.it/scotti/) is used with this project, you can request access by following the unstructions on the [Download and Term of use](https://homes.di.unimi.it/scotti/all/#download) page, you can also view [Reporting the results on ALL-IDB](https://homes.di.unimi.it/scotti/all/results.php) for information on how to organize and submit your findings.

Once you have access to the dataset, you should add your dataset to the 0 & 1 directories in the Model/Data directory, if you configure the same way you do not need to change any settings. I used 49 images from each folder resulting in a training / testing set of 98 images before data augmentation.

# The Augmentation Notebook
In the project Data folder you will find a Juypter Notebook named [Augmentation.ipynb](https://github.com/AMLResearchProject/AML-Classifiers/tree/master/Python/Augmentation.ipynb "Augmentation.ipynb"). This Notebook seems to not run well on Github, but if you have cloned the repo you should be able to launch the Notebook fine, the Notebook provides a full walk through of the data augmentation program.

# Contributing
We welcome contributions of the project. Please read [CONTRIBUTING.md](https://github.com/AMLResearchProject/AML-Classifiers/blob/master/CONTRIBUTING.md "CONTRIBUTING.md") for details on our code of conduct, and the process for submitting pull requests.

# Versioning
We use SemVer for versioning. For the versions available, see [Releases](https://github.com/AMLResearchProject/AML-Classifiers/releases "Releases").

# License
This project is licensed under the **MIT License** - see the [LICENSE](https://github.com/AMLResearchProject/AML-Classifiers/blob/master/LICENSE "LICENSE") file for details.

# Bugs/Issues
We use the [repo issues](issues "repo issues") to track bugs and general requests related to using this project. 

# About The Author
Adam is a [BigFinite](https://www.bigfinite.com "BigFinite") IoT Network Engineer, part of the team that works on the core IoT software for our platform. In his spare time he is an [Intel Software Innovator](https://software.intel.com/en-us/intel-software-innovators/overview "Intel Software Innovator") in the fields of Internet of Things, Artificial Intelligence and Virtual Reality.

[![Adam Milton-Barker: BigFinte IoT Network Engineer & Intel® Software Innovator](../Media/Images/Adam-Milton-Barker.jpg)](https://github.com/AdamMiltonBarker)