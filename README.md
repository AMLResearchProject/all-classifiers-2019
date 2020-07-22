# Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
## Acute Lymphoblastic Leukemia Classifiers 2019

![Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project](Media/Images/Peter-Moss-Acute-Myeloid-Lymphoblastic-Leukemia-Research-Project.png)

[![CURRENT RELEASE](https://img.shields.io/badge/CURRENT%20RELEASE-0.1.0-blue.svg)](https://github.com/AMLResearchProject/ALL-Classifiers-2019/tree/0.1.0)
[![UPCOMING RELEASE](https://img.shields.io/badge/UPCOMING%20RELEASE-0.2.0-blue.svg)](https://github.com/AMLResearchProject/ALL-Classifiers-2019/tree/0.2.0) [![Issues Welcome!](https://img.shields.io/badge/Contributions-Welcome-lightgrey.svg)](CONTRIBUTING.md) [![Issues](https://img.shields.io/badge/Issues-Welcome-lightgrey.svg)](issues) [![LICENSE](https://img.shields.io/badge/LICENSE-MIT-blue.svg)](LICENSE)

&nbsp;

# Table Of Contents

- [Introduction](#introduction)
- [DISCLAIMER](#disclaimer)
- [Projects](#projects)
- [Data Augmentation](#data-augmentation)
- [Contributing](#contributing)
    - [Contributors](#contributors)
    - [Student Contributors](#student-contributors)
- [Versioning](#versioning)
- [License](#license)
- [Bugs/Issues](#bugs-issues) 

&nbsp;

# Introduction
The Peter Moss Acute Lymphoblastic Leukemia classifiers are a collection of projects that use computer vision to classify Acute Lymphoblastic Leukemia (ALL) in unseen images.

This repository includes classifier projects made with Tensorflow, Caffe, Keras, FastAI & Intel Movidius (NCS).

&nbsp;

# DISCLAIMER
These projects should be used for research purposes only. The purpose of the projects is to show the potential of Artificial Intelligence for medical support systems such as diagnosis systems.

Although the classifiers are accurate and show good results both on paper and in real world testing, they are not meant to be an alternative to professional medical diagnosis.

Developers that have contributed to this repository have experience in using Artificial Intelligence for detecting certain types of cancer. They are not a doctors, medical or cancer experts.

Salvatore Raieli is a bioinformatician researcher and PhD in Immunology, but does not work in medical diagnosis.

Dr Amita Kapoor is Associate Professor at SRCASW, University of Delhi, and teaches Neural Networks, Artificial Intelligence, Operating system, Embedded system, Computer Communication and Networking.

Please use these systems responsibly.

&nbsp;

# Projects

This repository hosts a collection of classifiers that have been developed by the team using the Python programming language. These classifiers include Caffe, FastAI, Movidius NCS1 and Keras classifiers, each project may have multiple classifiers.

| Projects                                                                                                                         | Description                                                 | Status  | Author  |
| -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------- | ------- |
| [Data Augmentation](Projects/Augmentation "Data Augmentation") | Applies filters to datasets and increases the amount of training / test data. | Complete | [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") |
| [AllCNN Caffe Classifier](Projects/Caffe/allCNN "AllCNN Caffe Classifier") | Acute Lymphoblastic Leukemia classifier created using the Caffe framework. | Ongoing | [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") |
| [Movidius NCS Classifier](Projects/NCS1/ "Movidius NCS Classifier") | Acute Lymphoblastic Leukemia classifier created using the Intel Movidius NCS. | Complete | [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") |
| [FastAI Resnet50 Classifier](Projects/FastAI/Resnet50/ALL-FastAI-Resnet-50.ipynb "FastAI Resnet50 Classifier")         | Acute Lymphoblastic Leukemia classifier created using FastAI & Resnet50. | Complete | [Salvatore Raieli](https://github.com/salvatorera "Salvatore Raieli") |
| [FastAI  Resnet50(a) Classifier](Projects/FastAI/Resnet50/ALL-FastAI-Resnet-50-a.ipynb "FastAI  Resnet50(a) Classifier")         | Acute Lymphoblastic Leukemia classifier created using FastAI & Resnet50. | Complete | [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") |
| [FastAI Resnet34 Classifier](Projects/FastAI/Resnet34/ALL-FastAI-Resnet-34.ipynb "FastAI Resnet34 Classifier")         | Acute Lymphoblastic Leukemia classifier created using FastAI & Resnet34. | Complete | [Salvatore Raieli](https://github.com/salvatorera "Salvatore Raieli") |
| [FastAI Resnet18 Classifier](Projects/FastAI/Resnet18/ALL-FastAI-Resnet-18.ipynb "FastAI Resnet18 Classifier")         | Acute Lymphoblastic Leukemia classifier created using FastAI & Resnet18. | Complete | [Salvatore Raieli](https://github.com/salvatorera "Salvatore Raieli") |
| [QuantisedCode](Projects/Keras/QuantisedCode/QuantisedCode.ipynb "QuantisedCode")    | Acute Lymphoblastic Leukemia classifier created using Keras with Tensorflow Backend, [Paper 1](https://airccj.org/CSCP/vol7/csit77505.pdf "Paper 1") and the original [Dataset 2](https://homes.di.unimi.it/scotti/all/#datasets "Dataset 2"). | Complete | [Dr Amita Kapoor](https://www.petermossamlallresearch.com/team/amita-kapoor/profile "Dr Amita Kapoor") & [Taru Jain](https://www.petermossamlallresearch.com/students/student/taru-jain/profile "Taru Jain") |
| [AllCNN](Projects/Keras/AllCNN/Paper_1/ALL_IDB1/Non_Augmented/AllCNN.ipynb "AllCNN") | Acute Lymphoblastic Leukemia classifier created using Keras with Tensorflow Backend, [Paper 1](https://airccj.org/CSCP/vol7/csit77505.pdf "Paper 1") and the original [Dataset 1](https://homes.di.unimi.it/scotti/all/#datasets "Dataset 1"). | Complete | [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") |
| [AllCNN](Projects/Keras/AllCNN/Paper_1/ALL_IDB2/Non_Augmented/AllCNN.ipynb "AllCNN") | Acute Lymphoblastic Leukemia classifier created using Keras with Tensorflow Backend, [Paper 1](https://airccj.org/CSCP/vol7/csit77505.pdf "Paper 1") and the original [Dataset 2](https://homes.di.unimi.it/scotti/all/#datasets "Dataset 2"). | Ongoing | [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker")  |

&nbsp;

## Team Publications

A series of articles / tutorials by [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") that take you through attempting to replicate the work carried out in the [Acute Myeloid Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Myeloid Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper.

- [Acute Lymphoblastic Leukemia Data Augmentation (Intel® AI Developer Program)](https://software.intel.com/en-us/articles/acute-myeloidlymphoblastic-leukemia-data-augmentation "Acute Lymphoblastic Leukemia Data Augmentation (Intel® AI Developer Program)") - [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker")

- [Inception V3 Deep Convolutional Architecture For Classifying Acute Myeloid/Lymphoblastic Leukemia (Intel® AI Developer Program)](https://software.intel.com/en-us/articles/inception-v3-deep-convolutional-architecture-for-classifying-acute-myeloidlymphoblastic "Inception V3 Deep Convolutional Architecture For Classifying Acute Myeloid/Lymphoblastic Leukemia (Intel® AI Developer Program)") - [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker")

- [Introduction to convolutional neural networks in Caffe\*](https://software.intel.com/content/www/us/en/develop/articles/detecting-acute-lymphoblastic-leukemia-using-caffe-openvino-neural-compute-stick-2-part-1.html "Introduction to convolutional neural networks in Caffe*") - [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker")

- [Preparing the Acute Lymphoblastic Leukemia dataset](https://software.intel.com/content/www/us/en/develop/articles/detecting-acute-lymphoblastic-leukemia-using-caffe-openvino-neural-compute-stick-2-part-2.html "Preparing the Acute Lymphoblastic Leukemia dataset") - [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker")

&nbsp;

# Contributing

The Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourages and welcomes code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Contributors

- [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") - [Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss") President & Lead Developer, Sabadell, Spain

- [Salvatore Raieli](https://www.leukemiaresearchassociation.ai/team/salvatore-raieli  "Salvatore Raieli") - [Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss") Bioinformatics & Immunology AI R&D, Bologna, Italy

- [Dr Amita Kapoor](https://www.leukemiaresearchassociation.ai/team/amita-kapoor "Dr Amita Kapoor") - [Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss") Student Program Team / R&D, Delhi, India

## Students Contributors

- [Taru Jain](https://www.leukemiaresearchassociation.ai/student-program/student/taru-jain "Taru Jain") - [Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss") Student Program, Delhi, India

&nbsp;

# Versioning

We use SemVer for versioning. For the versions available.

&nbsp;

# License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE "LICENSE") file for details.

&nbsp;

# Bugs/Issues

We use the [repo issues](issues "repo issues") to track bugs and general requests related to using this project.
