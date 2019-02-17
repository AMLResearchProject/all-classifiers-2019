# Inception V3 Deep Learing Convolutional Architecture 
![Peter Moss Acute Myeloid/Lymphoblastic Leukemia Detection System](../Media/Images/banner.png)
Inception V3 by Google is the 3rd version in a series of Deep Learning Convolutional Architectures. Inception V3 was trained using a dataset of 1000 classes from the original ImageNet dataset which was trained with over 1 million training images. 

This README will take you through some information about Inception V3, transfer learning, and how we use these tools in the Acute Myeloid/Lymphoblastic Leukemia AI Research Project.

# Convolutional Neural Networks
![Inception v3 architecture](../Media/Images/CNN.jpg)  

Figure 1. Inception v3 architecture ([Source](https://github.com/tensorflow/models/tree/master/research/inception)).

Convolutional neural networks are a type of deep learning neural network. These types of neural nets are widely used in computer vision and have pushed the capabilities of computer vision over the last few years, performing exceptionally better than older, more traditional neural networks; however, studies show that there are trade-offs related to training times and accuracy.

# Transfer Learning
![Inception v3 model diagram](../Media/Images/Transfer-Learning.jpg)  

Figure 2. Inception V3 Transfer Learning ([Source](https://github.com/Hvass-Labs/TensorFlow-Tutorials)).

Transfer learning allows you to retrain the final layer of an existing model, resulting in a significant decrease in not only training time, but also the size of the dataset required. One of the most famous models that can be used for transfer learning is the Inception V3 model created by Google This model was trained on thousands of images from 1,001 classes on some very powerful devices. Being able to retrain the final layer means that you can maintain the knowledge that the model had learned during its original training and apply it to your smaller dataset, resulting in highly accurate classifications without the need for extensive training and computational power.

# TensorFlow-Slim image classification model library
TF-Slim is a high-level API for Tensorflow that allows you to program, train and evaluate Convolutional Neural Networks. TF-Slim is a lightweight API so is well suited for lower powered devices.

Github: [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim)

The Acute Myeloid/Lymphoblastic Leukemia AI Research Project Movidius NCS Classifier uses the following classes from the TensorFlow-Slim image classification model library:

- [inception_preprocessing.py](https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py)
- [inception_utils.py](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_utils.py)
- [inception_v3.py](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py)

In the project you will find these files in the [AML-ALL-Classifiers/Python/_Movidius/NCS/Classes](https://github.com/AdamMiltonBarker/AML-ALL-Classifiers/tree/master/Python/_Movidius/NCS/Classes/) directory.

## inception_preprocessing.py
[AML-ALL-Classifiers/Python/_Movidius/NCS/Classes/inception_preprocessing.py](https://github.com/AdamMiltonBarker/AML-ALL-Classifiers/tree/master/Python/_Movidius/NCS/Classes/inception_preprocessing.py)
The inception_preprocessing class provides the tools required to preprocess both training and evaluation images allowing them to be used with Inception Networks.

## inception_utils.py
[AML-ALL-Classifiers/Python/_Movidius/NCS/Classes/inception_utils.py](https://github.com/AdamMiltonBarker/AML-ALL-Classifiers/tree/master/Python/_Movidius/NCS/Classes/inception_utils.py)

## inception_v3.py
[AML-ALL-Classifiers/Python/_Movidius/NCS/Classes/inception_v3.py](https://github.com/AdamMiltonBarker/AML-ALL-Classifiers/tree/master/Python/_Movidius/NCS/Classes/inception_v3.py)
The v class provides the code required to create an Inception V3 network.

# Contributing
We welcome contributions of the project. Please read [CONTRIBUTING.md](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/CONTRIBUTING.md "CONTRIBUTING.md") for details on our code of conduct, and the process for submitting pull requests.

# Versioning
We use SemVer for versioning. For the versions available, see [Releases](https://github.com/AMLResearchProject/AML-ALL-Classifiers/releases "Releases").

# License
This project is licensed under the **MIT License** - see the [LICENSE](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/LICENSE "LICENSE") file for details.

# Bugs/Issues
We use the [repo issues](https://github.com/AMLResearchProject/AML-ALL-Classifiers/issues "repo issues") to track bugs and general requests related to using this project. 

# Repository Manager
Adam is a [BigFinite](https://www.bigfinite.com "BigFinite") IoT Network Engineer, part of the team that works on the core IoT software. In his spare time he is an [Intel Software Innovator](https://software.intel.com/en-us/intel-software-innovators/overview "Intel Software Innovator") in the fields of Internet of Things, Artificial Intelligence and Virtual Reality.

[![Adam Milton-Barker: BigFinte IoT Network Engineer & Intel® Software Innovator](../Media/Images/Adam-Milton-Barker.jpg)](https://github.com/AdamMiltonBarker)