# Detecting Acute Lymphoblastic Leukemia Using Caffe\*, OpenVINO™ and Intel® Neural Compute Stick 2

## Part 2: Preparing the Acute Lymphoblastic Leukemia dataset

![Detecting Acute Lymphoblastic Leukemia Using Caffe*, OpenVINO™ and Intel® Neural Compute Stick 2: Part 2](https://www.PeterMossAmlAllResearch.com/media/images/repositories/Part-2-Banner.png)

In the first part of this series: [Introduction to convolutional neural networks in Caffe](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/Python/_Caffe/allCNN/Caffe-Layers.md "Introduction to convolutional neural networks in Caffe"), I covered the steps to recreate the basics of the convolutional neural network proposed in the paper: [Acute Myeloid Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Myeloid Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System").

![Detecting Acute Lymphoblastic Leukemia Using Caffe*, OpenVINO™ and Intel® Neural Compute Stick 2: Part 2](https://www.PeterMossAmlAllResearch.com/media/images/repositories/Anh-Vo-Convolution.png)  
In this article I will cover the steps required to create the dataset required to train the model using the network we defined in the previous tutorial. The article will cover the paper exactly, and will use the original 108 image dataset (ALL_IDB1).

A reminder that we use the [ALL_IDB1 dataset from Acute Lymphoblastic Leukemia Image Database for Image Processing](https://homes.di.unimi.it/scotti/all/ "ALL_IDB1 dataset from Acute Lymphoblastic Leukemia Image Database for Image Processing") dataset, to use this dataset you must request access by visiting [this page](https://homes.di.unimi.it/scotti/all/#download "this page").

# Article Series

This is the second part of a series of articles that will take you through my experience building a custom classifier with Caffe\* that should be able to detect Acute Lymphoblastic Leukemia (ALL). I chose Caffe as I enjoyed working with it in a previous project, and I liked the intuitivity of defining the layers using prototxt files, however my R&D will include replicating both the augmentation script and the classifier using different languages and frameworks to compare results.

# Compatibility

This project has so far only been testing on Ubuntu 16.04

# Refresher

Before we begin, we can do some more visualization for the network we created in the previous article. We cloned the [AML-ALL-Classifiers](https://github.com/AMLResearchProject/AML-ALL-Classifiers/ "AML-ALL-Classifiers") repository and should have run the Setup.sh script in the allCNN project directory root. There have been some updates to the files in this repository so you should make sure you have the latest files. We can use the following command to view information about the network:

```
python3.5 Info.py NetworkInfo
```

We can save the network using:

```
python3.5 Info.py Save
```

## Additional visualization

We can also do some more visualization, using the following command we can loop through all of the 30 neurons in the conv1 and conv2 layers, saving the images inside the neurons to disk. Running the following command will write the output images to the Model/Output directory (conv1 & conv2).

```
python3.5 Info.py Outputs
```

### Conv1 output images

![Conv1 output images](https://www.PeterMossAmlAllResearch.com/media/images/repositories/Conv1-outputs.jpg)  
Figure 1. conv1 layer neuron output images

### Conv2 output images

![Conv2 output images](https://www.PeterMossAmlAllResearch.com/media/images/repositories/Conv2-outputs.jpg)  
Figure 2. conv2 layer neuron output images

# Preparing the Acute Lymphoblastic Leukemia dataset

The first thing we need to do is to sort our training and validation data. In the paper the authors state that they used the full 108 image dataset, ALL_IDB1. The paper shows that a training dataset of 80 images was used, and a validation dataset of 28. First of all we need to resize the dataset to 50px x 50px to match the input dimensions of our network, this process is handled by the functions provided in CaffeHelpers.py.

## Sorting the data

This article introduces two additional scripts in the allCNN project, [Data.py](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/Python/_Caffe/allCNN/Data.py "Data.py") & [Classes/CaffeHelpers.py](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/Python/_Caffe/allCNN/Classes/CaffeHelpers.py "Classes/CaffeHelpers.py"). These files will help us sort our data into training and validation sets, and create the LMDB databases required by Caffe.

## LMDB

[LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database "LMDB") or Lightning Mapped Database is used by Caffe to store our training/validation data and labels. In the [Classes/CaffeHelpers.py](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/Python/_Caffe/allCNN/Classes/CaffeHelpers.py "Classes/CaffeHelpers.py") file you will find a few functions that will help you convert your dataset into an LMDB database. [Data.py](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/Python/_Caffe/allCNN/Data.py "Data.py") is basically a wrapper around these functions which will do everything you need to do to create your LMDBs.

## Sort ALL_IDB1

First of all, you need to upload the ALL_IDB1 dataset to the [Model/Data/Train/0](https://github.com/AMLResearchProject/AML-ALL-Classifiers/tree/master/Python/_Caffe/allCNN/Model/Data/Train/0 "Model/Data/Train/0") and [Model/Data/Train/1](https://github.com/AMLResearchProject/AML-ALL-Classifiers/tree/master/Python/_Caffe/allCNN/Model/Data/Train/1 "Model/Data/Train/1") directories, to do this you can take the positive images from ALL_IDB1 (ending in \_1.jpg) and add to the Model/Data/Train/1 directory, then do the same for the negative images (ending in \_0.jpg).

## Proposed Training / Testing Sets

The training and validation sets proposed in the paper are as follows:

![Proposed Training / Testing Sets](https://www.PeterMossAmlAllResearch.com/media/images/repositories/ALL_IDB1_data.jpg)  
Figure 3. Training / testing data from paper ([Source](https://airccj.org/CSCP/vol7/csit77505.pdf "Source"))

## Recreating Proposed Training / Testing Sets

![Training/validation data sorting output](https://www.PeterMossAmlAllResearch.com/media/images/repositories/Data-Sizes.jpg)  
Figure 4. Training/validation data sorting output

In this article we are wanting to replicate the training and validation dataset sizes used in the paper. To ensure we get the correct training and validation sizes we use helper classes that I wrote that are provided in the Classes directory.

A reminder that we use the [ALL_IDB1 dataset from Acute Lymphoblastic Leukemia Image Database for Image Processing](https://homes.di.unimi.it/scotti/all/ "ALL_IDB1 dataset from Acute Lymphoblastic Leukemia Image Database for Image Processing") dataset, to use this dataset you must request access by visiting [this page](https://homes.di.unimi.it/scotti/all/#download "this page").

### CaffeHelpers

If you have the latest code from the repository, you should have the file: [Classes/CaffeHelpers.py](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/Python/_Caffe/allCNN/Classes/CaffeHelpers.py "Classes/CaffeHelpers.py") in the [allCNN project directory](https://github.com/AMLResearchProject/AML-ALL-Classifiers/tree/master/Python/_Caffe/allCNN/ "allCNN project directory"). This Python class will be used to handle Caffe related tasks for our network. The three main functions that we use in CaffeHelpers are recreatePaperData(), createTrainingLMDB(), createValidationLMDB() and computeMean().

#### recreatePaperData()

recreatePaperData() is the function that replicates the training and validation datasets using the sizes mentioned in the paper.

#### createTrainingLMDB()

createTrainingLMDB() is the function that converts our training dataset into an LMDB database.

#### createValidationLMDB ()

createValidationLMDB () is the function that converts our validation dataset into an LMDB database.

#### computeMean()

computeMean() is the function that removes the mean of each image.

### Data

[Data.py](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/Python/_Caffe/allCNN/Data.py "Data.py") in the [allCNN project directory](https://github.com/AMLResearchProject/AML-ALL-Classifiers/tree/master/Python/_Caffe/allCNN/ "allCNN project directory") provides an easy way to run the required functions for sorting our dataset. This file is basically a wrapper around the [CaffeHelpers class](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/Python/_Caffe/allCNN/Classes/CaffeHelpers.py "CaffeHelpers class").

### Creating the dataset

Have a quick look through the source code to familiarize yourself with what is going on, then assuming you are in the allCNN project root use the following command to run the data sorting process.

```
python3.5 Data.py
```

## Conclusion

As shown in Figure 4, we have now created training and validation datasets that match the ones used in the paper. In the next article we will train the convolutional neural network using this dataset.

# Detecting Acute Lymphoblastic Leukemia Using Caffe, OpenVino & Neural Compute Stick Series

- [Introduction to convolutional neural networks in Caffe](https://software.intel.com/en-us/articles/detecting-acute-lymphoblastic-leukemia-using-caffe-openvino-neural-compute-stick-2-part-1 "Introduction to convolutional neural networks in Caffe")
- [Preparing the Acute Lymphoblastic Leukemia dataset](https://www.linkedin.com/pulse/detecting-acute-lymphoblastic-leukemia-using-caffe-2-milton-barker-1f "Preparing the Acute Lymphoblastic Leukemia dataset")
- [Github Project](https://github.com/AMLResearchProject/AML-ALL-Classifiers/tree/master/Python/_Caffe/allCNN "Github Project")

&nbsp;

# Contributing

The Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourages and welcomes code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Team Contributors

The following Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project team members have contributed towards this repository:

- [Adam Milton-Barker](https://www.petermossamlallresearch.com/team/adam-milton-barker/profile "Adam Milton-Barker") - Bigfinite IoT Network Engineer & Intel Software Innovator, Barcelona, Spain
- [Salvatore Raieli](https://www.petermossamlallresearch.com/team/salvatore-raieli/profile "Salvatore Raieli") - PhD Immunolgy / Bioinformaticia, Bologna, Italy
- [Dr Amita Kapoor](https://www.petermossamlallresearch.com/team/amita-kapoor/profile "Dr Amita Kapoor") - Delhi University, Delhi, India

## Student Program Contributors

The following AML/ALL AI Student Program Students & Research Interns have contributed towards this repository:

- [Taru Jain](https://www.petermossamlallresearch.com/students/student/taru-jain/profile "Taru Jain") - Pre-final year undergraduate pursuing B.Tech in IT

&nbsp;

# Versioning

We use SemVer for versioning. For the versions available, see [Releases](https://github.com/AMLResearchProject/AML-ALL-Classifiers/releases "Releases").

# License

This project is licensed under the **MIT License** - see the [LICENSE](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/LICENSE "LICENSE") file for details.

# Bugs/Issues

We use the [repo issues](https://github.com/AMLResearchProject/AML-ALL-Classifiers/issues "repo issues") to track bugs and general requests related to using this project.
