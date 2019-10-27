# Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project

## Acute Myeloid & Lymphoblastic Leukemia Python Classifiers

![Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project](https://www.PeterMossAmlAllResearch.com/media/images/banner.png)

### Keras Classifiers

#### AllCNN Classifier (Paper 1/Dataset 1/Non Augmented)

The Peter Moss Acute Myeloid & Lymphoblastic Leukemia Keras AllCNN Paper 1/Dataset 1/Non Augmented Classifier uses the [ACUTE LEUKEMIA CLASSIFICATION USING CONVOLUTION NEURAL NETWORK IN CLINICAL DECISION SUPPORT SYSTEM](https://airccj.org/CSCP/vol7/csit77505.pdf "ACUTE LEUKEMIA CLASSIFICATION USING CONVOLUTION NEURAL NETWORK IN CLINICAL DECISION SUPPORT SYSTEM") paper and the original (non augmented) [ALL_IDB1](https://homes.di.unimi.it/scotti/all/#datasets "ALL_IDB1") dataset.

&nbsp;

# Model Architecture

<img src="https://www.PeterMossAmlAllResearch.com/media/images/repositories/paper_1_architecture.png" alt="Proposed Architecture" />

_Fig 1. Proposed Architecture ([Source](https://airccj.org/CSCP/vol7/csit77505.pdf "Source"))_

## Proposed Architecture

In the [ACUTE LEUKEMIA CLASSIFICATION USING CONVOLUTION NEURAL NETWORK IN CLINICAL DECISION SUPPORT SYSTEM](https://airccj.org/CSCP/vol7/csit77505.pdf "ACUTE LEUKEMIA CLASSIFICATION USING CONVOLUTION NEURAL NETWORK IN CLINICAL DECISION SUPPORT SYSTEM") paper the authors explain the layers they used to create their convolutional neural network.

> "In this work, we proposed a network contains 4 layers. The first 3 layers for detecting features
> and the other two layers (Fully connected and Softmax) are for classifying the features. The input
> image has the size [50x50x3]. The receptive field (or the filter size) is 5x5. The stride is 1 then we move the filters one pixel at a time. The zero-padding is 2. It will allow us to control the spatial
> size of the output image (we will use it to exactly preserve the spatial size of the input volume so
> the input and output width and height are the same). During the experiment, we found that in our
> case, altering the size of original image during the convolution lead to decrease the accuracy
> about 40%. Thus the output image after convolution layer 1 has the same size with the input
> image."

> "The convolution layer 2 has the same structure with the convolution layer 1. The filter size is 5x5,
> the stride is 1 and the zero-padding is 2. The number of feature maps (the channel or the depth) in
> our case is 30. If the number of feature maps is lower or higher than 30, the accuracy will
> decrease 50%. By experiment, we found the accuracy also decrease 50% if we remove
> Convolution layer 2.""

> "The Max-Pooling layer 25x25 has Filter size is 2 and stride is 2. The fully connected layer has 2
> neural. Finally, we use the Softmax layer for the classification. "

## Proposed Training / Validation Sets

In the paper the authors use the **ALL_IDB1** dataset. The paper proposes the following training and validation sets proposed in the paper, where **Normal cell** refers to ALL negative examples and **Abnormal cell** refers to ALL positive examples.

|               | Training Set | Test Set |
| ------------- | ------------ | -------- |
| Normal cell   | 40           | 19       |
| Abnormal cell | 40           | 9        |
| **Total**     | **80**       | **28**   |

You can view the notebook using **ALL_IDB1** here. In this notebook however, you are going to use the **ALL_IDB2** dataset. On [Fabio Scotti's ALL-IDB website](https://homes.di.unimi.it/scotti/all), Fabio provides a [guideline for reporting your results when using ALL-IDB](https://homes.di.unimi.it/scotti/all/results.php). In this guideline a benchmark is proposed, this benchmark includes testing with both **ALL_IDB1** & **ALL_IDB2**:

> "A system capable to identify the presence of blast cells in the input image can work with different structures of modules, for example, it can processes the following steps: (i) the identification of white cells in the image, (ii) the selection of Lymphocytes, (iii) the classification of tumor cell. Each single step typically contains segmentation/ classification algorithms. In order to measure and fairly compare the identification accuracy of different structures of modules, we propose a benchmark approach partitioned in three different tests, as follows:"

- Cell test - the benchmark account for the classification of single cells is blast or not (the test is positive if the considered cell is blast cell or not);
- Image level - the whole image is classified (the test is positive if the considered image contains at least one blast cell or not).

In the paper the authors do not cover using **ALL_IDB2**. As ALL_IDB2 has an equal amount of images in each class (130 per class) you will use the entire ALL_IDB2 dataset with a test split of 20%.

&nbsp;

# Clone AML & ALL Classifiers Repository

First of all you should clone the [AML & ALL Classifiers](https://github.com/AMLResearchProject/AML-ALL-Classifiers/ "AML & ALL Classifiers") repo to your device. To do this you can navigate to the location you want to clone the repository to on your device using terminal (cd Your/Clone/Location), and then use the following command:

```
  $ git clone https://github.com/AMLResearchProject/AML-ALL-Classifiers.git
```

Once you have used the command above you will see a directory called **AML-ALL-Classifiers** in the location you chose to clone the repo to. In terminal, navigate to the **AML-ALL-Classifiers/Python/\_Keras/AllCNN/Paper_1/ALL_IDB1/Non_Augmented/** directory, this is your project root directory.

&nbsp;

# Upload Project Root To Google Drive

Now you need to upload the project root to your Google Drive, placing the jpg files from the ALL_IDB1 dataset in the **Model/Data/Training/** directory. Once you have done this open **AML-ALL-Classifiers/Python/\_Keras/AllCNN/Paper_1/ALL_IDB1/Non_Augmented/AllCNN.ipynb** in Google Colab and continue from the **Google Drive / Colab** section to complete the project.

&nbsp;

# Results on ALL-IDB (Images)

In the paper the authors got 96.43% using a Matlab classifier. This network got around 93% accuracy so improvements should be made.

## Training Results

Below are the training results for 100 epochs.

| Loss          | Accuracy     | Precision     | Recall       | AUC          |
| ------------- | ------------ | ------------- | ------------ | ------------ |
| 0.131 (~0.13) | 0.928 (~93%) | 0.904 (~0.91) | 0.999 (~1.0) | 0.994 (~1.0) |

## Overall Results

| Figures of merit     | Value | Percentage |
| -------------------- | ----- | ---------- |
| True Positives       | 2     | 7.14%      |
| False Positives      | 7     | 25.00%     |
| True Negatives       | 19    | 67.96%     |
| False Negatives      | 0     | 0.00%      |
| Misclassification    | 7     | 25.00%     |
| Sensitivity / Recall | 1.0   | 100%       |
| Specificity          | 0.73  | 73%        |

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
