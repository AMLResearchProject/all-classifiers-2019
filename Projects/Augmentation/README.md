# Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
## Acute Lymphoblastic Leukemia Classifiers 2019
### Data Augmentation

![Peter Moss Acute Myeloid & Lymphoblastic Leukemia Research Project](Media/Images/ALL_IDB1-Augmentation.png)

&nbsp;

# Table Of Contents

- [Introduction](#introduction)
- [Projects](#projects)
- [Research Papers Followed](#research-papers-followed)
- [Dataset Used](#dataset-used)
- [Data augmentation](#data-augmentation)
    - [Resizing](#resizing)
    - [Grayscaling](#grayscaling)
    - [Histogram Equalization](#histogram-equalization)
    - [Reflection](#reflection)
    - [Gaussian Blur](#gaussian-blur)
    - [Translation](#translation)
    - [Rotation](#rotation)
- [System Requirements](#system-requirements)
- [Setup](#setup)
    - [Clone the repository](#clone-the-repository)
        - [Developer Forks](#developer-forks)
    - [Install Requirements](#install-requirements)
    - [Sort your dataset](#sort-your-dataset)
- [Run locally](#run-locally)
- [Run using Jupyter Notebook](#run-using-jupyter-notebook)
- [Your augmented dataset](#your-augmented-dataset)
- [Contributing](#contributing)
    - [Contributors](#contributors)
- [Versioning](#versioning)
- [License](#license)
- [Bugs/Issues](#bugs-issues)

&nbsp;

# Introduction
The Acute Lymphoblastic Leukemia Detection System 2019 Data Augmentation program applies augmentations/filters to datasets and increases the amount of training/test data available to use. The program is part of the computer vision research and development for the Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project. This page will provide general information, as well as a guide for installing and setting up the augmentation script.

&nbsp;

# Projects 
| Project                                                                                                                                                                                           | Description                                                                                                             | Author                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| [Data Augmentation Using Python](Augmentation.py "Data Augmentation Using Python")                              | A Python program for applying filters to datasets to increase the amount of training / test data.                       | [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") |
| [Data Augmentation Using Jupyter Notebook](Augmentation.ipynb "Data Augmentation Using Jupyter Notebook") | A Python tutorial and Jupyter Notebook for applying filters to datasets to increase the amount of training / test data. | [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") |

&nbsp;

# Research papers followed

The Acute Lymphoblastic Leukemia Detection System 2019 uses the data augmentation methods proposed in the [Leukemia Blood Cell Image Classification Using Convolutional Neural Network by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon](http://www.ijcte.org/vol10/1198-H0012.pdf "Leukemia Blood Cell Image Classification Using Convolutional Neural Network by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon").

| Paper                                                                       | Description                                                                  | Link                                                       |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Leukemia Blood Cell Image Classification Using Convolutional Neural Network | T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon | [Paper](http://www.ijcte.org/vol10/1198-H0012.pdf "Paper") |

&nbsp;

# Dataset Used

The [Acute Lymphoblastic Leukemia Image Database for Image Processing](https://homes.di.unimi.it/scotti/all/) dataset is used for this project. The dataset was created by [Fabio Scotti, Associate Professor Dipartimento di Informatica, Università degli Studi di Milano](https://homes.di.unimi.it/scotti/). Big thanks to Fabio for his research and time put in to creating the dataset and documentation, it is one of his personal projects. You will need to follow the steps outlined [here](https://homes.di.unimi.it/scotti/all/#download) to gain access to the dataset.

| Dataset                                                          | Description                                                                                                                                      | Link                                                                |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| Acute Lymphoblastic Leukemia Image Database for Image Processing | Created by [Fabio Scotti, Associate Professor Dipartimento di Informatica, Università degli Studi di Milano](https://homes.di.unimi.it/scotti/). | [Dataset](https://homes.di.unimi.it/scotti/all/#download "Dataset") |

&nbsp;

# Data augmentation

![AML & ALL Data Augmentation](Media/Images/ALL_IDB1-Augmented-Slides.png)

In this dataset there were 49 negative and 59 positive. To make this even I removed 10 images from the negative dataset. From here I removed a further 10 images per class for testing further on in the tutorial and for the purpose of demos etc. 

In my case I had 20 test images (10 pos/10 neg) and 39 images per class ready for augmentation. Place the original images that you wish to augment into the **Model/Data/0** & **Model/Data/1**. Using this program I was able to create a dataset of **1053** positive and **1053** negative augmented images.

The full Python class that holds the functions mentioned below can be found in [Classes/Data.py](Classes/Data.py), The Data class is a wrapper class around releated functions provided in popular computer vision libraries including as OpenCV and Scipy.

## Resizing

The first step is to resize the image this is done with the following function:

```
    def resize(self, filePath, savePath, show = False):

        """
        Writes an image based on the filepath and the image provided.
        """

        image = cv2.resize(cv2.imread(filePath), self.fixed)
        self.writeImage(savePath, image)
        self.filesMade += 1
        print("Resized image written to: " + savePath)

        if show is True:
            plt.imshow(image)
            plt.show()

        return image
```

## Grayscaling

In general grayscaled images are not as complex as color images and result in a less complex model. In the paper the authors described using grayscaling to create more data easily. To create a greyscale copy of each image I wrapped the built in OpenCV function, [cv2.cvtColor()](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html). The created images will be saved to the relevant directories in the default configuration.

```
    def grayScale(self, image, grayPath, show = False):

        """
        Writes a grayscale copy of the image to the filepath provided.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.writeImage(grayPath, gray)
        self.filesMade += 1
        print("Grayscaled image written to: " + grayPath)

        if show is True:
            plt.imshow(gray)
            plt.show()

        return image, gray
```
## Histogram Equalization

Histogram equalization is basically stretching the histogram horizontally on both sides, increasing the intensity/contrast. Histogram equalization is described in the paper to enhance the contrast.

In the case of this dataset, it makes both the white and red blood cells more distinguishable. The created images will be saved to the relevant directories in the default configuration.

```
    def equalizeHist(self, gray, histPath, show = False):

        """
        Writes histogram equalized copy of the image to the filepath provided.
        """

        hist = cv2.equalizeHist(gray)
        self.writeImage(histPath, cv2.equalizeHist(gray))
        self.filesMade += 1
        print("Histogram equalized image written to: " + histPath)

        if show is True:
            plt.imshow(hist)
            plt.show()

        return hist
```

## Reflection

Reflection is a way of increasing your dataset by creating a copy that is fliped on its X axis, and a copy that is flipped on its Y axis. The reflection function below uses the built in OpenCV function, cv2.flip, to flip the image on the mentioned axis. The created images will be saved to the relevant directories in the default configuration.

```
    def reflection(self, image, horPath, verPath, show = False):

        """
        Writes reflected copies of the image to the filepath provided.
        """

        horImg = cv2.flip(image, 0)
        self.writeImage(horPath, horImg)
        self.filesMade += 1
        print("Horizontally reflected image written to: " + horPath)

        if show is True:
            plt.imshow(horImg)
            plt.show()

        verImg = cv2.flip(image, 1)
        self.writeImage(verPath, verImg)
        self.filesMade += 1
        print("Vertical reflected image written to: " + verPath)

        if show is True:
            plt.imshow(verImg)
            plt.show()

        return horImg, verImg
```
## Gaussian Blur

Gaussian Blur is a popular technique used on images and is especially popular in the computer vision world. The function below uses the ndimage.gaussian_filter function. The created images will be saved to the relevant directories in the default configuration.

```
    def gaussian(self, filePath, gaussianPath, show = False):

        """
        Writes gaussian blurred copy of the image to the filepath provided.
        """

        gaussianBlur = ndimage.gaussian_filter(plt.imread(filePath), sigma=5.11)
        self.writeImage(gaussianPath, gaussianBlur)
        self.filesMade += 1
        print("Gaussian image written to: " + gaussianPath)

        if show is True:
            plt.imshow(gaussianBlur)
            plt.show()

        return gaussianBlur
```

## Translation

Translation is a type of Affine Transformation and basically repositions the image within itself. The function below uses the cv2.warpAffine function. The created images will be saved to the relevant directories in the default configuration.

```
    def translate(self, image, translatedPath, show = False):

        """
        Writes transformed copy of the image to the filepath provided.
        """

        cols, rows, chs = image.shape

        translated = cv2.warpAffine(image, np.float32([[1, 0, 84], [0, 1, 56]]), (rows, cols),
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(144, 159, 162))

        self.writeImage(filePath, translated)
        self.filesMade += 1
        print("Translated image written to: " + filePath)

        if show is True:
            plt.imshow(translated)
            plt.show()

        return translated
```

## Rotation

Gaussian Blur is a popular technique used on images and is especially popular in the computer vision world. The function below uses the ndimage.gaussian_filter function. The created images will be saved to the relevant directories in the default configuration.

```
    def rotation(self, path, filePath, filename, show=False):
            """
            Writes rotated copies of the image to the filepath provided. 
            """

            img = Image.open(filePath)

            image = cv2.imread(filePath)
            cols, rows, chs = image.shape

            for i in range(0, 20):
                # Seed needs to be set each time randint is called
                random.seed(self.seed)
                randDeg = random.randint(-180, 180)
                matrix = cv2.getRotationMatrix2D((cols/2, rows/2), randDeg, 0.70)
                rotated = cv2.warpAffine(image, matrix, (rows, cols), borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(144, 159, 162))
                fullPath = os.path.join(
                    path, str(randDeg) + '-' + str(i) + '-' + filename)

                self.writeImage(fullPath, rotated)
                self.filesMade += 1
                print("Rotated image written to: " + fullPath)

                if show is True:
                    plt.imshow(rotated)
                    plt.show()
```

&nbsp;

# System Requirements

- Tested on Ubuntu 20.04, 18.04 & 16.04
- [Tested with Python 3.0 and above](https://www.python.org/download/releases/3.0/ "Tested with Python 3.0 and above")
- Requires PIP3
- Jupyter Notebook (Optional)

&nbsp;

# Setup

Below is a guide on how to setup the augmentation program on your device, as mentioned above the program has been tested with Ubuntu 18.04 & 16.04, but may work on other versions of Linux and possibly Windows.

## Clone the repository

Clone the [ALL Classifiers 2019](https://github.com/AMLResearchProject/ALL-Classifiers-2019 "ALL Classifiers 2019") repository from the [Peter Moss Acute Myleoid & Lymphoblastic AI Research Project](https://github.com/AMLResearchProject "Peter Moss Acute Myleoid & Lymphoblastic AI Research Project") Github Organization.

To clone the repository and install the ALL Classifiers 2019, make sure you have Git installed. Now navigate to the home directory on your device using terminal/commandline, and then use the following command.

```
  git clone https://github.com/AMLResearchProject/ALL-Classifiers-2019.git
```

Once you have used the command above you will see a directory called **ALL-Classifiers-2019** in your home directory.

```
ls
```

Using the ls command in your home directory should show you the following.

```
ALL-Classifiers-2019
```

Navigate to **ALL-Classifiers-2019/Augmentation** directory, this is your project root directory for this tutorial.

### Developer Forks

Developers from the Github community that would like to contribute to the development of this project should first create a fork, and clone that repository. For detailed information please view the [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") guide. You should pull the latest code from the development branch.

```
  $ git clone -b "0.2.0" https://github.com/AMLResearchProject/ALL-Classifiers-2019.git
```

The **-b "0.2.0"** parameter ensures you get the code from the latest master branch. Before using the below command please check our latest master branch in the button at the top of the project README.

## Install Requirements

Once you have used the command above you will see a directory called **ALL-Classifiers-2019** in the location you chose to download the repo to. In terminal, navigate to the **ALL-Classifiers-2019/Augmentation** and use the following command to install the required software for this program.

```
  sed -i 's/\r//' setup.sh
  sh Setup.sh
```

## Sort your dataset

The **ALL_IDB_1** dataset is the one used in this tutorial. In this dataset there were 59 negative and 49 positive. To make this even I removed 10 images from the negative dataset. From here I removed a further 10 images per class for testing further on in the tutorial and for the purpose of demos etc. In my case I ended up with 20 test images (10 pos/10 neg) and 49 images per class ready for augmentation. Place the original images that you wish to augment into the **Model/Data/0** & **Model/Data/1**. Using this program I was able to create a dataset of **1053** positive and **1053** negative augmented images.

You are now ready to move onto starting your Jupyter Notebook server or run the data augmentation locally.

&nbsp;

# Run locally

If you would like to run the program locally you can navigate to the Augmentation directory and use the following command:

```
  python3 Augmentation.py
```

# Run using Jupyter Notebook

You need to make sure you have Jupyter Notebook installed, you can use the following commands to install Jupyter, if you are unsure if you have it installed you can run the commands and it will tell you if you already have it installed and exit the download.

```
pip3 install --upgrade --force-reinstall --no-cache-dir jupyter
sudo apt install jupyter-notebook
```

Once you have completed the above, make sure you are in the **ALL-Classifiers-2019/Augmentation** directory and use the following commands to start your server, a URL will be shown in your terminal which will point to your Juupyter Notebook server with the required authentication details in the URL paramaters.

Below you would replace **###.###.#.##** with local IP address of the device you are going to be running the augmentation on.

```
  sudo jupyter notebook --ip ###.###.#.##
```

Using the URL provided to you in the above step, you should be able to access a copy of this directory hosted on your own device. From here you can navigate the project files and source code, you need to navigate to the **ALL-Classifiers-2019/Augmentation/Augmentation.ipynb** file on your own device which will take you to the second part of this tutorial. If you get stuck with anything in the above or following tutorial, please use the repository [issues](../issues "issues") and fill out the request information.

# Your augmented dataset

If you head to your **Model/Data/** directory you will notice the **Augmented** directory. Inside the augmented directory you will find **0** (negative) and **1** (postive) directories including resized copies of the original along with augmented copies.

Using data augmentation I was able to increase the dataset from **39** images per class to **1053** per class.

&nbsp;

# Contributing

The Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourages and welcomes code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Contributors

- [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") - [Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss") President & Lead Developer, Sabadell, Spain

&nbsp;

# Versioning

We use SemVer for versioning.

&nbsp;

# License

This project is licensed under the **MIT License** - see the [LICENSE](../LICENSE "LICENSE") file for details.

&nbsp;

# Bugs/Issues

We use the [repo issues](../issues "repo issues") to track bugs and general requests related to using this project.