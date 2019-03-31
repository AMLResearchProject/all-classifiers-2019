# Acute Myeloid/Lymphoblastic Leukemia Classifier Data Augmentation Program
![Peter Moss Acute Myeloid/Lymphoblastic Leukemia Classifier Data Augmentation Program](Media/Images/banner.png) 

The AML/ALL Classifier Data Augmentation program applies filters to datasets and increases the amount of training / test data available to use. The program is part of the computer vision research and development for the Peter Moss Acute Myeloid/Lymphoblastic (AML/ALL) Leukemia AI Research Project. 

This page will provide general information, as well as a guide for installing and setting up the augmentation script. The augmentation program can currently be run by using a local configuration file, but you will soon be able to manage the configurations using the GeniSys UI also. 

&nbsp;

| Project  | Description | Author | 
| ------------- | ------------- | ------------- |
| [Data Augmentation Using Python](https://github.com/AMLResearchProject/AML-ALL-Classifiers/tree/master/Augmentation/Augmentation.ipynb "Data Augmentation Using Python") | A Python tutorial and Jupyter Notebook for applying filters to datasets to increase the amount of training / test data. | [Adam Milton-Barker](https://github.com/AdamMiltonBarker "Adam Milton-Barker") |

&nbsp;

# Research papers followed
Research papers used in this part of the project were shared by project team member, [Ho Leung Ng](https://github.com/holeung "Ho  Leung Ng"), Associate Professor of Biochemistry & Molecular Biophysics at Kansas State University.

| Paper  | Description | Link | 
| ------------- | ------------- | ------------- |
| Leukemia Blood Cell Image Classification Using Convolutional Neural Network | T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon | [Paper](http://www.ijcte.org/vol10/1198-H0012.pdf "Paper") |

&nbsp;

# Datasets
The [Acute Lymphoblastic Leukemia Image Database for Image Processing](https://homes.di.unimi.it/scotti/all/) dataset is used for this project. The dataset was created by [Fabio Scotti, Associate Professor Dipartimento di Informatica, Università degli Studi di Milano](https://homes.di.unimi.it/scotti/). Big thanks to Fabio for his research and time put in to creating the dataset and documentation, it is one of his personal projects. You will need to follow the steps outlined [here](https://homes.di.unimi.it/scotti/all/#download) to gain access to the dataset.

| Dataset  | Description | Link | 
| ------------- | ------------- | ------------- |
| Acute Lymphoblastic Leukemia Image Database for Image Processing | Created by [Fabio Scotti, Associate Professor Dipartimento di Informatica, Università degli Studi di Milano](https://homes.di.unimi.it/scotti/).  | [Download Dataset](https://homes.di.unimi.it/scotti/all/#download "Download Dataset") |

&nbsp;

# Data augmentation

![Acute Myeloid Leukemia Research Python Classifier](Media/Images/slides.png) 

I decided to use some augmentation proposals outlined in Leukemia Blood Cell Image Classification Using Convolutional Neural Network by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon. The augmentations I chose were grayscaling, histogram equalization, horizontal and vertical reflection, translation and gaussian blur to start with. 

In this dataset there were 49 negative and 59 positive. To make this even I removed 10 images from the positive dataset. From here I removed a further 10 images per class for testing further on in the tutorial and for the purpose of demos etc. In my case I ended up with 20 test images (10 pos/10 neg) and 39 images per class ready for augmentation. Place the original images that you wish to augment into the __Model/Data/0__ & __Model/Data/1__. Using this program I was able to create a dataset of __1053__ positive and __1053__ negative augmented images.

The full Python class that holds the functions mentioned below can be found in [Classes/Data.py](Classes/Data.py), The Data class is a wrapper class around releated functions provided in popular computer vision libraries including as OpenCV and Scipy.

&nbsp;

## Resizing

The first step is to resize the image this is done with the following function:

    def resize(self, filePath, savePath, show = False):
        
        ###############################################################
        #
        # Writes an image based on the filepath and the image provided. 
        #
        ###############################################################

        image = cv2.resize(cv2.imread(filePath), self.fixed)
        self.writeImage(savePath, image)
        self.filesMade += 1
        print("Resized image written to: " + savePath)
        
        if show is True:
            plt.imshow(image)
            plt.show()
            
        return image

&nbsp;

## Grayscaling

In general grayscaled images are not as complex as color images and result in a less complex model. In the paper the authors described using grayscaling to create more data easily. To create a greyscale copy of each image I wrapped the built in OpenCV function, [cv2.cvtColor()](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html). The created images will be saved to the relevant directories in the default configuration.

    def grayScale(self, image, grayPath, show = False):
        
        ###############################################################
        #
        # Writes a grayscale copy of the image to the filepath provided. 
        #
        ###############################################################
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.writeImage(grayPath, gray)
        self.filesMade += 1
        print("Grayscaled image written to: " + grayPath)
        
        if show is True:
            plt.imshow(gray)
            plt.show()
            
        return image, gray

&nbsp;

## Histogram Equalization

Histogram equalization is basically stretching the histogram horizontally on both sides, increasing the intensity/contrast. Histogram equalization is described in the paper to enhance the contrast. 

In the case of this dataset, it makes both the white and red blood cells more distinguishable. The created images will be saved to the relevant directories in the default configuration.

    def equalizeHist(self, gray, histPath, show = False):
        
        ###############################################################
        #
        # Writes histogram equalized copy of the image to the filepath 
        # provided. 
        #
        ###############################################################
        
        hist = cv2.equalizeHist(gray)
        self.writeImage(histPath, cv2.equalizeHist(gray))
        self.filesMade += 1
        print("Histogram equalized image written to: " + histPath)
        
        if show is True:
            plt.imshow(hist)
            plt.show()
            
        return hist

&nbsp;

## Reflection

Reflection is a way of increasing your dataset by creating a copy that is fliped on its X axis, and a copy that is flipped on its Y axis. The reflection function below uses the built in OpenCV function, cv2.flip, to flip the image on the mentioned axis.  The created images will be saved to the relevant directories in the default configuration.

    def reflection(self, image, horPath, verPath, show = False):
        
        ###############################################################
        #
        # Writes histogram equalized copy of the image to the filepath 
        # provided. 
        #
        ###############################################################
        
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

&nbsp;

## Gaussian Blur

Gaussian Blur is a popular technique used on images and is especially popular in the computer vision world. The function below uses the ndimage.gaussian_filter function. The created images will be saved to the relevant directories in the default configuration.

    def gaussian(self, filePath, gaussianPath, show = False):
        
        ###############################################################
        #
        # Writes gaussian blurred copy of the image to the filepath 
        # provided. 
        #
        ###############################################################
        
        gaussianBlur = ndimage.gaussian_filter(plt.imread(filePath), sigma=5.11)
        self.writeImage(gaussianPath, gaussianBlur)
        self.filesMade += 1
        print("Gaussian image written to: " + gaussianPath)

        if show is True:
            plt.imshow(gaussianBlur)
            plt.show()
            
        return gaussianBlur

&nbsp;
        
## Translation

Translation is a type of Affine Transformation and basically repositions the image within itself. The function below uses the cv2.warpAffine function. The created images will be saved to the relevant directories in the default configuration.

    def translate(self, image, translatedPath, show = False):

        y, x, c = image.shape
        translated = cv2.warpAffine(image, np.float32([[1, 0, 84], [0, 1, 56]]), (x, y))
        self.writeImage(translatedPath, translated)
        self.filesMade += 1
        print("Translated image written to: " + translatedPath)

        if show is True:
            plt.imshow(translated)
            plt.show()

        return translated

&nbsp;

## Rotation

Gaussian Blur is a popular technique used on images and is especially popular in the computer vision world. The function below uses the ndimage.gaussian_filter function. The created images will be saved to the relevant directories in the default configuration.
        
    def rotation(self, path, filePath, filename, show = False): 
        
        ###############################################################
        #
        # Writes rotated copies of the image to the filepath 
        # provided. 
        #
        ###############################################################
        
        img = Image.open(filePath)

        for i in range(0, 20):
            randDeg = random.randint(-180, 180)
            fullPath = os.path.join(path, str(randDeg) + '-' + str(i) + '-' + filename)

            try:
                if show is True:
                    img.rotate(randDeg, expand=True).resize((self.confs["Settings"]["ImgDims"], self.confs["Settings"]["ImgDims"])).save(fullPath).show()
                    self.filesMade += 1
                else:
                    img.rotate(randDeg, expand=True).resize((self.confs["Settings"]["ImgDims"], self.confs["Settings"]["ImgDims"])).save(fullPath)
                    self.filesMade += 1
                print("Rotated image written to: " + fullPath)
            except:
                print("File was not written! "+filename)

&nbsp;

# System Requirements
- Tested on Ubuntu 18.04 & 16.04
- [Tested with Python 3.5](https://www.python.org/downloads/release/python-350/ "Tested with Python 3.5")
- Requires PIP3
- Jupyter Notebook

&nbsp;

# Installation
Below is a guide on how to install the augmentation program on your device, as mentioned above the program has been tested with Ubuntu 18.04 & 16.04, but may work on other versions of Linux and possibly Windows.

## Clone the repository
First of all you should clone the [AML/ALL Detection System](https://github.com/AMLResearchProject/AML-ALL-Classifiers/ "AML/ALL Detection System") repo to your device. To do this can you navigate to the location you want to download to on your device using terminal  (cd Your/Download/Location), and then use the following commands:

```
  $ git clone https://github.com/AMLResearchProject/AML-ALL-Classifiers.git
```

Once you have used the command above you will see a directory called __AML-ALL-Classifiers__ in the location you chose to download the repo to. In terminal, navigate to the __AML-ALL-Classifiers/Augmentation__ and use the following command to install the required software for this program. 

```
 $ sh Setup.sh
```

If you have problems running the above program and have errors try run the following command before executing the shell script. You may be getting errors due to the shell script having been edited on Windows, the following command will clean the setup file.

```
 $ sed -i 's/\r//' setup.sh
 $ sh setup.sh
```

## Sort your dataset
The ALL IDB_1 dataset is the one used in this tutorial. In this dataset there were 49 negative and 59 positive. To make this even I removed 10 images from the positive dataset. From here I removed a further 10 images per class for testing further on in the tutorial and for the purpose of demos etc. In my case I ended up with 20 test images (10 pos/10 neg) and 39 images per class ready for augmentation. Place the original images that you wish to augment into the __Model/Data/0__ & __Model/Data/1__. Using this program I was able to create a dataset of __624__ positive and __624__ negative augmented images.

You are now ready to move onto starting your Jupyter Notebook server!

## Jupyter Notebook
You need to make sure you have Jupyter Notebook installed, you can use the following commands to install Jupyter, if you are unsure if you have it installed you can run the commands and it will tell you if you already have it installed and exit the download. 

```
  $ pip3 install --upgrade pip
  $ pip3 install jupyter
```
Once you have completed the above, make sure you are in the __AML-ALL-Classifiers/Augmentation__ directory and use the following commands to start your server, a URL will be shown in your terminal which will point to your Juupyter Notebook server with the required authentication details in the URL paramaters.

Below you would replace __###.###.#.##__ with local IP address of your device.

```
  $ jupyter notebook --ip ###.###.#.##
```

Using the URL provided to you in the above step, you should be able to access a copy of this directory hosted on your own device. From here you can navigate the project files and source code, you need to navigate to the __AML-ALL-Classifiers/Augmentation/Augmentation.ipynb__ file on your own device which will take you to the second part of this tutorial. If you get stuck with anything in the above or following tutorial, please use the repository [issues](https://github.com/AMLResearchProject/AML-ALL-Classifiers/issues "issues") and fill out the request information.

## Run locally
If you would like to run the program locally you can navigate to the Augmentation directory and use the following command:

```
  $ python3.5 Manual.py
```

&nbsp;

# Your augmented dataset
If you head to your __Model/Data/__ directory you will notice the augmented directory. Inside the augmented directory you will find 0 (negative) and 1 (postive) directories including resized copies of the original along with Grayscaled, Histogram Equalized, Reflected, Gaussian Blurred and rotated copies.

Using data augmentation I was able to increase the dataset from 39 images per class to 1053 per class. This dataset will be used in the [AML/ALL Movidius NCS Classifier](https://github.com/AMLResearchProject/AML-ALL-Classifiers/tree/master/Classifiers/Movidius/NCS).

&nbsp;

# Contributing
The Peter Moss Acute Myeloid/Lymphoblastic Leukemia AI Research project encourages and welcomes code contributions, bug fixes and enhancements from the community to the Github repositories. Please read the [CONTRIBUTING](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull request. You will also find information about our code of conduct on this page.

## Acute Lymphoblastic Leukemia Convolutional Neural Network Contributors

- [Adam Milton-Barker](https://github.com/AdamMiltonBarker "Adam Milton-Barker") - Bigfinite IoT Network Engineer & Intel Software Innovator, Barcelona, Spain

&nbsp;

# Versioning
We use SemVer for versioning. For the versions available, see [Releases](https://github.com/AMLResearchProject/AML-ALL-Classifiers/releases "Releases").

# License
This project is licensed under the **MIT License** - see the [LICENSE](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/LICENSE "LICENSE") file for details.

# Bugs/Issues
We use the [repo issues](https://github.com/AMLResearchProject/AML-ALL-Classifiers/issues "repo issues") to track bugs and general requests related to using this project.  

&nbsp;

# Project Author
Adam is a [BigFinite](https://www.bigfinite.com "BigFinite") IoT Network Engineer, part of the team that works on the core IoT software. In his spare time he is an [Intel Software Innovator](https://software.intel.com/en-us/intel-software-innovators/overview "Intel Software Innovator") in the fields of Internet of Things, Artificial Intelligence and Virtual Reality.

[![Adam Milton-Barker: BigFinte IoT Network Engineer & Intel® Software Innovator](../../Media/Images/Adam-Milton-Barker.jpg)](https://github.com/AdamMiltonBarker)