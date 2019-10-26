# Detecting Acute Lymphoblastic Leukemia Using Caffe\*, OpenVINO™ and Intel® Neural Compute Stick 2

## Part 1: Introduction to convolutional neural networks in Caffe

![Detecting Acute Lymphoblastic Leukemia Using Caffe*, OpenVINO™ and Intel® Neural Compute Stick 2: Part 1](https://www.PeterMossAmlAllResearch.com/media/images/repositories/Anh-Vo-Convolution.png)  
_IMAGE CREDIT: Anh Vo_

As part of my R&D for the Acute Myeloid & Lymphoblastic Leukemia (AML/ALL) AI Research Project, I am reviewing a selection of papers related to using Convolutional Neural Networks (CNN) for detecting AML/ALL. These papers share various ways of creating CNNs, and include useful information about the structure of the layers and the methods used which will help to reproduce the work outlined in the papers.

# Article Series

This is the first part of a series of articles that will take you through my experience building a custom classifier with Caffe\* that should be able to detect Acute Lymphoblastic Leukemia (ALL). I chose Caffe as I enjoyed working with it in a previous project, and I liked the intuitivity of defining the layers using prototxt files, however my R&D will include replicating both the augmentation script and the classifier using different languages and frameworks to compare results.

Previously I had followed the Leukemia Blood Cell Image Classification Using Convolutional Neural Network paper by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon paper to create a simple data augmentation program that would match the methods carried out in the paper. This was my first time translating a research paper into code, and although the resulting code is fairly basic in this case (mostly a wrapper around OpenCV\* functions), it was a cool experience.

# AML/ALL Classifiers Github Repo

Within the AML/ALL AI Research Project Github there is repository dedicated to open source classifiers, the [AML-ALL-Classifiers](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/CONTRIBUTING.md "AML-ALL-Classifiers") repo, in this directory the team and GitHub developer community we hope to attract will share tutorials that use various languages, frameworks and technologies to create convolutional neural networks.

# Introduction to convolutional neural networks in Caffe\*

In this technical article I will explain my experience of creating a custom convolutional neural network in Caffe using an architecture based on the [Acute Myeloid Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Myeloid Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper by Thanh.TTP, Giao N. Pham, Jin-Hyeok Park, Kwang-Seok Moon, Suk-Hwan Lee, and Ki-Ryong Kwon, and the [ALL_IDB1 dataset from Acute Lymphoblastic Leukemia Image Database for Image Processing](https://github.com/AMLResearchProject/AML-ALL-Classifiers/blob/master/CONTRIBUTING.md "ALL_IDB1 dataset from Acute Lymphoblastic Leukemia Image Database for Image Processing") dataset by Fabio Scotti, University of Milan.

In the augmentation paper, the authors mentioned that they were unable to reproduce a good accuracy using the augmented dataset, I will try to reproduce this and if I am unable to get good results will work on recreating the proposed architecture from the beginning of the augmentation paper.

“Our experiments were conducted on Matlab with 1188 images, 70% (831 images) of them for training and the remaining 30% (357 images) for testing our model. The slightly narrow architecture used dramatically failed to reach an appropriate accuracy when applied to this augmented dataset. Therefore, we have presented here a deeper CNN architecture and changed the size of the input volume in order to improve the accuracy rate of the recognition of leukemia (our proposed CNN model achieved 96.6%)”

**This tutorial can be found on the following platforms:**

- [Intel AI Developer Program Documentation](https://software.intel.com/en-us/articles/detecting-acute-lymphoblastic-leukemia-using-caffe-openvino-neural-compute-stick-2-part-1 "Intel AI Developer Program Documentation")
- [Linkedin Pulse](https://www.linkedin.com/pulse/detecting-acute-lymphoblastic-leukemia-using-caffe-2-milton-barker "Linkedin Pulse")

## Hardware:

• UP2 Development board (can be other Linux device)
• Neural Compute Stick 2 (Part of project can be run without)

# Operating System

- This project has been tested on Ubuntu 16.04
- This project has not been tested on any other operating systems

## Software:

• Caffe
• Intel OpenVino required if using NCS2

## Caffe Installation:

In my case I installed Caffe on an UP2, but as stated above this is not a requirement. During installation I ran into issues whilst following the [Caffe Ubuntu 16.04 installation guide](http://caffe.berkeleyvision.org/install_apt.html "Caffe Ubuntu 16.04 installation guide"), which led me to find the following tutorial.

[Follow this tutorial to install Caffe & PyCaffe Ubuntu 16.04](http://caffe.berkeleyvision.org/install_apt.html "Follow this article to install Caffe & PyCaffe Ubuntu 16.04") (Not tested in other OS). If the tutorial did not work for you, you will need to work out how to install Caffe and PyCaffe on your development machine and then come back to this tutorial, installing and debugging Caffe installation is out of the scope of this tutorial.

_If you are installing on an UP2 or similar this may take some time._

![Caffe Installation](https://www.PeterMossAmlAllResearch.com/media/images/repositories/Caffe-installation.jpg)  
_Figure 1. Caffe Installation_

Now that we have Caffe installed, I will explain a little bit about it. Caffe is another framework that we can use for building deep learning networks, including convolutional neural networks. I have used Caffe before with Neural Compute Stick (NCS) and YOLO for object detection, but have never really gone too deep into it the framework.

## Proposed Architecture

![Proposed Architecture](https://www.PeterMossAmlAllResearch.com/media/images/repositories/Proposed-Architecture.jpg)  
_Figure 1. Proposed Architecture_ ([Source](https://airccj.org/CSCP/vol7/csit77505.pdf "Source"))

In [Acute Myeloid Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Myeloid Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") the authors explain the methods they used to define their convolutional neural network’s architecture. Through the use of prototxt files used by Caffe, we can easily, and fairly visually, set up our layers based on the information found in the paper. For more information about convolutions you can check out [Caffe’s convolutions page](https://airccj.org/CSCP/vol7/csit77505.pdf "Caffe’s convolutions page") or for a more in depth explanation you can check out the information in [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/ "CS231n: Convolutional Neural Networks for Visual Recognition"). The remainder of this part of the article will focus on Caffe and the layers used in the paper, in the future I will cover convolutions in more detail.

As mentioned above, in the paper the authors share information about their architecture, they state how they use an architecture of using a 50 x 50 x 3 input layer (an image), 2 convolutional layers, a max pooling layer, a fully connected layer and softmax layer as an output. The convolutional layers and the max pooling layers are used for feature detection, while the fully connected and softmax layers are used for feature classification.

### Input Layer

The input layer is what feeds data into the network, in our case we were using an image that is 257px x 257px x 3px so our input size would need to be 257 x 257 x 3 (Height, width, depth), for this project a new augmented dataset will be created using the dimensions specified in the paper.

We can create a simple input layer using the following in a prototxt file: [allCNN.prototxt](https://www.linkedin.com/pulse/detecting-acute-lymphoblastic-leukemia-using-caffe-2-milton-barker-1f "allCNN.prototxt"), the additional dim, dim: 1, is the batch size meaning we will only send one image through the network per iteration, dim: 3/50/50 are the dimensions shown above which are the result of print(image.shape) (CV2).

```
  layer {
    name: "data"
    type: "Input"
    input_param { shape: { dim: 1 dim: 3 dim: 50 dim: 50 }}
  }
```

## Feature Detection Layers

### Convolutional Layers

![Convolutional Layers](https://www.PeterMossAmlAllResearch.com/media/images/repositories/Anh-Vo-Convolution.png)  
_Figure 2. Convolutional Layers_ ([Source](https://anhvnn.wordpress.com/2018/02/01/deep-learning-computer-vision-and-convolutional-neural-networks/ "Source"))

As mentioned in the paper, 2 convolution layers were used in the proposed architecture. The convolutional layers produce a feature map of a filter’s output activations. During convolution a filter is moved across the image and creates a new pixel in the output image.

We can define the layers as shown below. You will notice the bottom and top settings, these position this layer below the data (input) layer and top is itself conv1, num_outputs is the number of filters, kernel_size represents the size of the filters, stride represents how many pixels the kernel will move by, pad is padding added to the input image (required if we increase the size of the filter larger than the image), engine specifies which engine the model will use (CAFFE/CUDNN), weight_filler initializes the weights, we use the algorithm xavier which allows us to keep a stable signal, and finally bias_filter initializes the bias to 0, in the future I will cover more information about these parameters.

```
  layer {
    name: "conv1"
    type: "Convolution"
    convolution_param {
      num_output: 3
      kernel_size: 5.5
      stride: 1
      weight_filler {
        type: "Xavier"
      }
      bias_filler {
        type: "constant"
        value: 0
      }
    }
    bottom: "data"
    top: "conv2"
  }
```

### Pooling Layer

![Max Pooling Layer](https://www.PeterMossAmlAllResearch.com/media/images/repositories/Pooling.jpg)  
_Figure 3. Max Pooling Layer_ ([Source](https://cs231n.github.io/convolutional-networks/#pool "Source"))

The authors propose a pooling layer as the final layer in the feature extraction layers. Pooling layers help to reduce overfitting by reducing the size of the representation and the amount of activations/computation used by the network.

The authors state they use a 25 x 25 layer with a filter size of 2 and using a stride of 2. We can define the pooling layer using the allCNN.prototxt file with the following:

```
layer {
  name: "pool1"
  type: "Pooling"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
  bottom: "conv2"
  top: "fc"
}
```

## Feature Classification Layers

### Fully Connected Layer

![Fully Connected Layer](https://www.PeterMossAmlAllResearch.com/media/images/repositories/Fully-Connected-Layer.jpg)  
_Figure 4. Fully Connected Layer_ ([Source](https://medium.com/@eternalzer0dayx/demystifying-convolutional-neural-networks-ca17bdc75559 "Source"))

The proposed architecture for feature classification includes a 2 x 2 fully connected or inner product layer. The name fully connected layers means the fc layers are fully connected to the activations of the layers they follow. Fully connected layers used with a softmax output layer are used to classify the input image using the trained classes. For more information about fully connected layers visit this link.
The authors state a fully connected layer with 2 neurons. We can recreate this layer using the following in the allCNN.prototxt file.

```
layer {
  name: "fc"
  type: "InnerProduct"
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
  bottom: "pool"
  top: "fc"
}
```

### Softmax Layer

![Softmax Layer](https://www.PeterMossAmlAllResearch.com/media/images/repositories/Softmax.png)  
_Figure 5. Softmax Layer_ ([Source](https://towardsdatascience.com/deep-learning-concepts-part-1-ea0b14b234c8 "Source"))

The softmax layer proposed in the paper will output a probabilities distribution of an image being from each of the trained classes, each of the probabilities will add up to 1.0. For more information about softmax you can visit this link.

We can recreate the proposed softmax layer using allCNN.prototxt using the following:

```
layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc"
  top: "prob"
}
```

# Conclusion:

In [allCNN.prototxt](https://www.linkedin.com/pulse/detecting-acute-lymphoblastic-leukemia-using-caffe-2-milton-barker-1f "allCNN.prototxt") we should now have the architecture proposed in the [Acute Myeloid Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Myeloid Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper, it is not quite ready for training yet but we can use it to check if the network matches the one proposed in the paper, and visualize the network.

```
layer {
  name: "data"
  type: "Input"
  input_param { shape: { dim: 1 dim: 3 dim: 50 dim: 50 }}
  top: "data"
}
layer {
  name: "conv1"
  type: "Convolution"
  convolution_param {
    num_output: 30
    kernel_size: 5
    stride: 1
    pad: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
  bottom: "data"
  top: "conv1"
}
layer {
  name: "conv2"
  type: "Convolution"
  convolution_param {
    num_output: 30
    kernel_size: 5
    stride: 1
    pad: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
  bottom: "conv1"
  top: "conv2"
}
layer {
 name: "pool1"
 type: "Pooling"
 pooling_param {
   pool: MAX
   kernel_size: 2
   stride: 2
 }
 bottom: "conv2"
 top: "pool1"
}
layer {
 name: "fc"
 type: "InnerProduct"
 inner_product_param {
   num_output: 2
   weight_filler {
     type: "xavier"
   }
   bias_filler {
     type: "constant"
     value: 0
   }
 }
 bottom: "pool1"
 top: "fc"
}
layer {
 name: "prob"
 type: "Softmax"
 bottom: "fc"
 top: "prob"
}
```

Using the [Info.py](https://github.com/AdamMiltonBarker/AML-ALL-Classifiers/blob/master/Python/_Caffe/allCNN/Info.py "Info.py") script in the [AML / ALL Classifiers repository](https://github.com/AdamMiltonBarker/AML-ALL-Classifiers/blob/master/Python/_Caffe/allCNN/Info.py "AML / ALL Classifiers repository") we can check our networks match. First you need to clone the repository using the following commands:

```
git clone https://github.com/AMLResearchProject/AML-ALL-Classifiers.git
```

Then navigate to the allCNN directory:

```
cd Python/_Caffe/allCNN
```

Now you need to install any requirements:

```
sed -i 's/\r//' Setup.sh
sh Setup.sh
```

And finally we can check our network:

```
python3.5 Info.py NetworkInfo

```

The output of the script will include the following, showing that our network was created correctly:

```
I0309 16:11:30.786394 13920 layer_factory.hpp:77] Creating layer data
I0309 16:11:30.786425 13920 net.cpp:86] Creating Layer data
I0309 16:11:30.786440 13920 net.cpp:382] data -> data
I0309 16:11:30.786473 13920 net.cpp:124] Setting up data
I0309 16:11:30.786490 13920 net.cpp:131] Top shape: 1 3 50 50 (7500)
I0309 16:11:30.786499 13920 net.cpp:139] Memory required for data: 30000
I0309 16:11:30.786507 13920 layer_factory.hpp:77] Creating layer conv1
I0309 16:11:30.786526 13920 net.cpp:86] Creating Layer conv1
I0309 16:11:30.786538 13920 net.cpp:408] conv1 <- data
I0309 16:11:30.786551 13920 net.cpp:382] conv1 -> conv1
I0309 16:11:30.786855 13920 net.cpp:124] Setting up conv1
I0309 16:11:30.786872 13920 net.cpp:131] Top shape: 1 30 50 50 (75000)
I0309 16:11:30.786880 13920 net.cpp:139] Memory required for data: 330000
I0309 16:11:30.786901 13920 layer_factory.hpp:77] Creating layer conv2
I0309 16:11:30.786921 13920 net.cpp:86] Creating Layer conv2
I0309 16:11:30.786931 13920 net.cpp:408] conv2 <- conv1
I0309 16:11:30.786943 13920 net.cpp:382] conv2 -> conv2
I0309 16:11:30.787359 13920 net.cpp:124] Setting up conv2
I0309 16:11:30.787375 13920 net.cpp:131] Top shape: 1 30 50 50 (75000)
I0309 16:11:30.787384 13920 net.cpp:139] Memory required for data: 630000
I0309 16:11:30.787398 13920 layer_factory.hpp:77] Creating layer pool1
I0309 16:11:30.787411 13920 net.cpp:86] Creating Layer pool1
I0309 16:11:30.787420 13920 net.cpp:408] pool1 <- conv2
I0309 16:11:30.787432 13920 net.cpp:382] pool1 -> pool1
I0309 16:11:30.787456 13920 net.cpp:124] Setting up pool1
I0309 16:11:30.787470 13920 net.cpp:131] Top shape: 1 30 25 25 (18750)
I0309 16:11:30.787478 13920 net.cpp:139] Memory required for data: 705000
I0309 16:11:30.787487 13920 layer_factory.hpp:77] Creating layer fc
I0309 16:11:30.787501 13920 net.cpp:86] Creating Layer fc
I0309 16:11:30.787510 13920 net.cpp:408] fc <- pool1
I0309 16:11:30.787523 13920 net.cpp:382] fc -> fc
I0309 16:11:30.788055 13920 net.cpp:124] Setting up fc
I0309 16:11:30.788071 13920 net.cpp:131] Top shape: 1 2 (2)
I0309 16:11:30.788079 13920 net.cpp:139] Memory required for data: 705008
I0309 16:11:30.788110 13920 layer_factory.hpp:77] Creating layer prob
I0309 16:11:30.788125 13920 net.cpp:86] Creating Layer prob
I0309 16:11:30.788133 13920 net.cpp:408] prob <- fc
I0309 16:11:30.788144 13920 net.cpp:382] prob -> prob
I0309 16:11:30.788161 13920 net.cpp:124] Setting up prob
I0309 16:11:30.788175 13920 net.cpp:131] Top shape: 1 2 (2)
I0309 16:11:30.788183 13920 net.cpp:139] Memory required for data: 705016
I0309 16:11:30.788197 13920 net.cpp:202] prob does not need backward computation.
I0309 16:11:30.788205 13920 net.cpp:202] fc does not need backward computation.
I0309 16:11:30.788214 13920 net.cpp:202] pool1 does not need backward computation.
I0309 16:11:30.788223 13920 net.cpp:202] conv2 does not need backward computation.
I0309 16:11:30.788231 13920 net.cpp:202] conv1 does not need backward computation.
I0309 16:11:30.788240 13920 net.cpp:202] data does not need backward computation.
I0309 16:11:30.788249 13920 net.cpp:244] This network produces output prob
I0309 16:11:30.788262 13920 net.cpp:257] Network initialization done.
```

Finally we can visualize our network using the following command:

```
python3.5 /home/upsquared/caffe/python/draw_net.py allCNN.prototxt Model/allCNN.png
```

The above command will produce the following image:

![Network Architecture](https://www.PeterMossAmlAllResearch.com/media/images/repositories/allCNN.png)  
_Figure 6. allCNN Network Architecture_

To save our network we can use the following command:

```
python3.5 Info.py Save
```

This will save the network to the location Model/allCNN.caffemodel. In the next part of this series of articles I will create the training and validation datasets that we will use for this network.

Thanks to AML/ALL AI Research Project team members Amita Kapoor (Associate Professor @ Delhi University, New Dehli, India) and Ho Leung Ng (Kansas State University, Dept. Biochemistry & Molecular Biophysics) for their assistance with the article.

# References

UP Squared: https://up-board.org/upsquared/specifications/  
Neural Compute Stick: https://software.intel.com/es-es/neural-compute-stick  
Caffe Installation: https://gist.github.com/nikitametha/c54e1abecff7ab53896270509da80215  
Convolutions: http://aishack.in/tutorials/convolutions

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
