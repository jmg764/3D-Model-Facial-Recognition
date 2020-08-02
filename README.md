# 3D-Model-Facial-Recognition

This is the final project for the CS-GY-6643 Computer Vision course at NYU Tandon School of Engineering.

## Introduction

**In this project, we explore the use of synthetic 2D images from 3D models for use in a 2D facial recognition system.** This may prove to be useful in situations where identification of an individual is warranted, but only a scant number of images of that person's face are available. Using those images, it is often possible to create a 3D facial reconstruction from which a substantial amount of 2D images can be generated for training. Additionally, whereas non-synthetic images of a person's face may exhibit varying lighting, facial expressions, and head poses, synthetic images can preserve greater consistency in terms of these conditions. Thus, the generation of a large, homogenous collection of synthetic face images for training can significantly improve the reliability of a given facial recognition model.

We generated synthetic images based on the 3D models provided in the [Florence 3D Faces dataset](https://www.micc.unifi.it/resources/datasets/florence-3d-faces), and then implemented facial recognition through deep metric learning using Histogram of Oriented Gradients (HOG) and k-Nearest Neighbors (k-NN). Facial recognition implementation was based off of [this tutorial](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/), and makes use of the [dlib library](http://dlib.net) and [face_recognition module](https://github.com/ageitgey/face_recognition).

## The Dataset

Downloading the Florence 3D Faces dataset requires submission of a license agreement (information can be found [here](http://www.micc.unifi.it/vim/3dfaces-dataset/index.html#!prettyPhoto)). Once downloaded, the dataset should exist in a directory called ```Florence Face``` and contain subdirectories for each individual with the following structure: 

```bash
├── subject_01
│   └── Model
│       ├── frontal1
│       │   ├── obj
│       │   └── vrml
│       ├── frontal2
│       │   ├── obj
│       │   └── vrml
│       ├── sidel
│       │   ├── obj
│       │   └── vrml
│       └── sider
│           ├── obj
│           └── vrml
```
The 3D models for this project are the ```.obj``` files in ```../frontal1/obj``` and ```../frontal2/obj```. The following is an example of one such model opened in Xcode 11.6: 

<img src="3d_model_sample.gif"  alt="drawing" width="300"/>

Additionally, the images used for testing our model are the ```.bmp``` files in each ```../frontal1/obj``` (only one side of a subject's face is used):

<img src="test-data/test1.png"  alt="drawing" width="225"/>


## Synthetic Code Generation

The code for synthetic image generation can be found in [Synthetic Image Genration.ipynb](https://github.com/jmg764/3D-Model-Facial-Recognition/blob/master/Synthetic%20Image%20Generation.ipynb). It requires installation of the packages ```pywavefront``` and ```pyglet```. 

The number of snapshots taken per 3D model is set to 50 by default, but can be changed by altering the value of ```MAX_SNAPSHOTS``` in Synthetic Image Genration.ipynb.

Each time a snapshot of a subject is taken, it undergoes a random transformation dictated by the following, thereby creating a variety of angles and positions for our training data:

```python 
if snapshotsTaken < MAX_SNAPSHOTS and transformations is None:
        xOffset = random.uniform(-100.0, 100.0)
        yOffset = random.uniform(-100.0, 100.0)
        zOffset = random.uniform(-500.0, -300.0)
        xRotate = random.uniform(-50.0, 50.0)
        yRotate = random.uniform(-5.0, 5.0)
        transformations = (xOffset, yOffset, zOffset, xRotate, yRotate)
```

Synthetic images are saved in a file called ```synthetic_training_data``` which is organized into subdirectories corresponding to each subject. Here is an example of one of the snapshots created:

<img src="synthetic_image_example.png"  alt="drawing" width="225"/>

## Deep Metric Learning

Face recognition through deep metric learning involves the use of a neural network to output a real-valued feature vector, or embedding, which is used to quantify a given face. The network used in this project is based on the ResNet-34 architecture described [here](https://arxiv.org/abs/1512.03385), but with a few layers removed and the number of filters per layer reduced by half. It was already trained on a dataset of approximately 3 million images, and obtained a 99.38% accuracy on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) benchmark. This means that, given two images, it correctly predicts if the images are of the same person 99.38% of the time. In this project, the network is further trained on the Florence 3D Faces dataset.

**Creation of the embeddings used for training involves facial detection, affine transformation, and encoding faces, as detailed below: **

### 1. Facial Detection with HOG

Before we can go about recognizing faces, we must first locate them in our images. In general, any sort of object detection requires comparison between a known image and the image in question to see if a pattern or feature match exists. It is tempting to compare pixels directly, but very dark and very light images of the same object will have completely different pixel values. Histogram of Oriented Gradients (HOG) is an object detection method that solves this problem by only considering the direction that brightness changes (gradient orientation) in a particular region of an image. This captures the major features of an image regardless of image brightness. Comparison of a given image with a HOG face pattern generated from many images can aid in facial detection as shown in the following image obtained from [this article](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78):

<img src="hog_example.png"  alt="drawing" width="550"/>

### 2. Affine Transformation

Now that faces are isolated in our images, each image needs to be warped so that facial features are in the same location for each image. This makes it easier for our neural network to compare faces later on. Once facial features are identified through [facial landmark estimation](http://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf), an affine transformation is used to accomplish this warping:

<img src="face_transformation_example.png"  alt="drawing" width="750"/>

### 3. Encoding Faces

The simplest approach to facial recognition would be to compare an unknown face with every image that is already labeled. If there is a large number of images, however, this can take an unnecessarily long time. A faster way would be to use a few basic measurements from each face as a basis for comparison. Given an unknown face, the goal would be to find the labeled face with the closest measurements.

But which measurements should we consider? It turns out that the measurements that seem obvious to us, such as eye color, nose length, and ear size, aren't necessarily valuable measurements to a computer looking at individual pixels of an image. The most accurate approach is to use deep learning to identify the parts of a face that are important to measure. This is the idea behind "training" a neural network for facial recognition. Here, we will be generating 128 measurements for each face in the form of a vector (in other words, a *128-d embedding*).

Training involves a "triplet training step" in which the network creates embeddings for three unique face images –– two of which are the same person. The network is tweaked slightly so that the measurements it generates for the two images of the same person are closer via distance metric than those for the image of the other person:

<img src="triplet_training_example.png"  alt="drawing" width="550"/>
