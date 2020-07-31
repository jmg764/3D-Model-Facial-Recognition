# 3D-Model-Facial-Recognition

This is the final project for the CS-GY-6643 Computer Vision course at NYU Tandon School of Engineering.

## Introduction

**In this project, we explore the use of synthetic 2D images from 3D models for use in a 2D facial recognition system.** This may prove to be useful in situations where identification of an individual is warranted, but only a scant number of images of that person's face are available. Using those images, it is often possible to create a 3D facial reconstruction from which a substantial amount of 2D images can be generated for training. Additionally, whereas non-synthetic images of a person's face may exhibit varying lighting, facial expressions, and head poses, synthetic images can preserve greater consistency in terms of these conditions. Thus, the generation of a large, homogenous collection of synthetic face images for training can significantly improve the reliability of a given facial recognition model.

We generated synthetic images based on the 3D models provided in the (Florence 3D Faces dataset)[https://www.micc.unifi.it/resources/datasets/florence-3d-faces], and then implemented facial recognition using Histogram of Oriented Gradients (HOG) and k-Nearest Neighbors (k-NN) as described in [this tutorial](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/).

## Synthetic Image Generation



## HoG

