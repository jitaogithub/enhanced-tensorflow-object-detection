# Improved Object Detection on Tensorflow Object Detection API

## Introduction
The [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) has significantly simplified the development of an object-detecting application by providing well-built models with standardized I/O as well as easy-to-use but powerful utilities that saves time. However, some functionalities that come with the API are not satisfactory. This repository is the result of efforts to improve.

## Major improvements
- Rewrote the file I/O and added video support
- Rewrote the utility to visualize the inference results with huge performance gain
- Splitted only the necessary part from the bulky original repository
- Optimized the directory structure