# Improved Object Detection on Tensorflow Object Detection API

## Introduction
The [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) (the API) has significantly simplified the development of an object-detecting application by providing well-built, standardized, but cutting-edge models as well as easy-to-use but powerful utilities that saves time. However, some functionalities that come with the API are not satisfactory. This repository is the result of my efforts to improve.

## Major features
### Rewritten file I/O
In the original sample provided by the API, PIL is used to read image files, which was okay in terms of performance since the sample came with only 2 test images. I have discovered, though, that when it comes to processing a large number of images consecutively, the overhead, compared to the time actually consumed by computation is rather significant. Therefore, I made use of OpenCV instead of PIL, which has shortened the I/O time for a 1024x768 image from around 0.4s to around 0.01s on my machine.

### Video support
With OpenCV's VideoCapture method, I added some pieces of code that makes it possible to test a model on a video. To be honest,the experience is not good enough, for example the label of an object may seem like vibrating. It will be polished further in the future.

### Rewritten visualization utility
After adding the video support, a sample H.264 video of resolution 1024x768 would play at around 12fps with detection, which was not satisfactory enough. So I test the time consumed by the visualization utility that comes with the API and discovered a 0.3s-overhead to simply label the objects. So I rewrote this part using OpenCV and now the same piece of video plays at 15+fps.

### Splitted from the original repository
This is hardly a contribution. But I kind of hate git cloning a huge repository 99% of which makes no sense to me and have to run my code on that repository. This may also benefit those who wish to test on embedded systems such as Jetson of Nvidia that has limited space. So I splitted necessary dependencies (just the label map reading utility for now), and came up with an easier-to-understand directory structure. So it is now stand-alone and concise.

## Disclaimer
Utilities under `utils/tensorflow_authors` and the proto file `string_int_label_map.proto` are files splitted from the  [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) repository. I don't claim ownership of any of them. The authors of these contents reserve their legal rights.

## Configuration (on Linux)
There are basically 5 things (in the following order) that you need to get the ball rolling:
1. Python
2. Tensorflow (with its dependencies)
3. OpenCV-Python
4. Pre-trained models
5. Input (image or video)

### Python
Although there is no constraint on the version, this repository is written with Python 3.5.2 (which comes with Ubuntu 16.04). So Python 3.5.2+ is recommended.

### Tensorflow
It shouldn't be difficult for you install and configure tensorflow following the [official guide](https://www.tensorflow.org/install/). It is strongly recommended that you [install from source](https://www.tensorflow.org/install/install_sources). By compiling on your own computer, Tensorflow can fully utilize the features that increases its speed. 

### OpenCV-Python
OpenCV-Python is [OpenCV](https://opencv.org/)'s Python library. As is mentioned in the previous chapter, OpenCV is indispensable for performance. To install, Python 3 users should execute:
```bash
$ sudo pip3 install opencv-python
```  
Python 2 users should execute:
```bash
$ sudo pip install opencv-python
```

### Pre-trained models
One of the benefits brought by [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) is that many pre-trained models are available for downloading. Most of these models are state-of-the-art and well-defined so that you can experience the best performance that human have achieved without the pains of reading papers and building your own. You may even do [transfer learning](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md) easily based on these models. To download the models, visit [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).  
 
A label map is a map between model output and the actual class of the object. The label map differs if different training sets are used. Having determined your model to use, [download corresponding labels](https://github.com/tensorflow/models/tree/master/research/object_detection/data) for its training set.

By default settings, file `checkpoint` along with 3 files with `.ckpt` in the names are needed and should be extracted to `<repository>/models/<model-name>/`, while label maps are put in `<repository>/data/`. Specify the model name in `settings.py`.

### Input
As is mentioned, this application supports detecting objects in both images and videos. Images should be put into `<repository>/input/<image-set>/`, and the program will iterate through all the supported image files automatically. Videos should go to `<repository>/input/<video-set>/`. Edit `settings.py` to set the set names and video file name.

## Run inference
After all configures are set, we are good to go. In terminal direct to the repository. For Python 3, execute:
```bash
$ python3 inference.py
```
For Python 2, execute:
```bash
$ python inference.py
```
You may append argument `-v` to run the video

## Comparison among pre-trained models
### Test machine
- CPU: Intel Core i7-2600 @ 3.40GHz  
- Memory: 16GB @ 1333MHz  
- GPU: Integrated with CPU

### Sample clip
- Resolution: 1024x768
- Length: Only first 30s are tested
- Encoding: H.264  
- Bitrate: 9915 Kbps

### Result
|Model|Framerate (3-time average) (fps)|
|-|-|
|ssd_mobilenet_v1_coco|15.58|
|ssd_inception_v2_coco|5.76|
|faster_rcnn_inception_v2_coco|0.82|
|faster_rcnn_resnet101_coco|0.13|

### Conclusion
In the experiment, on the same machine and using the same video clip, two SSD models and two Faster-RCNN models all trained by the MSCOCO training set are compared. It is obvious that under this circumstance only the model SSD with MobileNet can perform inference on the video with acceptable framerate.  

Unfortunately, the fastest one is said to be the least accurate one (see [this](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)). Nevertheless, MSCOCO is a training set that contains a large numebr of general objects, which makes it extremely difficult to gain accuracy. It is reasonable to speculate that, were the training set to focus on a more specific area with less classes, it could be more accurate.

## Training
Coming soon.