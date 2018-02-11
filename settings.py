import os
import sys
import re

# Model information
MODEL_DIR = 'models'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(MODEL_DIR, MODEL_NAME)

# Label information
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Post-process configuration
THRESHOLD = 0.23
BOX_COLOR = (0, 255, 0) # Support for lambda expression on score and class id to be added
BOX_THICKNESS = 2
FONT_FACE = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_COLOR = (0, 0, 0)
FONT_THICKNESS = 1

# I/O configuration
INPUT_DIR = 'input'
VIDEO_SET = 'sample_video'
VIDEO_NAME = 'sample.mp4'
VIDEO_PATH = os.path.join(INPUT_DIR, VIDEO_SET, VIDEO_NAME)
IMAGE_SET = 'sample_images'
IMAGE_PATHS = [ os.path.join(INPUT_DIR, IMAGE_SET, file)  \
    for file in os.listdir(os.path.join(INPUT_DIR, IMAGE_SET)) \
    if re.match('.*\.(jpg|jpeg|png)$', file, re.IGNORECASE)]
IMAGE_NUM = len(IMAGE_PATHS)
OUTPUT_DIR = os.path.join('output', IMAGE_SET)
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
