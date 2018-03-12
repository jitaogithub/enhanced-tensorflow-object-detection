import os
import re

# Model information
MODEL_DIR = 'models'
MODEL_NAME = 'ssd_mobilenet_v1_lisa_2018_03_04'
# MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
# MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(MODEL_DIR, MODEL_NAME)

# Label information
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 14

# Post-process configuration
THRESHOLD = 0.4
BOX_COLOR = (0, 255, 0) # Support for lambda expression on score and class id to be added
BOX_THICKNESS = 2
FONT_FACE = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_COLOR = (0, 0, 0)
FONT_THICKNESS = 1
MIN_BOX_HEIGHT = 0.012
MIN_BOX_WIDTH = 0.009


# I/O configuration
INPUT_DIR = 'input'

VIDEO_SET = 'sample_video'
VIDEO_NAME = 'sample.mp4'
VIDEO_PATH = os.path.join(INPUT_DIR, VIDEO_SET, VIDEO_NAME)

IMAGE_SET = 'sample_images'

if os.path.exists(os.path.join(INPUT_DIR, IMAGE_SET)):
    IMAGE_PATHS = [ os.path.join(INPUT_DIR, IMAGE_SET, file)  \
        for file in os.listdir(os.path.join(INPUT_DIR, IMAGE_SET)) \
        if re.match('.*\.(jpg|jpeg|png)$', file, re.IGNORECASE)]
    IMAGE_NUM = len(IMAGE_PATHS)
    OUTPUT_DIR = os.path.join('output', IMAGE_SET)
    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
