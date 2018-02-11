import numpy as np
import tensorflow as tf
import cv2
import os
import sys
import time
import re

from utils.tensorflow_authors import label_map_util

# -------- Settings --------

# Model information
MODEL_DIR = 'models'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(MODEL_DIR, MODEL_NAME)

# Label information
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Post-process configuration
THRESHOLD = 0.22
BOX_THICKNESS = 2
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_COLOR = (0, 0, 0)
FONT_THICKNESS = 1
BACKGROUND_COLOR = lambda classId: (int(classId / NUM_CLASSES) % 255, 
  255 - int(classId / NUM_CLASSES) % 255, 127)

# I/O configuration
INPUT_DIR = 'input'
if len(sys.argv) > 1 and sys.argv[1] == '-v':
  VIDEO_SET = 'sample_video'
  VIDEO_NAME = 'sample.mp4'
  VIDEO_PATH = os.path.join(INPUT_DIR, VIDEO_SET, VIDEO_NAME)
else:
  IMAGE_SET = 'sample_images'
  IMAGE_PATHS = [ os.path.join(INPUT_DIR, IMAGE_SET, file)  \
    for file in os.listdir(os.path.join(INPUT_DIR, IMAGE_SET)) \
    if re.match('.*\.(jpg|jpeg|png)$', file, re.IGNORECASE)]
  IMAGE_NUM = len(IMAGE_PATHS)
  OUTPUT_DIR = os.path.join('output', IMAGE_SET)
  if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# -------- End of settings --------

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load frozen graph
detection_graph = tf.Graph()
# with detection_graph.as_default():
#   od_graph_def = tf.GraphDef()
#   with tf.gfile.GFile(os.path.join(PATH_TO_CKPT, 'frozen_inference_graph.pb'), 'rb') as fid:
#     serialized_graph = fid.read()
#     od_graph_def.ParseFromString(serialized_graph)
#     tf.import_graph_def(od_graph_def, name='')

# -------- Run Tensorflow --------
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Load checkpoint
    saver = tf.train.import_meta_graph(os.path.join(PATH_TO_CKPT, 'model.ckpt.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(PATH_TO_CKPT))

    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    total_comp_time = total_time = 0
    if len(sys.argv) > 1 and sys.argv[1] == '-v':
      # Open video file and get the dimensions
      video = cv2.VideoCapture(VIDEO_PATH)
      video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
      video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

      # Create and adjust window
      cv2.namedWindow('traffic-sign', cv2.WINDOW_NORMAL)
      cv2.resizeWindow('traffic-sign', (int(video_width), int(video_height)))
      
      # Process frame-by-frame
      frame_count = 1
      while(video.isOpened()):
        ret, frame = video.read()
        current_start = time.time()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame, axis=0)
        # Actual detection with timing.
        current_comp_start = time.time()
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        current_comp_end = time.time()

        # Visualization of the results of a detection.
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes) #.astype(np.int32)
        for box, score, category in zip(boxes, scores, classes):
          if score < THRESHOLD:
            continue

          text = category_index[category]['name'] + ': {:.0f}%'.format(score*100)
          box_denormalized = np.multiply(box, [video_height, video_width, video_height, video_width]).astype(np.int32)
          frame = cv2.rectangle(frame, (box_denormalized[1], box_denormalized[0]), 
            (box_denormalized[3], box_denormalized[2]), BACKGROUND_COLOR(category), BOX_THICKNESS, cv2.LINE_AA, 0)
          
          ((text_width, text_height), baseline)  = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, FONT_THICKNESS)
          frame = cv2.rectangle(frame, (box_denormalized[1], box_denormalized[0] - text_height - baseline), 
            (box_denormalized[1] + text_width, box_denormalized[0]), BACKGROUND_COLOR(category), -1, cv2.LINE_AA, 0)
          frame = cv2.putText(frame, text, (box_denormalized[1], box_denormalized[0] - baseline),
            FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # Show the processed frame, press q to quit
        cv2.imshow('traffic-sign', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

        current_end = time.time()
        total_comp_time += current_comp_end - current_comp_start
        total_time += current_end - current_comp_start
        # print('Frame {} consumed {:.3f}s in total, including {:.3f}s computation time'\
        #   .format(frame_count, current_end - current_start, current_comp_end - current_comp_start))
        frame_count += 1

      video.release()
      cv2.destroyWindow('traffic-sign')
      print('\nTotal elapsed time {:.3f}s, on average {:.3f}fps\nTotal computation time {:.3f}s, on average {:.3f}s per frame'\
        .format(total_time, frame_count / total_time, total_comp_time, total_comp_time / frame_count))
        
    else:
      for image_path in IMAGE_PATHS:
        current_start = time.time()

        frame = cv2.imread(image_path)
        video_height, video_width, dummy = frame.shape
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame, axis=0)
        # Actual detection with timing.
        current_comp_start = time.time()
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        current_comp_end = time.time()

        # Visualization of the results of a detection.
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes) #.astype(np.int32)
        for box, score, category in zip(boxes, scores, classes):
          if score < THRESHOLD:
            continue

          text = category_index[category]['name'] + ': {:.0f}%'.format(score*100)
          box_denormalized = np.multiply(box, [video_height, video_width, video_height, video_width]).astype(np.int32)
          frame = cv2.rectangle(frame, (box_denormalized[1], box_denormalized[0]), 
            (box_denormalized[3], box_denormalized[2]), BACKGROUND_COLOR(category), BOX_THICKNESS, cv2.LINE_AA, 0)
          
          ((text_width, text_height), baseline)  = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, FONT_THICKNESS)
          frame = cv2.rectangle(frame, (box_denormalized[1], box_denormalized[0] - text_height - baseline), 
            (box_denormalized[1] + text_width, box_denormalized[0]), BACKGROUND_COLOR(category), -1, cv2.LINE_AA, 0)
          frame = cv2.putText(frame, text, (box_denormalized[1], box_denormalized[0] - baseline),
            FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        
        # Output as file
        cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(image_path)), frame)

        current_end = time.time()
        total_comp_time += current_comp_end - current_comp_start
        total_time += current_end - current_comp_start
        print('{} consumed {:.3f}s in total, including {:.3f}s computation time'\
          .format(os.path.basename(image_path), current_end - current_start, current_comp_end - current_comp_start))
      print('\nTotal elapsed time {:.3f}s, on average {:.3f}s\nTotal computation time {:.3f}s, on average {:.3f}s per frame'\
        .format(total_time, total_time / IMAGE_NUM, total_comp_time, total_comp_time / IMAGE_NUM))
