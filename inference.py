import numpy as np
import tensorflow as tf
import cv2
import os
import sys
import time

from settings import *
from utils.tensorflow_authors import label_map_util
from utils.visualization import visualize_results


# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# -------- Run Tensorflow --------

# Load frozen graph
detection_graph = tf.Graph()
# with detection_graph.as_default():
#   od_graph_def = tf.GraphDef()
#   with tf.gfile.GFile(os.path.join(PATH_TO_CKPT, 'frozen_inference_graph.pb'), 'rb') as fid:
#     serialized_graph = fid.read()
#     od_graph_def.ParseFromString(serialized_graph)
#     tf.import_graph_def(od_graph_def, name='')
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Load checkpoint
    saver = tf.train.import_meta_graph(os.path.join(PATH_TO_CKPT, 'model.ckpt.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(PATH_TO_CKPT))

    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
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
      cv2.namedWindow('object-detection', cv2.WINDOW_NORMAL)
      cv2.resizeWindow('object-detection', (int(video_width), int(video_height)))
      
      (boxes, scores, classes, num) = (None, None, None, None)
      # Process frame-by-frame
      frame_count = 1
      while(video.isOpened()):
        ret, frame = video.read()        
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)   
        current_start = time.time()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame, axis=0)
        # Actual detection with timing.
        current_comp_start = time.time()
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        current_comp_end = time.time()

        frame = visualize_results(frame, category_index, boxes, scores, classes, video_height, video_width, 
          THRESHOLD, BOX_COLOR, BOX_THICKNESS, FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

        # Show the processed frame, press q to quit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
        cv2.imshow('object-detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

        current_end = time.time()
        total_comp_time += current_comp_end - current_comp_start
        total_time += current_end - current_comp_start
        frame_count += 1

      video.release()
      cv2.destroyWindow('object-detection')
      print('Total elapsed time {:.3f}s, on average {:.3f}fps\nTotal computation time {:.3f}s, on average {:.3f}s per frame'\
        .format(total_time, frame_count / total_time, total_comp_time, total_comp_time / frame_count))
        
    else:
      for image_path in IMAGE_PATHS:
        current_start = time.time()

        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)    
        image_height, image_width, dummy = frame.shape
        
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame, axis=0)
        
        # Actual detection with timing.
        current_comp_start = time.time()
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        current_comp_end = time.time()

        # Visualization
        frame = visualize_results(frame, category_index, boxes, scores, classes, image_height, image_width, 
          THRESHOLD, BOX_COLOR, BOX_THICKNESS, FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        
        # Output as file
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
        cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(image_path)), frame)

        current_end = time.time()
        total_comp_time += current_comp_end - current_comp_start
        total_time += current_end - current_comp_start
        print('{} consumed {:.3f}s in total, including {:.3f}s computation time'\
          .format(os.path.basename(image_path), current_end - current_start, current_comp_end - current_comp_start))
      
      print('\nTotal elapsed time {:.3f}s, on average {:.3f}s\nTotal computation time {:.3f}s, on average {:.3f}s per frame'\
        .format(total_time, total_time / IMAGE_NUM, total_comp_time, total_comp_time / IMAGE_NUM))
