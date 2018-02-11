import cv2
import numpy as np



# Visualization of the results of a detection
def visualize_results(frame, category_index, boxes, scores, classes, frame_height, frame_width, 
    threshold=0.3, box_color=(0, 255, 0), box_thickness=2, font_face=cv2.FONT_HERSHEY_SIMPLEX, 
    font_scale=0.4, font_color=(0, 0, 0), font_thickness=1):
    
    # Pre-process input
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)
    
    for box, score, category in zip(boxes, scores, classes):
        if score < threshold or box[2] - box[0] < 0.04 or box[3] - box[1] < 0.032:
            continue
        
        # Generate label to put
        text = category_index[category]['name'] + ': {:.0f}%'.format(score*100)
        # Denormalized coordinates
        box_denormalized = np.multiply(box, [frame_height, frame_width, frame_height, frame_width]).astype(np.int32)
        # Draw the box
        frame = cv2.rectangle(frame, (box_denormalized[1], box_denormalized[0]), 
        (box_denormalized[3], box_denormalized[2]), box_color, box_thickness, cv2.LINE_AA, 0)
        
        # Calculate label position
        ((text_width, text_height), baseline)  = cv2.getTextSize(text, font_face, font_scale, font_thickness)
        # Draw label background
        frame = cv2.rectangle(frame, (box_denormalized[1], box_denormalized[0] - text_height - baseline), 
            (box_denormalized[1] + text_width, box_denormalized[0]), box_color, -1, cv2.LINE_AA, 0)
        # Put the label
        frame = cv2.putText(frame, text, (box_denormalized[1], box_denormalized[0] - baseline),
            font_face, font_scale, font_color, font_thickness, cv2.LINE_AA)
    
    return frame
