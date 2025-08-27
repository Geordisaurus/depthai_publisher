#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time

# Configuration
MODEL_PATH = "fire_smoke_detector.blob"  # Update this path to your downloaded .blob file
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_SIZE = 416

# Class names - update these based on your training data
# From your notebook, it looks like you were training on fire & smoke detection
CLASS_NAMES = ["fire", "smoke"]  # Update based on your data.yaml file

def create_pipeline():
    """Create DepthAI pipeline"""
    pipeline = dai.Pipeline()

    # Create ColorCamera node
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(INPUT_SIZE, INPUT_SIZE)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    # Create Neural Network node
    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setBlobPath(MODEL_PATH)
    detection_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    detection_nn.input.setBlocking(False)

    # Create output streams
    cam_rgb.preview.link(detection_nn.input)
    
    # Create output queues
    preview_out = pipeline.create(dai.node.XLinkOut)
    preview_out.setStreamName("rgb")
    cam_rgb.preview.link(preview_out.input)

    detection_out = pipeline.create(dai.node.XLinkOut)
    detection_out.setStreamName("detections")
    detection_nn.out.link(detection_out.input)

    return pipeline

def parse_yolo_output(output, img_width, img_height):
    """Parse YOLO output to get bounding boxes"""
    # This is a simplified parser - you might need to adjust based on your model's exact output format
    detections = []
    
    # YOLO output format: [batch, num_detections, 85] where 85 = 4 coords + 1 confidence + 80 classes
    # For your custom model, the last dimension will be 4 + 1 + num_classes
    for detection in output[0]:
        confidence = detection[4]
        if confidence > CONFIDENCE_THRESHOLD:
            # Get class scores
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            
            if class_confidence > CONFIDENCE_THRESHOLD:
                # Convert center coordinates to corner coordinates
                center_x, center_y, width, height = detection[0:4]
                
                # Convert to pixel coordinates
                x1 = int((center_x - width/2) * img_width)
                y1 = int((center_y - height/2) * img_height)
                x2 = int((center_x + width/2) * img_width)
                y2 = int((center_y + height/2) * img_height)
                
                detections.append({
                    'class_id': int(class_id),
                    'confidence': float(class_confidence),
                    'box': [x1, y1, x2, y2]
                })
    
    return detections

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame"""
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        class_id = detection['class_id']
        confidence = detection['confidence']
        
        # Get class name
        if class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_id]
        else:
            class_name = f"Class_{class_id}"
        
        # Draw bounding box
        color = (0, 255, 0) if class_name == "fire" else (0, 0, 255)  # Green for fire, red for smoke
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame

def main():
    """Main function"""
    print("Starting fire and smoke detection...")
    
    # Create pipeline and connect to device
    pipeline = create_pipeline()
    
    with dai.Device(pipeline) as device:
        print(f"Connected to device: {device.getDeviceName()}")
        
        # Get output queues
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_det = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
        frame_count = 0
        fps_start = time.time()
        
        while True:
            # Get RGB frame
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()
            
            # Get detections
            in_det = q_det.tryGet()
            if in_det is not None:
                # Parse detections
                detections = parse_yolo_output(in_det.getFirstLayerFp16(), INPUT_SIZE, INPUT_SIZE)
                
                # Draw detections on frame
                frame = draw_detections(frame, detections)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                print(f"FPS: {fps:.1f}")
                fps_start = time.time()
            
            # Show frame
            cv2.imshow("Fire & Smoke Detection", frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) == ord('q'):
                break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
