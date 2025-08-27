#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time

# Configuration
MODEL_PATH = "/Users/geordihills/.cache/blobconverter/best_openvino_2022.1_6shave.blob"
CONFIDENCE_THRESHOLD = 0.3  # Start lower to see more detections
NMS_THRESHOLD = 0.4
INPUT_SIZE = 416

# Class names - based on your fire & smoke detection training
CLASS_NAMES = ["fire", "smoke"]

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
    detections = []
    
    # Get the raw output data
    if hasattr(output, 'getFirstLayerFp16'):
        data = np.array(output.getFirstLayerFp16()).astype(np.float32)
    elif hasattr(output, 'getLayerFp16'):
        data = np.array(output.getLayerFp16(output.getAllLayerNames()[0])).astype(np.float32)
    else:
        print("Could not get output data from model")
        return detections
    
    print(f"Model output shape: {data.shape}")  # Debug info
    
    # YOLOv5 output format is typically [1, 25200, 85] for COCO (80 classes)
    # For your custom model with 2 classes, it should be [1, 25200, 7] (4 coords + 1 conf + 2 classes)
    if len(data.shape) == 3:
        # Remove batch dimension if present
        data = data[0]
    
    if len(data.shape) != 2:
        print(f"Unexpected output shape: {data.shape}")
        return detections
    
    # Each detection: [center_x, center_y, width, height, confidence, class1_prob, class2_prob, ...]
    for detection in data:
        if len(detection) < 5:
            continue
            
        center_x, center_y, width, height, obj_conf = detection[0:5]
        
        # Skip low confidence detections
        if obj_conf < CONFIDENCE_THRESHOLD:
            continue
        
        # Get class probabilities
        if len(detection) > 5:
            class_probs = detection[5:]
            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id] * obj_conf  # Multiply by objectness confidence
        else:
            # If no class probabilities, assume single class
            class_id = 0
            class_conf = obj_conf
        
        # Skip if final confidence is too low
        if class_conf < CONFIDENCE_THRESHOLD:
            continue
        
        # Convert normalized coordinates to pixel coordinates
        # YOLOv5 outputs normalized coordinates (0-1)
        x1 = int((center_x - width/2) * img_width)
        y1 = int((center_y - height/2) * img_height)
        x2 = int((center_x + width/2) * img_width)
        y2 = int((center_y + height/2) * img_height)
        
        # Clamp coordinates to image bounds
        x1 = max(0, min(x1, img_width-1))
        y1 = max(0, min(y1, img_height-1))
        x2 = max(0, min(x2, img_width-1))
        y2 = max(0, min(y2, img_height-1))
        
        detections.append({
            'class_id': int(class_id),
            'confidence': float(class_conf),
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
                detections = parse_yolo_output(in_det, INPUT_SIZE, INPUT_SIZE)
                
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
