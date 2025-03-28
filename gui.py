import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load your YOLO model
model = YOLO("best.pt")

# Load labels
labels = [
    'Speed Limit 50', 'Speed Limit 100',
    'No Overtaking', 'Yield',
    'Stop', 'No Entry',
    'Danger Ahead', 'Road Work Ahead',
    'Pedestrian Crossing', 'Children Crossing'
]

# CONFIDENCE_THRESHOLD = 0.70
IOU_THRESHOLD = 0.2

def process_detections(frame, results):
    boxes, confidences, class_ids = [], [], []

    for det in results[0].boxes:
        conf = det.conf[0].item()
        if conf > confidence:
            x1, y1, x2, y2 = det.xyxy[0]
            cls = int(det.cls[0].item())

            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            confidences.append(float(conf))
            class_ids.append(cls)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence, IOU_THRESHOLD)

    if indices is not None and len(indices) > 0:
        for i in indices.flatten():
            x1, y1, x2, y2 = boxes[i]
            conf = confidences[i]
            cls = class_ids[i]
            label = labels[cls]
            conf_text = f'{conf:.2f}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} {conf_text}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Streamlit interface
st.title("YOLOv11 Real-Time Traffic Sign Detection")
run = st.checkbox('Start Camera')
confidence = st.number_input("Confidence", min_value=0.0, max_value=1.0, value=0.6, step=0.01, format="%.2f")

frame_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)  # Use 0 or 1 depending on your camera index

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame.")
            break

        results = model(frame)
        frame = process_detections(frame, results)

        # Convert BGR to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

        # Optional sleep to control frame rate
        time.sleep(0.003)

    cap.release()
