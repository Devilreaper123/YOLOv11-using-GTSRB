import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Load YOLO model
model = YOLO("best.pt")

# Label list
labels = [
    'Speed Limit 50', 'Speed Limit 100',
    'No Overtaking', 'Yield',
    'Stop', 'No Entry',
    'Danger Ahead', 'Road Work Ahead',
    'Pedestrian Crossing', 'Children Crossing'
]

# Title & Confidence Slider
st.title("🚦 YOLOv8 Traffic Sign Detection")
confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.6)

# File uploader
file_type = st.radio("Choose file type", ["Image", "Video"])

uploaded_file = st.file_uploader("Upload a file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

def draw_detections(frame, results, conf_threshold):
    boxes = results[0].boxes
    cropped_images = []  # Store cropped images for Streamlit display

    for det in boxes:
        conf = det.conf[0].item()
        if conf > conf_threshold:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cls = int(det.cls[0].item())
            label = labels[cls] if cls < len(labels) else f"Class {cls}"
            
            # Draw rectangle on the frame (optional)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Crop the detected region
            cropped_image = frame[y1:y2, x1:x2]
            cropped_images.append((cropped_image, label, conf))

    return frame, cropped_images

if uploaded_file:
    if file_type == "Image":
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        results = model(image)
        boxes = results[0].boxes

        detected_labels = []
        cropped_images = []

        for det in boxes:
            conf = det.conf[0].item()
            if conf > confidence:
                cls = int(det.cls[0].item())
                label = labels[cls] if cls < len(labels) else f"Class {cls}"
                detected_labels.append(f"{label} ({conf:.2f})")

        # Show detected labels in Streamlit
        if detected_labels:
            st.subheader("🧠 Detected Labels:")
            for lbl in detected_labels:
                st.markdown(f"- {lbl}")
        else:
            st.info("No traffic signs detected with the given confidence threshold.")

        # Show image (optional, unmodified)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Input Image", channels="RGB")

    elif file_type == "Video":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            frame, cropped_images = draw_detections(frame, results, confidence)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            stframe.image(frame, channels="RGB")

            # Display cropped images in Streamlit below the video frame
            if cropped_images:
                for cropped_image, label, conf in cropped_images:
                    st.subheader(f"Detected: {label} ({conf:.2f})")
                    st.image(cropped_image, channels="RGB")

        cap.release()
        os.remove(tfile.name)
