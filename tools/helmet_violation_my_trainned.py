import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained model
model = YOLO("/home/prageeth/proj/LVLane/best.pt")  # Replace 'trainX' with your run (e.g., train6)

# Open the video
video_path = "/home/prageeth/proj/LVLane/violation_16.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction
    results = model(frame, conf=0.1)  # Lower conf to catch more objects (adjust as needed)

    # Draw results on frame
    annotated_frame = results[0].plot()  # Plots boxes and labels

    # Display the frame
    cv2.imshow("YOLO11x Prediction", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Cleanup
cv2.destroyAllWindows()