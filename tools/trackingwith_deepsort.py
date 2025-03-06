import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("yolov8n.pt")  # Use 'yolov8n.pt' for real-time processing

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open video file
video_path = "violation_13.mp4"  # Change to your video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter for output
out = cv2.VideoWriter("output_tracking.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

# Vehicle class IDs from COCO (car, truck, bus, motorcycle)
vehicle_classes = {2, 3, 5, 7}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Perform YOLO detection
    results = model(frame)

    detections = []
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if int(cls) in vehicle_classes:  # Only track vehicles
                x1, y1, x2, y2 = map(int, box)
                detections.append(([x1, y1, x2, y2], conf.item(), int(cls)))

    # Update DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if track.is_confirmed() and track.time_since_update == 0:
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            track_id = track.track_id

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Write to output video
    out.write(frame)
    cv2.imshow("Vehicle Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
