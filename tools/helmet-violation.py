from ultralytics import YOLO
import cv2

# Load YOLOv12 model
model = YOLO("yolo11n.pt")  # Downloads a stable version
# Open video
cap = cv2.VideoCapture("helmetV.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Process results
    for result in results:
        boxes = result.boxes.xyxy  # Bounding boxes
        classes = result.boxes.cls  # Class IDs
        for i, cls in enumerate(classes):
            label = model.names[int(cls)]
            if label == "person":  # Adjust based on your classes
                # Check for helmet (simplified logic; refine with training)
                x1, y1, x2, y2 = map(int, boxes[i])
                roi = frame[y1:y2, x1:x2]
                # Add helmet detection logic here (e.g., color analysis or secondary model)

    # Display frame (optional)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
