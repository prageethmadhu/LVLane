from pathlib import Path
import numpy as np
import torch
import cv2
import os
import os.path as osp
import argparse
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network
from lanedet.core.lane import Lane
from ultralytics import YOLO  # Import YOLO model from Ultralytics
import matplotlib.pyplot as plt
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

class VideoLaneDetection:
    def __init__(self, cfg, yolo_model_path='yolov8n.pt'):
        """Initialize lane detection model and YOLO vehicle detector."""
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)

        # Load Lane Detection Model
        self.net = build_net(self.cfg)
        self.net = torch.nn.DataParallel(self.net).to(device)
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

        # Load YOLO Model for Vehicle Detection
        self.yolo_model = YOLO(yolo_model_path)  # Load YOLOv8 model for vehicle detection

    def preprocess(self, frame):
        """Prepare image for lane detection model."""
        ori_img = frame.copy()
        img = cv2.resize(ori_img[self.cfg.cut_height:, :, :], (800, 288))  # Resize for lane model
        img = img.astype(np.float32).transpose(2, 0, 1)  # Convert to [channels, height, width]
        img = torch.from_numpy(img).unsqueeze(0)  # Add batch dimension
        data = {'img': img.to(device), 'lanes': [], 'ori_img': ori_img}
        return data

    def detect_vehicles(self, frame):
        """Run YOLO inference to detect vehicles in a frame."""
        results = self.yolo_model(frame)[0]  # Get YOLO detections
        vehicle_boxes = []

        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if int(cls) in [2, 3, 5, 7]:  # Only keep cars, buses, trucks, motorcycles
                vehicle_boxes.append((int(x1), int(y1), int(x2), int(y2)))

        return vehicle_boxes

    def inference(self, data):
        """Run lane detection inference."""
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.get_lanes(data)
        return data

    def visualize(self, data, vehicle_boxes, out_file=None):
        """Visualize detected lanes and vehicles on the frame.
        
        - Draws lane points as green dots.
        - For each vehicle, an inner rectangle is computed at the bottom of its detection box.
        - A violation is flagged (and drawn in red with label "Violation") only if any lane point
          falls within this inner rectangle. Otherwise, the vehicle is labeled "Vehicle" and drawn in green.
        """
        img = data['ori_img'].copy()  # Work on a copy of the original image

        # Collect lane points (convert normalized coordinates to pixels)
        lane_points = []
        for lane in data.get('lanes', []):
            if isinstance(lane, Lane):
                pts = lane.points
                if pts is not None:
                    for x, y in pts:
                        if x > 0 and y > 0:
                            lane_points.append((int(x * img.shape[1]), int(y * img.shape[0])))
            elif isinstance(lane, list):
                for l in lane:
                    if isinstance(l, Lane):
                        pts = l.points
                        if pts is not None:
                            for x, y in pts:
                                if x > 0 and y > 0:
                                    lane_points.append((int(x * img.shape[1]), int(y * img.shape[0])))

        # Draw lane points as green circles
        for (lx, ly) in lane_points:
            cv2.circle(img, (lx, ly), 4, (0, 255, 0), 2)

        # Process each vehicle detection box
        for (x1, y1, x2, y2) in vehicle_boxes:
            # Calculate outer box dimensions
            box_width = x2 - x1
            box_height = y2 - y1

            # Compute inner rectangle proportional to the outer box.
            # Here, inner rectangle width is 80% of outer width and height is 30% of outer height,
            # anchored to the bottom of the outer box.
            inner_width = int(box_width * 0.8)
            inner_height = int(box_height * 0.3)
            inner_x1 = x1 + (box_width - inner_width) // 2
            inner_y2 = y2  # bottom of the outer box
            inner_y1 = inner_y2 - inner_height
            inner_x2 = inner_x1 + inner_width

            # Check violation only using the inner rectangle area
            violation = any(inner_x1 <= lx <= inner_x2 and inner_y1 <= ly <= inner_y2 
                            for (lx, ly) in lane_points)

            if violation:
                outer_color = (0, 0, 255)  # Red outer box for violation
                label = "Violation"
            else:
                outer_color = (0, 255, 0)  # Green outer box otherwise
                label = "Vehicle"

            # Draw the outer bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), outer_color, 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, outer_color, 2)

            # Draw the inner rectangle (always drawn in blue)
            cv2.rectangle(img, (inner_x1, inner_y1), (inner_x2, inner_y2), (255, 0, 0), 2)

        # Save output if specified
        if out_file:
            cv2.imwrite(out_file, img)
            
        # Update the image in data for display/output
        data['ori_img'] = img

    def process_video(self, input_path, output_path=None):
        """Process input video, detect lanes and vehicles, and save the output."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video {input_path}")

        video_writer = None
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame.")
                break

            try:
                # Lane Detection
                data = self.preprocess(frame)
                data['lanes'] = self.inference(data)['lane_output']

                # Vehicle Detection
                vehicle_boxes = self.detect_vehicles(frame)

                # Visualization (this updates data['ori_img'])
                self.visualize(data, vehicle_boxes, out_file=None)

                # Write to output file if writer is initialized
                if video_writer:
                    video_writer.write(data['ori_img'])

                # Show video frame with detections
                cv2.imshow("Lane & Vehicle Detection", data['ori_img'])

                # Exit on pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue  # Skip to the next frame

        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--video', help='Path to the input video file')
    parser.add_argument('--output', help='Path to save the output video file')
    parser.add_argument('--load_from', type=str, default='best.pth', help='Path to the pretrained lane model')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt', help='Path to YOLOv8 model')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.load_from
    video_detector = VideoLaneDetection(cfg, args.yolo_model)
    video_detector.process_video(args.video, args.output)
