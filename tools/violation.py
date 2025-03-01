import os
import cv2
import glob
import torch
import numpy as np
import argparse
import logging
import random
from tqdm import tqdm
from pathlib import Path
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network
from lanedet.core.lane import Lane
from ultralytics import YOLO  # For vehicle detection
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Set up logging to display debug messages.
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# Set device – change to 'cuda' if a GPU is available.
device = torch.device('cpu')

class VideoLaneDetection:
    def __init__(self, cfg, yolo_model_path='yolov8n.pt'):
        """Initialize lane detection model and YOLO vehicle detector."""
        self.cfg = cfg
        # Use the appropriate process (here we use infer_process as in your working code)
        self.processes = Process(cfg.infer_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(self.net, device_ids=range(1)).to(device)
        total_params = sum(param.numel() for param in self.net.parameters())
        logging.info(f"total number of params: {total_params}")
        self.net.eval()
        load_network(self.net, self.cfg.load_from)
        # Initialize YOLO model for vehicle detection.
        self.yolo_model = YOLO(yolo_model_path)

    def preprocess(self, frame):
        """Prepare image for lane detection model."""
        ori_img = frame.copy()
        # Crop and resize to match model input.
        img = cv2.resize(ori_img[self.cfg.cut_height:, :, :], (800, 288))
        img = img.astype(np.float32).transpose(2, 0, 1)  # [channels, height, width]
        img = torch.from_numpy(img).unsqueeze(0)  # Add batch dimension.
        data = {'img': img.to(device), 'lanes': []}
        data['ori_img'] = ori_img
        return data

    def detect_vehicles(self, frame):
        """Run YOLO inference to detect vehicles in a frame."""
        results = self.yolo_model(frame)[0]  # Get YOLO detections.
        vehicle_boxes = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            # Only keep specific vehicle classes (cars, buses, trucks, motorcycles).
            if int(cls) in [2, 3, 5, 7]:
                vehicle_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return vehicle_boxes

    def inference(self, data):
        """
        Run lane detection inference.
        If classification is enabled in the config, return a tuple: (lane_detections, lane_classes).
        Otherwise, return the lane detections.
        """
        with torch.no_grad():
            outputs = self.net(data)
            # Get lanes from the network.
            # The working Detect code does:
            #   lane_detection, lane_indx = list(self.net.module.get_lanes(outputs).values())
            # Here we assume the key 'lane_output' exists.
            lanes_dict = self.net.module.get_lanes(outputs)
            # For compatibility, assume lanes_dict has keys such as 'lane_output' and an index key.
            # (Adjust if your model returns a different dictionary.)
            if self.cfg.classification:
                # Assume the network outputs a dictionary where one value is lane detections
                # and another (e.g., lane index) is used for classification.
                # Here we mimic the working code:
                lane_detection, lane_indx = list(lanes_dict.values())
                lane_classes = self.get_lane_class(outputs, lane_indx)
                return lane_detection[0], lane_classes
            else:
                return lanes_dict['lane_output']

    def get_lane_class(self, predictions, lane_indx):
        """
        Compute lane classes from the network predictions.
        Expects predictions['category'] to be available.
        """
        score = F.softmax(predictions['category'], dim=1)
        y_pred = score.argmax(dim=1).squeeze()
        return y_pred[lane_indx].detach().cpu().numpy()

    def get_lane_color(self, lane_type):
        """
        Returns a BGR color based on the lane type.
        Customize the mapping as needed.
        """
        lane_color_map = {
            "solid": (0, 255, 255),   # Yellow
            "dashed": (255, 0, 255),  # Magenta
            "double": (255, 255, 0),  # Cyan
            "default": (0, 255, 0)    # Green as default
        }
        color = lane_color_map.get(lane_type, lane_color_map["default"])
        logging.debug(f"Lane type '{lane_type}' maps to color {color}")
        return color

    def get_lane_type_from_class(self, lane_class):
        """
        Map numeric lane class (from classification) to a string lane type.
        Adjust the mapping based on your training setup.
        For example, if:
            0 -> "solid"
            1 -> "dashed"
            2 -> "double"
        """
        mapping = {0: "solid", 1: "dashed", 2: "double"}
        return mapping.get(lane_class, "solid")

    def visualize(self, data, vehicle_boxes, lane_classes=None, out_file=None):
        """
        Visualize detected lanes and vehicles.
        
        - For each lane, if lane_classes are provided, use the classification result (mapped to a string)
          to choose the drawing color; otherwise, fall back to the lane attribute.
        - For each vehicle, compute an inner rectangle (80% width, 30% height, bottom-anchored)
          and flag a violation if any lane point falls within that inner rectangle.
        """
        img = data['ori_img'].copy()
        violation_lane_points = []  # Collect lane points (in pixel coordinates).
        
        lanes = data.get('lanes', [])
        if not isinstance(lanes, list):
            lanes = [lanes]
        
        # Process each lane. We assume lane_classes (if provided) aligns with lanes.
        for idx, lane in enumerate(lanes):
            if isinstance(lane, Lane):
                # If classification is enabled and a class exists for this lane, use it.
                if lane_classes is not None and idx < len(lane_classes):
                    lane_type = self.get_lane_type_from_class(lane_classes[idx])
                    logging.debug(f"Lane {idx}: using classified type '{lane_type}'")
                else:
                    lane_type = getattr(lane, 'lane_type', "solid")
                    logging.debug(f"Lane {idx}: using attribute type '{lane_type}'")
                color = self.get_lane_color(lane_type)
                pts = lane.points
                if pts is not None:
                    for x, y in pts:
                        if x > 0 and y > 0:
                            px = int(x * img.shape[1])
                            py = int(y * img.shape[0])
                            violation_lane_points.append((px, py))
                            cv2.circle(img, (px, py), 4, color, 2)
                else:
                    logging.warning("Lane.points is None.")
            elif isinstance(lane, list):
                # For lists of lanes, process each sub-lane.
                for j, sub_lane in enumerate(lane):
                    if isinstance(sub_lane, Lane):
                        # Determine lane type from classification if available.
                        # Here we assume sub-lanes follow the same order.
                        if lane_classes is not None and idx < len(lane_classes):
                            lane_type = self.get_lane_type_from_class(lane_classes[idx])
                            logging.debug(f"Sub-lane {idx}-{j}: using classified type '{lane_type}'")
                        else:
                            lane_type = getattr(sub_lane, 'lane_type', "solid")
                            logging.debug(f"Sub-lane {idx}-{j}: using attribute type '{lane_type}'")
                        color = self.get_lane_color(lane_type)
                        pts = sub_lane.points
                        if pts is not None:
                            for x, y in pts:
                                if x > 0 and y > 0:
                                    px = int(x * img.shape[1])
                                    py = int(y * img.shape[0])
                                    violation_lane_points.append((px, py))
                                    cv2.circle(img, (px, py), 4, color, 2)
                        else:
                            logging.warning("Sub-lane.points is None.")
            else:
                logging.warning("Unexpected lane format encountered.")
        
        # Process each detected vehicle box.
        for (x1, y1, x2, y2) in vehicle_boxes:
            box_width = x2 - x1
            box_height = y2 - y1
            # Compute inner rectangle: 80% width, 30% height, centered horizontally and bottom-anchored.
            inner_width = int(box_width * 0.8)
            inner_height = int(box_height * 0.3)
            inner_x1 = x1 + (box_width - inner_width) // 2
            inner_y2 = y2  # Bottom of the outer box.
            inner_y1 = inner_y2 - inner_height
            inner_x2 = inner_x1 + inner_width

            # Check if any lane point falls within the inner rectangle.
            violation = any(inner_x1 <= lx <= inner_x2 and inner_y1 <= ly <= inner_y2 
                            for (lx, ly) in violation_lane_points)

            if violation:
                outer_color = (0, 0, 255)  # Red for violation.
                label = "Violation"
                logging.debug(f"Vehicle box {(x1, y1, x2, y2)} flagged as violation.")
            else:
                outer_color = (0, 255, 0)  # Green otherwise.
                label = "Vehicle"
                logging.debug(f"Vehicle box {(x1, y1, x2, y2)} is clear.")

            # Draw the outer vehicle bounding box and label.
            cv2.rectangle(img, (x1, y1), (x2, y2), outer_color, 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, outer_color, 2)
            # Draw the inner rectangle in blue.
            cv2.rectangle(img, (inner_x1, inner_y1), (inner_x2, inner_y2), (255, 0, 0), 2)

        if out_file:
            cv2.imwrite(out_file, img)
            logging.info(f"Saved visualization to {out_file}")
        data['ori_img'] = img

    def process_video(self, input_path, output_path=None):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video {input_path}")

        video_writer = None
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.info("End of video or failed to read frame.")
                break

            try:
                data = self.preprocess(frame)
                if self.cfg.classification:
                    lanes, lane_classes = self.inference(data)
                    data['lanes'] = lanes
                else:
                    data['lanes'] = self.inference(data)
                    lane_classes = None
                vehicle_boxes = self.detect_vehicles(frame)
                self.visualize(data, vehicle_boxes, lane_classes=lane_classes, out_file=None)

                if video_writer:
                    video_writer.write(data['ori_img'])

                cv2.imshow("Lane & Vehicle Violation Detection", data['ori_img'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                logging.error(f"Error processing frame: {e}")
                continue

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
